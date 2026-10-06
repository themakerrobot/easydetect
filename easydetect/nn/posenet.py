# Apache-2.0
"""A small top-down keypoint network: one person crop in, 17 COCO keypoints out.

The backbone is D-FINE's HGNetv2; the head reads each keypoint's position as
two classifications, one over the crop's columns and one over its rows, at
twice the pixel resolution (the SimCC formulation: Li et al., "SimCC: a Simple
Coordinate Classification Perspective for Human Pose Estimation", ECCV 2022).
No heatmap upsampling is needed, so the head stays cheap at stride 16, and a
keypoint's confidence is how peaked its two distributions are.

Written for easydetect from the paper; trained by tools/train_pose.py on COCO
2017 keypoints only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pose import INPUT, KEYPOINT_NAMES, SIGMA, SPLIT
from .hgnetv2 import HGNetv2

NUM_KEYPOINTS = len(KEYPOINT_NAMES)

#: Per-size settings: the backbone, and which of its stages feed the head.
#: ``use_lab`` as the detector of that size has it, so its weights carry over.
SIZE_CFG = {
    "s": {"backbone": "B0", "channels": (512, 1024), "use_lab": True},    # dfine-s
    "m": {"backbone": "B2", "channels": (768, 1536), "use_lab": True},    # dfine-m
    "l": {"backbone": "B4", "channels": (1024, 2048), "use_lab": False},  # dfine-l
}


class ConvBN(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 1):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU())


class KeypointMixer(nn.Module):
    """One pre-norm transformer layer over the 17 keypoint tokens: an elbow
    is easier to place knowing where the shoulder and wrist are."""

    def __init__(self, dim: int, heads: int = 4, ff: int = 512):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.SiLU(), nn.Linear(ff, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads) \
            .permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d // self.heads))
        y = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(y)
        return x + self.ff(self.norm2(x))


class PoseNet(nn.Module):
    """``PoseNet(size)(crops)`` with crops ``(B, 3, 256, 192)`` RGB 0-255 →
    ``(x_logits (B, 17, 384), y_logits (B, 17, 512))``."""

    def __init__(self, size: str = "s", num_keypoints: int = NUM_KEYPOINTS,
                 pretrained_backbone: bool = False, dim: int = 256):
        super().__init__()
        if size not in SIZE_CFG:
            raise ValueError(f"size must be one of {tuple(SIZE_CFG)}")
        cfg = SIZE_CFG[size]
        self.size = size
        self.num_keypoints = num_keypoints
        self.backbone = HGNetv2(cfg["backbone"], use_lab=cfg["use_lab"], return_idx=[2, 3],
                                freeze_at=-1,
                                freeze_norm=False, pretrained=pretrained_backbone)
        c4, c5 = cfg["channels"]
        self.lateral4 = ConvBN(c4, dim)
        self.lateral5 = ConvBN(c5, dim)
        self.fuse = nn.Sequential(ConvBN(dim, dim, 3), nn.Conv2d(dim, num_keypoints, 3, padding=1))
        h, w = INPUT
        cells = (h // 16) * (w // 16)
        self.embed = nn.Sequential(nn.Linear(cells, dim), nn.LayerNorm(dim))
        self.keypoint_embed = nn.Parameter(torch.zeros(1, num_keypoints, dim))
        self.mixer = KeypointMixer(dim)
        self.norm = nn.LayerNorm(dim)
        self.x_head = nn.Linear(dim, w * SPLIT)
        self.y_head = nn.Linear(dim, h * SPLIT)
        nn.init.trunc_normal_(self.keypoint_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p4, p5 = self.backbone(x / 255.0)  # D-FINE's backbone reads RGB in 0-1
        f = self.lateral4(p4) + F.interpolate(self.lateral5(p5), size=p4.shape[-2:],
                                              mode="nearest")
        maps = self.fuse(f).flatten(2)  # (B, K, H/16 * W/16): one map per keypoint
        tokens = self.mixer(self.embed(maps) + self.keypoint_embed)
        tokens = self.norm(tokens)
        return self.x_head(tokens), self.y_head(tokens)

    def freeze(self, stages: int) -> None:
        """Stop training the stem and the first ``stages`` backbone stages: their
        low-level features come from the detector already, and on a CPU skipping
        their backward pass makes a step about half again as fast."""
        self.frozen = stages
        for module in self._frozen():
            for p in module.parameters():
                p.requires_grad_(False)
        self.train(self.training)

    def _frozen(self) -> list[nn.Module]:
        n = getattr(self, "frozen", 0)
        return [self.backbone.stem, *self.backbone.stages[:n]] if n else []

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self._frozen():
            module.eval()  # frozen means the batch-norm statistics too
        return self

    def load_detector_backbone(self, state_dict: dict) -> int:
        """Start from a D-FINE checkpoint's backbone (COCO-trained, so it already
        knows people); returns how many tensors were taken."""
        mine = self.backbone.state_dict()
        taken = {k[len("backbone."):]: v for k, v in state_dict.items()
                 if k.startswith("backbone.") and k[len("backbone."):] in mine
                 and mine[k[len("backbone."):]].shape == v.shape}
        mine.update(taken)
        self.backbone.load_state_dict(mine)
        return len(taken)


def simcc_targets(xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keypoints in crop pixels ``(B, K, 2)`` → Gaussian target distributions
    over the x and y bins, each summing to one."""
    h, w = INPUT
    xs = torch.arange(w * SPLIT, device=xy.device, dtype=xy.dtype)
    ys = torch.arange(h * SPLIT, device=xy.device, dtype=xy.dtype)
    tx = torch.exp(-((xs - xy[..., :1] * SPLIT) ** 2) / (2 * SIGMA[1] ** 2))
    ty = torch.exp(-((ys - xy[..., 1:] * SPLIT) ** 2) / (2 * SIGMA[0] ** 2))
    tx = tx / tx.sum(-1, keepdim=True).clamp_min(1e-12)
    ty = ty / ty.sum(-1, keepdim=True).clamp_min(1e-12)
    return tx, ty


def simcc_loss(x_logits: torch.Tensor, y_logits: torch.Tensor, xy: torch.Tensor,
               weight: torch.Tensor) -> torch.Tensor:
    """KL divergence from the Gaussian targets, over the keypoints that are labelled
    and inside the crop (``weight`` 1) — the others teach nothing."""
    tx, ty = simcc_targets(xy)
    kl_x = (tx * (tx.clamp_min(1e-12).log() - F.log_softmax(x_logits.float(), -1))).sum(-1)
    kl_y = (ty * (ty.clamp_min(1e-12).log() - F.log_softmax(y_logits.float(), -1))).sum(-1)
    return ((kl_x + kl_y) * weight).sum() / weight.sum().clamp_min(1.0)
