# Apache-2.0
"""A frozen image model's features as a training-only teacher:
``train(extra_loss=FeatureTeacher())``.

The encoder's top feature map (stride 32, after its attention layer) learns,
through a linear projection used only in training, to point the same way
(cosine) as the patch features of DINOv2, resized to the same grid. The
deployed network is unchanged: the projection is not part of it. The idea is
RT-DETRv4's (arXiv:2510.25257), with DINOv2 (Apache-2.0) as the teacher.

Needs ``pip install timm``; the weights come from Hugging Face on first use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_TEACHER = "vit_small_patch14_dinov2.lvd142m"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class FeatureTeacher:
    """The extra loss; the trainer calls ``bind(net)`` once, then it once a batch."""

    def __init__(self, model: str = DEFAULT_TEACHER, weight: float = 1.0, size: int = 224):
        try:
            import timm
        except ImportError as exc:
            raise ImportError("FeatureTeacher needs timm: pip install timm") from exc
        self.teacher = timm.create_model(model, pretrained=True, num_classes=0,
                                         dynamic_img_size=True).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.weight, self.size = float(weight), int(size)
        self.patch = self.teacher.patch_embed.patch_size[0]
        self.proj: nn.Module | None = None
        self._seen: dict[str, torch.Tensor] = {}

    def bind(self, net) -> None:
        """Watch ``net``'s top encoder feature, and make the projection for it."""
        net.encoder.encoder[-1].register_forward_hook(
            lambda m, i, o: self._seen.__setitem__("student", o))
        self.proj = nn.Linear(net.encoder.hidden_dim, self.teacher.num_features)

    def parameters(self):
        return self.proj.parameters() if self.proj is not None else iter(())

    def to(self, device) -> FeatureTeacher:
        self.teacher.to(device)
        if self.proj is not None:
            self.proj.to(device)
        return self

    @torch.no_grad()
    def _teach(self, imgs: torch.Tensor, h: int, w: int) -> torch.Tensor:
        side = self.size // self.patch * self.patch
        x = F.interpolate(imgs.float(), size=(side, side), mode="bilinear", align_corners=False)
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        tokens = self.teacher.forward_features((x - mean) / std)
        tokens = tokens[:, self.teacher.num_prefix_tokens:]           # the patches only
        g = side // self.patch
        grid = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, g, g)
        grid = F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)
        return grid.flatten(2).transpose(1, 2)                        # B, h*w, D

    def __call__(self, imgs, out) -> tuple[torch.Tensor, dict[str, float]]:
        student = self._seen["student"].float()                      # B, h*w, C
        h, w = imgs.shape[-2] // 32, imgs.shape[-1] // 32
        if student.shape[1] != h * w:
            raise ValueError(f"the top encoder feature has {student.shape[1]} places, "
                             f"not {h}x{w}")
        target = self._teach(imgs, h, w)
        loss = (1 - F.cosine_similarity(self.proj(student), target, dim=-1)).mean()
        return self.weight * loss, {"kd_feat": float(loss.detach())}
