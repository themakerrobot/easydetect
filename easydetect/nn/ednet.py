# Apache-2.0
"""The ed detector in five sizes, assembled from backbone, encoder and decoder.

The per-size settings are the original released COCO configs
(one per size), so the official
checkpoints load with ``strict=True``. See NOTICE.
"""

from __future__ import annotations

import copy

import torch.nn as nn

from .decoder import EDTransformer
from .encoder import HybridEncoder
from .hgnetv2 import HGNetv2

#: Per-size settings; anything not listed takes the module's default.
SIZE_CFG = {
    "n": {
        "backbone": {"name": "B0", "return_idx": [2, 3], "freeze_at": -1, "freeze_norm": False,
                     "use_lab": True},
        "encoder": {"in_channels": [512, 1024], "feat_strides": [16, 32], "hidden_dim": 128,
                    "use_encoder_idx": [1], "dim_feedforward": 512, "expansion": 0.34,
                    "depth_mult": 0.5},
        "decoder": {"feat_channels": [128, 128], "feat_strides": [16, 32], "hidden_dim": 128,
                    "dim_feedforward": 512, "num_levels": 2, "num_layers": 3,
                    "num_points": [6, 6]},
    },
    "s": {
        "backbone": {"name": "B0", "freeze_at": -1, "freeze_norm": False, "use_lab": True},
        "encoder": {"in_channels": [256, 512, 1024], "depth_mult": 0.34, "expansion": 0.5},
        "decoder": {"num_layers": 3},
    },
    "m": {
        "backbone": {"name": "B2", "freeze_at": -1, "freeze_norm": False, "use_lab": True},
        "encoder": {"in_channels": [384, 768, 1536], "depth_mult": 0.67},
        "decoder": {"num_layers": 4},
    },
    "l": {
        "backbone": {"name": "B4"},
        "encoder": {},
        "decoder": {},
    },
    "x": {
        "backbone": {"name": "B5"},
        "encoder": {"hidden_dim": 384, "dim_feedforward": 2048},
        "decoder": {"feat_channels": [384, 384, 384], "reg_scale": 8},
    },
}
SIZES = tuple(SIZE_CFG)

# what every size shares (the original shared config)
_BACKBONE = {"return_idx": [1, 2, 3], "freeze_stem_only": True, "freeze_at": 0,
             "freeze_norm": True, "use_lab": False}
_ENCODER = {"in_channels": [512, 1024, 2048], "feat_strides": [8, 16, 32], "hidden_dim": 256,
            "use_encoder_idx": [2], "num_encoder_layers": 1, "nhead": 8,
            "dim_feedforward": 1024, "dropout": 0.0, "enc_act": "gelu", "expansion": 1.0,
            "depth_mult": 1.0, "act": "silu"}
_DECODER = {"feat_channels": [256, 256, 256], "feat_strides": [8, 16, 32], "hidden_dim": 256,
            "num_levels": 3, "num_layers": 6, "eval_idx": -1, "num_queries": 300,
            "num_denoising": 100, "label_noise_ratio": 0.5, "box_noise_scale": 1.0,
            "reg_max": 32, "reg_scale": 4, "layer_scale": 1, "num_points": [3, 6, 3],
            "cross_attn_method": "default", "query_select_method": "default"}


class EDNet(nn.Module):
    """``variant`` is the size letter: n, s, m, l or x."""

    def __init__(self, variant: str = "s", num_classes: int = 80, pretrained_backbone: bool = True):
        super().__init__()
        if variant not in SIZE_CFG:
            raise ValueError(f"size must be one of {SIZES}")
        cfg = SIZE_CFG[variant]
        self.variant = variant
        self.num_classes = num_classes
        self.backbone = HGNetv2(**{**_BACKBONE, **cfg["backbone"]}, pretrained=pretrained_backbone)
        # no fixed eval size: anchors and position embeddings follow the input,
        # so one network serves 320, 480 or 640 alike
        self.encoder = HybridEncoder(**copy.deepcopy({**_ENCODER, **cfg["encoder"]}))
        self.decoder = EDTransformer(num_classes=num_classes,
                                        **copy.deepcopy({**_DECODER, **cfg["decoder"]}))

    def forward(self, x, targets=None):
        return self.decoder(self.encoder(self.backbone(x)), targets)

    def trim(self, layers: int | None = None, queries: int | None = None) -> EDNet:
        """Run fewer decoder layers or queries (in place, before :meth:`deploy`).

        Every decoder layer is trained to answer on its own (each has its own
        score and box heads and its own loss), so stopping after ``layers`` of
        them gives a working detector that skips the rest. ``queries`` is how
        many candidate boxes the encoder hands the decoder (300 as trained);
        a scene with a handful of objects needs far fewer. Both cost accuracy
        and buy speed — measure on your data (docs/performance.md).
        """
        dec = self.decoder
        if hasattr(dec.decoder, "project"):          # convert_to_deploy() has run
            raise RuntimeError("trim() before deploy(): a deployed network has dropped its layers")
        if layers is not None:
            if not 1 <= int(layers) <= dec.num_layers:
                raise ValueError(
                    f"layers must be 1..{dec.num_layers} for ed-{self.variant}")
            dec.eval_idx = dec.decoder.eval_idx = int(layers) - 1
        if queries is not None:
            if not 1 <= int(queries) <= dec.num_queries:
                raise ValueError(f"queries must be 1..{dec.num_queries}")
            dec.num_queries = int(queries)
        return self

    def deploy(self) -> EDNet:
        """Fuse re-parameterisable blocks and drop training-only heads (in place)."""
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


class DeployWrapper(nn.Module):
    """ONNX/OpenVINO export head: returns (boxes cxcywh 0..1, scores sigmoid)."""

    def __init__(self, net: EDNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out["pred_boxes"], out["pred_logits"].sigmoid()
