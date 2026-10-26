# Apache-2.0
"""Box utilities. Torch versions are only imported inside functions that need them."""

import numpy as np


def cxcywh2xyxy_np(x):
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# ---------------- torch ops (training) ----------------


def cxcywh2xyxy(x):
    import torch

    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy2cxcywh(x):
    import torch

    x1, y1, x2, y2 = x.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)


def box_iou(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> iou (N,M), union (N,M)."""
    import torch

    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9), union


def generalized_box_iou(a, b):
    """GIoU (N,M) for xyxy boxes."""
    import torch

    iou, union = box_iou(a, b)
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / area.clamp(min=1e-9)


def inverse_sigmoid(x, eps=1e-5):
    import torch

    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))
