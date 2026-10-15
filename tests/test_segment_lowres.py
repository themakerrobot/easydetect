# Apache-2.0
"""Scaling the decoder's low-res logits up one box at a time gives what SAM's
postprocessing (up to 1024, crop the padding, to the picture) gives."""

from __future__ import annotations

import numpy as np
import pytest

from easydetect.segment import upscale_logits

from .conftest import needs_torch


def _sam_postprocess(low: np.ndarray, h: int, w: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(low)[None, None]
    x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
    s = 1024 / max(h, w)
    x = x[..., :int(np.floor(h * s + 0.5)), :int(np.floor(w * s + 0.5))]
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()


def _logits(seed: int) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    low = cv2.resize(rng.normal(0, 4, (16, 16)).astype(np.float32), (256, 256),
                     interpolation=cv2.INTER_CUBIC)
    return low


@needs_torch
@pytest.mark.parametrize("h, w", [(480, 640), (2160, 3840), (777, 333), (1000, 1501), (200, 90)])
def test_scaled_up_logits_match_sams_postprocessing(h, w):
    low = _logits(h + w)
    ref = _sam_postprocess(low, h, w)
    whole = upscale_logits(low, (h, w), (0, 0, w, h))
    assert whole.shape == (h, w)
    assert np.abs(whole - ref).max() < 0.05
    assert ((whole > 0) == (ref > 0)).mean() > 0.9999
    # a region is the same pixels the whole picture has there, edges included
    for x1, y1, x2, y2 in [(37, 21, w - 50, h - 13), (0, 0, w // 3, h // 4),
                           (w // 2, h // 2, w, h), (5, 7, 6, 8)]:
        part = upscale_logits(low, (h, w), (x1, y1, x2, y2))
        np.testing.assert_allclose(part, whole[y1:y2, x1:x2], atol=1e-4)
