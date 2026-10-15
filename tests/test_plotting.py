# Apache-2.0
"""Mask drawing: a uint8 blend inside each mask, as the float one was."""

from __future__ import annotations

import numpy as np

from easydetect.plotting import color_for, draw_masks


def _float_blend(img, masks, classes, alpha=0.45):
    out = img.astype(np.float32)
    for mask, cls in zip(masks, classes, strict=True):
        out[mask] = out[mask] * (1 - alpha) + np.array(color_for(int(cls)), np.float32) * alpha
    return out


def test_masks_are_tinted_as_a_float_blend_would():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (120, 160, 3), np.uint8)
    masks = np.zeros((3, 120, 160), bool)
    yy, xx = np.mgrid[:120, :160]
    masks[0] = (yy - 50) ** 2 + (xx - 60) ** 2 < 30 ** 2
    masks[1, 40:100, 50:150] = True                          # overlaps the first
    masks[2, 0, 159] = True                                  # one pixel in a corner
    out = draw_masks(img, masks, [0, 3, 7])
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.abs(out.astype(float) - _float_blend(img, masks, [0, 3, 7])).max() <= 1
    outside = ~masks.any(0)
    assert (out[outside] == img[outside]).all()


def test_empty_masks_and_the_original_are_left_alone():
    img = np.full((20, 30, 3), 77, np.uint8)
    out = draw_masks(img, np.zeros((2, 20, 30), bool), [0, 1])
    assert (out == img).all() and out is not img
    masks = np.zeros((1, 20, 30), bool)
    masks[5:10, 5:10] = True
    draw_masks(img, masks, [0])
    assert (img == 77).all()
