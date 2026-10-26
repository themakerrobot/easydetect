# Apache-2.0
"""task="segment": a mask per box, drawn, summarised, and in step with the boxes."""

from __future__ import annotations

import numpy as np
import pytest

from easydetect import Detector
from easydetect.results import Masks, Results
from easydetect.segment import BoxSegmenter

from .conftest import draw


def _disc(h=120, w=160, center=(60, 50), radius=30):
    yy, xx = np.mgrid[:h, :w]
    return (yy - center[1]) ** 2 + (xx - center[0]) ** 2 <= radius ** 2


def test_masks_give_outlines_and_areas():
    masks = Masks(np.stack([_disc(), np.zeros((120, 160), bool)]))
    assert len(masks) == 2 and masks.area[0] == _disc().sum() and masks.area[1] == 0
    outline, empty = masks.xy
    assert outline.shape[1] == 2 and len(outline) > 8 and empty.shape == (0, 2)
    assert abs(outline[:, 0].min() - 30) <= 1 and abs(outline[:, 0].max() - 90) <= 1


def test_a_result_with_masks_draws_and_summarises_them():
    img = np.full((120, 160, 3), 90, np.uint8)
    boxes = np.array([[30, 20, 90, 80, 0.9, 1]], np.float32)
    r = Results(img, names={1: "ball"}, boxes=boxes, masks=_disc()[None])
    drawn = r.plot()
    assert (drawn[50, 60] != img[50, 60]).any()             # tinted inside the mask
    assert (drawn[5, 150] == img[5, 150]).all()              # untouched outside
    row = r.summary()[0]
    assert row["name"] == "ball" and row["mask"]["area"] == _disc().sum()
    assert len(row["mask"]["polygon"]) > 8
    assert Results(img, boxes=boxes).masks is None           # a detect result has none


class _FakeSegmenter:
    """The box itself as the mask — enough to check the plumbing."""

    def __init__(self):
        self.calls = []

    def __call__(self, img, xyxy):
        self.calls.append(np.asarray(xyxy).copy())
        masks = np.zeros((len(xyxy), *img.shape[:2]), bool)
        for m, (x1, y1, x2, y2) in zip(masks, np.asarray(xyxy, int), strict=True):
            m[y1:y2, x1:x2] = True
        return masks, np.ones(len(xyxy), np.float32)


def test_segment_puts_one_mask_on_every_box(tiny_ir):
    model = Detector(str(tiny_ir), task="segment", verbose=False)
    model.segmenter = _FakeSegmenter()
    r = model(draw(), conf=0.0, max_det=4)[0]
    assert len(r.masks) == len(r.boxes) == 4 and "segment" in r.speed
    np.testing.assert_allclose(model.segmenter.calls[0], r.boxes.xyxy)
    # and a plain detector never builds one
    plain = Detector(str(tiny_ir), verbose=False)
    assert plain(draw(), conf=0.0)[0].masks is None and plain.segmenter is None


def test_the_task_is_checked():
    with pytest.raises(ValueError, match="task must be"):
        Detector("ed-s", task="dance")


def test_the_segmenter_sizes_the_picture_like_mobile_sam():
    tensor, scale = BoxSegmenter.preprocess(np.full((480, 640, 3), 200, np.uint8))
    assert tensor.shape == (1, 3, 1024, 1024) and scale == pytest.approx(1.6)
    assert tensor[0, :, 767, 1023].max() == 200 and tensor[0, :, 768, 0].max() == 0   # padded below


def test_the_command_line_segments_too(tiny_ir, tmp_path, monkeypatch):
    import cv2

    from easydetect import cli, segment

    made = []
    monkeypatch.setattr(segment, "default_segmenter",
                        lambda **kw: made.append(_FakeSegmenter()) or made[-1])
    picture = tmp_path / "p.jpg"
    cv2.imwrite(str(picture), draw())
    code = cli.main(["predict", f"model={tiny_ir}", "task=segment", f"source={picture}",
                     "conf=0.0", "save=false", f"project={tmp_path}"])
    assert code == 0 and len(made) == 1 and len(made[0].calls) == 1
