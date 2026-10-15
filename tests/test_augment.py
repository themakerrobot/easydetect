# Apache-2.0
"""Zoom-out, crop and flip must move the boxes exactly as they move the pixels."""

from __future__ import annotations

import random

import numpy as np
import pytest

from easydetect.data import augment as aug


def _scene(h=120, w=200):
    """Black picture with two white boxes; labels in cls cx cy w h."""
    img = np.zeros((h, w, 3), np.uint8)
    boxes = [(0, 20, 30, 70, 90), (1, 120, 10, 190, 60)]       # cls, x1, y1, x2, y2
    labels = []
    for c, x1, y1, x2, y2 in boxes:
        img[y1:y2, x1:x2] = 255
        labels.append([c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
    return img, np.array(labels, np.float32)


@pytest.fixture
def no_colour(monkeypatch):
    monkeypatch.setattr(aug, "PHOTO_P", 0.0)    # keep white white, to find the boxes


def test_every_box_still_covers_its_object(no_colour):
    random.seed(0)
    zoomed = cropped = 0
    for _ in range(300):
        img, labels = _scene()
        out, got = aug.apply(img, labels)
        h, w = out.shape[:2]
        zoomed += h > 120
        cropped += h < 120 or w < 200
        assert got.shape[1] == 5 and set(got[:, 0]) <= {0, 1}
        assert (got[:, 1:] >= 0).all() and (got[:, 1:] <= 1 + 1e-6).all()
        for _, cx, cy, bw, bh in got:
            x1, x2 = int(round((cx - bw / 2) * w)), int(round((cx + bw / 2) * w))
            y1, y2 = int(round((cy - bh / 2) * h)), int(round((cy + bh / 2) * h))
            inside = out[y1 + 1:y2 - 1, x1 + 1:x2 - 1]
            if inside.size:
                assert inside.mean() > 240, "a box that no longer sits on its object"
    assert zoomed > 50 and cropped > 50             # both happen, often


def test_the_plain_phase_only_flips(no_colour):
    random.seed(1)
    img, labels = _scene()
    for _ in range(20):
        out, got = aug.apply(img.copy(), labels.copy(), strong=False)
        assert out.shape == img.shape and len(got) == 2
        assert np.allclose(sorted(got[:, 3]), sorted(labels[:, 3]))


def test_a_background_picture_goes_through(no_colour):
    random.seed(2)
    for _ in range(50):
        out, got = aug.apply(np.zeros((80, 80, 3), np.uint8), np.zeros((0, 5), np.float32))
        assert got.shape == (0, 5) and out.ndim == 3 and min(out.shape[:2]) > 0


def test_colour_jitter_keeps_the_picture_valid():
    random.seed(3)
    img = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    for _ in range(30):
        out = aug.photometric(img)
        assert out.dtype == np.uint8 and out.shape == img.shape


def test_the_last_tenth_trains_on_plain_pictures():
    from easydetect.trainer import Trainer

    class Stub:                                   # only clean_from and _augment_lines are used
        clean_from = Trainer.clean_from
        _augment_lines = Trainer._augment_lines
        multiscale, mosaic, mixup = Trainer.multiscale, Trainer.mosaic, Trainer.mixup

    stub = Stub()
    for epochs, clean in ((20, 18), (50, 45), (100, 90), (5, 4), (3, 3), (1, 1)):
        stub.epochs = epochs
        assert stub.clean_from == clean
    stub.epochs, stub.imgsz, stub.augment = 20, 640, True
    assert any("last 2 epoch" in line for line in stub._augment_lines())
    stub.augment = False
    assert stub._augment_lines() == [aug.DESCRIPTION[-1], "resize to 640×640"]


def test_rendered_at_training_size_the_boxes_still_cover_their_objects(no_colour):
    """The path training takes: zoom, crop and flip drawn straight at 96 x 96."""
    random.seed(5)
    for _ in range(300):
        img, labels = _scene()
        out, got = aug.apply(img, labels, size=96)
        assert out.shape == (96, 96, 3)
        for _, cx, cy, bw, bh in got:
            x1, x2 = int(np.ceil((cx - bw / 2) * 96)) + 1, int((cx + bw / 2) * 96) - 1
            y1, y2 = int(np.ceil((cy - bh / 2) * 96)) + 1, int((cy + bh / 2) * 96) - 1
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                assert out[y1:y2, x1:x2].mean() > 230, "a box that no longer sits on its object"


def test_a_mosaic_keeps_every_box_on_its_object(no_colour):
    random.seed(1)
    for _ in range(50):
        canvas, labels = aug.mosaic([_scene() for _ in range(4)], 160)
        assert canvas.shape == (160, 160, 3) and len(labels) <= 8
        assert (labels[:, 1:] >= 0).all() and (labels[:, 1:] <= 1 + 1e-6).all()
        for _, cx, cy, w, h in labels:
            if w * 160 >= 4 and h * 160 >= 4:  # big enough to sample its middle
                assert canvas[int(cy * 160), int(cx * 160)].mean() > 128


def test_mixup_lays_two_pictures_over_each_other():
    a = (np.full((64, 64, 3), 200, np.uint8), np.array([[0, 0.5, 0.5, 0.2, 0.2]], np.float32))
    b = (np.zeros((64, 64, 3), np.uint8), np.array([[1, 0.2, 0.2, 0.1, 0.1]], np.float32))
    img, labels = aug.mixup(a, b)
    assert 200 * 0.4 - 1 <= img.mean() <= 200 * 0.6 + 1
    assert labels[:, 0].tolist() == [0, 1]


def test_a_mosaic_tile_is_never_zoomed_out(monkeypatch):
    def refuse(view):
        raise AssertionError("zoom-out inside a mosaic tile")

    monkeypatch.setattr(aug, "zoom_out", refuse)
    random.seed(0)
    aug.mosaic([_scene() for _ in range(4)], 160)


def test_a_shrink_averages_instead_of_aliasing():
    # one-pixel stripes: averaged they are grey, sampled they stay black or white
    img = np.zeros((400, 400, 3), np.uint8)
    img[:, ::2] = 255
    for view in (np.array([0, 0, 400, 400], np.float32), np.array([10, 0, 390, 380], np.float32)):
        out = aug.render(img, view, (100, 100), flip=False)
        assert out.std() < 20, "a 4x shrink kept the stripes as aliasing"


def test_multiscale_sizes_favour_the_trained_size():
    from easydetect.trainer import BASE_SIZE_REPEAT, Trainer

    stub = type("T", (), {"imgsz": 640})()
    sizes = Trainer.train_sizes(stub)
    assert min(sizes) == 480 and max(sizes) == 800
    assert sizes.count(640) == BASE_SIZE_REPEAT and len(sizes) == 13   # D-FINE's list
