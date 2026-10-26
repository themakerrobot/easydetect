# Apache-2.0
"""Decoding rules — including the sigmoid bug that cost us an afternoon once."""

from __future__ import annotations

import json

import numpy as np
import pytest

from easydetect.predictor import _looks_like_logits, read_names
from easydetect.utils.ops import cxcywh2xyxy_np

from .conftest import needs_ov, needs_torch


class _Decoder:
    """postprocess() without compiling anything — it is a pure function."""

    from easydetect.predictor import OVPredictor

    postprocess = OVPredictor.postprocess


def decode(boxes, scores, shape=(100, 100), **kwargs):
    return _Decoder.postprocess(_Decoder(), boxes, scores, shape, **kwargs)


def test_probabilities_from_the_exported_ir_are_not_sigmoided_twice():
    """sigmoid(0.95) = 0.72 — double-squashing silently breaks conf thresholds."""
    boxes = np.full((4, 4), 0.5, np.float32)
    scores = np.zeros((4, 3), np.float32)
    scores[0, 1] = 0.95
    det = decode(boxes, scores, conf=0.9)
    assert len(det) == 1 and det[0, 4] == pytest.approx(0.95)


def test_raw_logits_still_get_their_sigmoid():
    boxes = np.full((4, 4), 0.5, np.float32)
    scores = np.full((4, 3), -8.0, np.float32)
    scores[0, 1] = 3.0
    det = decode(boxes, scores, conf=0.9)
    assert len(det) == 1 and det[0, 4] == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-4)
    assert _looks_like_logits(scores) and not _looks_like_logits(np.zeros((2, 2), np.float32))


def test_boxes_come_back_in_pixels_sorted_by_confidence_and_clipped():
    boxes = np.array([[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.6, 0.6]], np.float32)
    scores = np.array([[0.4, 0.0], [0.9, 0.0]], np.float32)
    det = decode(boxes, scores, shape=(200, 100), conf=0.1)
    assert det[:, 4].tolist() == pytest.approx([0.9, 0.4])
    assert det[:, 0].min() >= 0 and det[:, 2].max() <= 100 and det[:, 3].max() <= 200
    assert det[1, :4].tolist() == pytest.approx([30.0, 60.0, 70.0, 140.0])


def test_conf_max_det_and_classes_all_filter():
    boxes = np.full((5, 4), 0.5, np.float32)
    scores = np.array([[0.9, 0.1], [0.8, 0.1], [0.1, 0.7], [0.05, 0.0], [0.6, 0.0]], np.float32)
    # five boxes on one spot: the duplicate filter would keep one, so it is off here
    assert len(decode(boxes, scores, conf=0.5, iou=None)) == 4
    assert len(decode(boxes, scores, conf=0.5, max_det=2, iou=None)) == 2
    only_car = decode(boxes, scores, conf=0.5, classes=[1], iou=None)
    assert only_car[:, 5].tolist() == [1.0]


def test_one_object_found_twice_under_two_classes_keeps_the_better_box():
    """ed-s on a street frame: truck 0.83 and car 0.57 on one vehicle, IoU 0.93."""
    boxes = np.array([[0.50, 0.50, 0.40, 0.30],     # the truck
                      [0.50, 0.51, 0.40, 0.29],     # the same vehicle, as a car
                      [0.20, 0.20, 0.10, 0.10]],    # something else
                     np.float32)
    scores = np.array([[0.83, 0.0], [0.0, 0.57], [0.60, 0.0]], np.float32)
    det = decode(boxes, scores, conf=0.25)
    assert det[:, 4].tolist() == pytest.approx([0.83, 0.60])
    assert len(decode(boxes, scores, conf=0.25, iou=None)) == 3


def test_neighbours_that_only_touch_are_both_kept():
    boxes = np.array([[0.30, 0.5, 0.4, 0.4], [0.62, 0.5, 0.4, 0.4]], np.float32)  # IoU ~0.1
    scores = np.array([[0.9], [0.8]], np.float32)
    assert len(decode(boxes, scores, conf=0.25)) == 2


def test_drop_duplicates_is_greedy_from_the_best():
    from easydetect.predictor import drop_duplicates

    xyxy = np.array([[0, 0, 10, 10], [0, 0, 10, 9], [0, 0, 10, 8], [50, 50, 60, 60]], np.float32)
    assert drop_duplicates(xyxy, 0.7).tolist() == [0, 3]
    assert drop_duplicates(xyxy, 0.85).tolist() == [0, 2, 3]      # IoU 0.9 goes, 0.8 stays


def test_cxcywh_to_xyxy_matches_the_hand_worked_example():
    box = np.array([[0.5, 0.5, 0.2, 0.4]], np.float32)
    assert cxcywh2xyxy_np(box) == pytest.approx(np.array([[0.4, 0.3, 0.6, 0.7]]))


def test_class_names_are_read_from_labels_txt_or_the_json_sidecar(tmp_path):
    (tmp_path / "labels.txt").write_text("can\nbottle\n")
    assert read_names(tmp_path / "m.xml") == {0: "can", 1: "bottle"}
    (tmp_path / "m.names.json").write_text(json.dumps({"0": "cat", "1": "dog"}))
    assert read_names(tmp_path / "m.xml") == {0: "cat", 1: "dog"}
    assert read_names(tmp_path / "elsewhere" / "m.xml") == {}


@needs_torch
@needs_ov
def test_a_real_ir_round_trips_from_pixels_to_detections(tiny_ir):
    from easydetect.predictor import OVPredictor

    predictor = OVPredictor(tiny_ir, device="CPU")
    assert predictor.imgsz == 64 and predictor.names == {0: "can", 1: "bottle"}
    det, speed = predictor(np.zeros((120, 90, 3), np.uint8), conf=0.0, max_det=5)
    assert det.shape == (5, 6)
    assert det[:, 0].max() <= 90 and det[:, 3].max() <= 120
    assert set(speed) == {"preprocess", "inference", "postprocess"}



@needs_ov
@needs_torch
def test_one_predictor_can_be_shared_between_threads(tiny_ir):
    """CompiledModel's own infer request is shared; a second caller used to get
    "Infer Request is busy". Each thread now has a request of its own."""
    import threading

    from easydetect.predictor import OVPredictor

    predictor = OVPredictor(tiny_ir, device="CPU", precision="f32")
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (48, 64, 3), np.uint8) for _ in range(4)]
    expected = [predictor(img, conf=0.0)[0] for img in images]

    failures: list[str] = []

    def hammer(k: int) -> None:
        for _ in range(20):
            try:
                det = predictor(images[k], conf=0.0)[0]
                if det.shape != expected[k].shape or not np.allclose(det, expected[k], atol=1e-4):
                    failures.append(f"thread {k}: answer changed under load")
            except Exception as exc:  # noqa: BLE001 - the point is that nothing raises
                failures.append(f"thread {k}: {exc}")

    threads = [threading.Thread(target=hammer, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_a_half_hidden_object_and_its_pieces_become_one_box():
    from easydetect.predictor import drop_contained, drop_duplicates

    # best first: a piece (0.67), the whole chair (0.64), another piece (0.61),
    # and a second chair beside it that only touches the first
    xyxy = np.array([[320, 520, 372, 640],     # lower right piece
                     [285, 455, 373, 640],     # the whole chair
                     [285, 470, 330, 640],     # left piece
                     [380, 455, 470, 640]],    # another chair
                    np.float32)
    scores = np.array([0.67, 0.64, 0.61, 0.60])
    cls = np.array([56, 56, 56, 56])
    assert list(drop_duplicates(xyxy, 0.7)) == [0, 1, 2, 3]     # IoU sees four chairs
    # the whole chair stays, its pieces go; the chair beside it is left alone
    assert list(drop_contained(xyxy, scores, cls, 0.8)) == [1, 3]


def test_a_loose_box_around_a_group_does_not_erase_its_members():
    from easydetect.predictor import drop_contained

    xyxy = np.array([[0, 0, 50, 100], [50, 0, 100, 100], [0, 0, 100, 100]], np.float32)
    scores = np.array([0.9, 0.85, 0.3])               # two people, and a box around both
    assert list(drop_contained(xyxy, scores, np.zeros(3), 0.8)) == [0, 1]


def test_containment_never_crosses_classes():
    from easydetect.predictor import drop_contained

    xyxy = np.array([[0, 0, 100, 100], [10, 10, 40, 40]], np.float32)   # a cup on a table
    scores = np.array([0.8, 0.7])
    assert list(drop_contained(xyxy, scores, np.array([60, 41]), 0.8)) == [0, 1]
    assert list(drop_contained(xyxy, scores, np.array([60, 60]), 0.8)) == [0]


def test_the_whole_path_merges_pieces_only_when_asked():
    """postprocess as predict() runs it: model outputs in, boxes out, at 640 x 720."""
    from easydetect.predictor import Predictor

    pieces = [(320, 520, 372, 640, 0.67), (285, 455, 373, 640, 0.64), (285, 470, 330, 640, 0.61),
              (40, 100, 340, 640, 0.97)]                   # three chair boxes and a person
    boxes = np.array([[(x1 + x2) / 2 / 640, (y1 + y2) / 2 / 720, (x2 - x1) / 640,
                       (y2 - y1) / 720] for x1, y1, x2, y2, _ in pieces], np.float32)
    scores = np.zeros((4, 80), np.float32)
    for i, (*_, s) in enumerate(pieces):
        scores[i, 0 if i == 3 else 56] = s
    post = Predictor.postprocess.__get__(object.__new__(Predictor))
    default = post(boxes, scores, (720, 640), conf=0.5)
    merged = post(boxes, scores, (720, 640), conf=0.5, contain=0.8)
    assert len(default) == 4                                # NMS at 0.7 keeps every piece
    assert sorted(merged[:, 5].tolist()) == [0, 56]         # one person, one chair
    chair = merged[merged[:, 5] == 56][0]
    assert chair[:4].round().tolist() == [285, 455, 373, 640]   # the whole of it


def test_the_defaults_are_the_measured_ones():
    """conf 0.5, NMS 0.7, contain off: docs/performance.md, 'Confidence and overlap defaults'."""
    import inspect

    from easydetect import Detector

    for method in (Detector.predict, Detector.track):
        params = inspect.signature(method).parameters
        assert params["conf"].default == 0.5 and params["iou"].default == 0.7
        assert params["contain"].default is None
