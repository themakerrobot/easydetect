# Apache-2.0
"""Keypoint datasets of your own (Ultralytics' pose format): read, trained,
and used — a keypoint set that is not COCO's 17 end to end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from easydetect.data.labels import label_row_to_box, label_row_to_keypoints
from easydetect.pose import COCO, KeypointSpec

from .conftest import needs_torch

NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]


def _toy(root, n_train=8, n_val=3, seed=0):
    """Light rectangles on dark pictures, a keypoint on each corner; even K so
    a label line has an odd number of values, like a polygon."""
    import cv2

    rng = np.random.default_rng(seed)
    for split, count in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            w, h = 96, 80
            img = np.full((h, w, 3), 30, np.uint8)
            x0, y0 = rng.integers(8, 30), rng.integers(8, 25)
            x1, y1 = x0 + rng.integers(30, 55), y0 + rng.integers(25, 45)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (200, 220, 240), -1)
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            kp = " ".join(f"{x / w:.5f} {y / h:.5f} 2" for x, y in corners)
            box = f"{(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} {(x1 - x0) / w:.5f} " \
                  f"{(y1 - y0) / h:.5f}"
            (root / "labels" / split / f"{i}.txt").write_text(f"0 {box} {kp}\n")
    (root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(root), "train": "images/train", "val": "images/val",
        "names": {0: "card"}, "kpt_shape": [4, 3], "flip_idx": [1, 0, 3, 2],
        "kpt_names": NAMES, "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]]}))
    return root / "data.yaml"


def test_a_keypoint_line_is_not_read_as_a_polygon():
    line = "0 0.5 0.5 0.4 0.3 0.3 0.35 2 0.7 0.35 2 0.7 0.65 2 0.3 0.65 0".split()
    assert len(line) % 2 == 1  # what a polygon looks like to a count
    assert label_row_to_box(line, kpt_shape=[4, 3]) == [0.0, 0.5, 0.5, 0.4, 0.3]
    assert label_row_to_box(line)[3] != 0.4  # without the shape: a polygon's box
    box, kpts = label_row_to_keypoints(line, [4, 3])
    assert box == [0.0, 0.5, 0.5, 0.4, 0.3] and kpts.shape == (4, 3) and kpts[3, 2] == 0
    _, two = label_row_to_keypoints("0 .5 .5 .4 .3 .3 .35 0 0 .7 .65".split(), [3, 2])
    assert two[:, 2].tolist() == [2, 0, 2]  # (0, 0) is the unlabelled one


def test_a_spec_reads_from_a_data_yaml_and_round_trips(tmp_path):
    from easydetect.data.dataset import load_data_yaml

    spec = KeypointSpec.from_data(load_data_yaml(_toy(tmp_path)))
    assert spec.names == tuple(NAMES) and spec.flip == (1, 0, 3, 2)
    assert spec.classes == (0,) and np.allclose(spec.sigmas, 0.25)
    assert KeypointSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert not spec.is_coco and COCO.is_coco
    with pytest.raises(ValueError, match="flip_idx"):
        KeypointSpec(NAMES, flip=[0, 0, 1, 2])


def test_the_dataset_crops_carry_the_corners(tmp_path):
    from easydetect.data.keypoints import KeypointDataset
    from easydetect.pose import apply, invert

    data = _toy(tmp_path)
    ds = KeypointDataset.from_yolo(data, "train", augment=False)
    assert len(ds) == 8 and len(ds.spec) == 4
    crop, xy, weight, i = ds[0]
    kpts = np.asarray(ds.items[0][1]["keypoints"]).reshape(4, 3)
    np.testing.assert_allclose(apply(invert(ds.matrix(0)), xy), kpts[:, :2], atol=1e-3)
    aug = KeypointDataset.from_yolo(data, "train", augment=True)
    for k in range(8):
        c, p, w, _ = aug[k]
        assert p.shape == (4, 2) and w.shape == (4,)


@needs_torch
def test_training_on_a_keypoint_dataset_gives_a_model_that_predicts_it(tmp_path):
    import cv2

    from easydetect import Detector
    from easydetect.pose import spec_of

    data = _toy(tmp_path / "data")
    model = Detector("dfine-n", pretrained=False, verbose=False)
    heard = []
    best = model.train(data=str(data), epochs=1, imgsz=64, batch=4, workers=0, device="cpu",
                       project=str(tmp_path / "runs"), amp=False, pose_epochs=2, pose_batch=4,
                       on_progress=heard.append)
    beside = best.parent / "pose.onnx"
    assert beside.exists()
    stage = [h for h in heard if h["phase"] == "keypoints"]
    assert stage and stage[-1]["epoch"] == 2 and "ap" in stage[-1]

    # an export takes the keypoint model along
    onnx = model.export(format="onnx", out_dir=tmp_path / "exported", verbose=False)
    assert (Path(onnx).parent / "pose.onnx").exists()
    spec = spec_of(beside)
    assert spec.names == tuple(NAMES) and spec.skeleton[0] == (0, 1) and spec.classes == (0,)

    img = cv2.imread(str(next((tmp_path / "data" / "images" / "val").glob("*.jpg"))))
    r = Detector(str(best), task="pose", verbose=False)(img, conf=0.0, max_det=3)[0]
    assert r.keypoints.data.shape == (len(r.boxes), 4, 3)
    assert r.keypoints.names == tuple(NAMES)
    rows = [row for row in r.summary() if "keypoints" in row]
    assert rows and set(rows[0]["keypoints"]) == set(NAMES)
    assert r.plot().shape == img.shape

    # boxes only, when asked
    other = Detector("dfine-n", pretrained=False, verbose=False)
    best2 = other.train(data=str(data), epochs=1, imgsz=64, batch=4, workers=0, device="cpu",
                        project=str(tmp_path / "runs"), name="boxes", amp=False, pose=False)
    assert not (best2.parent / "pose.onnx").exists()
