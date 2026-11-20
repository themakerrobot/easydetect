# Apache-2.0
"""tools/hub_tasks.py: masks laid out from a dataset's own annotations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
hub_tasks = pytest.importorskip("hub_tasks")


def test_an_object_in_parts_fills_as_one():
    parts = [[0, 0, 10, 0, 10, 10, 0, 10], [20, 0, 30, 0, 30, 10, 20, 10]]
    poly = hub_tasks.joined(parts)
    canvas = np.zeros((12, 32), np.uint8)
    cv2.fillPoly(canvas, [np.round(poly).astype(np.int32)], 1)
    assert canvas[5, 5] and canvas[5, 25] and not canvas[5, 15]
    assert hub_tasks.joined([[1, 2, 3]]) is None


def _coco(tmp: Path) -> Path:
    coco = tmp / "coco"
    (coco / "annotations").mkdir(parents=True)
    cats = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
    for split, part, n in (("train", "train2017", 3), ("val", "val2017", 2)):
        (coco / "images" / part).mkdir(parents=True)
        images, anns = [], []
        for i in range(n):
            iid = 100 * (split == "val") + i + 1
            cv2.imwrite(str(coco / "images" / part / f"{iid:012d}.jpg"),
                        np.zeros((40, 60, 3), np.uint8))
            images.append({"id": iid, "file_name": f"{iid:012d}.jpg", "width": 60, "height": 40})
            if i == n - 1 and split == "val":
                continue                                    # a picture with nothing in it
            anns.append({"id": 10 * iid, "image_id": iid, "category_id": 3, "iscrowd": 0,
                         "bbox": [6, 4, 24, 16], "area": 384,
                         "segmentation": [[6, 4, 30, 4, 30, 20, 6, 20]]})
            anns.append({"id": 10 * iid + 1, "image_id": iid, "category_id": 1, "iscrowd": 1,
                         "bbox": [0, 0, 5, 5], "area": 25,
                         "segmentation": {"size": [40, 60], "counts": [0, 2400]}})
        (coco / "annotations" / f"instances_{part}.json").write_text(
            json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return coco


def test_masks_are_laid_out_on_the_detectors_splits(tmp_path):
    coco = _coco(tmp_path)
    data = hub_tasks.seg_data("coco", tmp_path / "hub", coco)
    folder = data.parent
    label = (folder / "labels" / "train" / "1.txt").read_text().split()
    assert label[0] == "1" and len(label) == 9                # car is class 1; no crowd line
    assert np.allclose([float(v) for v in label[1:3]], [0.1, 0.1])
    truth = json.loads((folder / "val.json").read_text())
    assert [im["id"] for im in truth["images"]] == [101, 102]    # the empty one scores too
    assert {a["iscrowd"] for a in truth["annotations"]} == {0, 1}
    assert truth["categories"][1] == {"id": 1, "name": "car"}
