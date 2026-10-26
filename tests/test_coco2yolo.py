# Apache-2.0
"""tools/coco2yolo.py: COCO JSON in, labels the trainer reads out, classes in checkpoint order."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "coco2yolo.py"
# COCO's category ids skip numbers (12, 26, 29, 30, …): index must follow the order, not the id
COCO_IDS = [i for i in range(1, 91) if i not in (12, 26, 29, 30, 45, 66, 68, 69, 71, 83)]


def _tool():
    spec = importlib.util.spec_from_file_location("coco2yolo", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coco(root: Path, split: str, names: list[str]) -> None:
    (root / "images" / split).mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        cv2.imwrite(str(root / "images" / split / name), np.zeros((100, 200, 3), np.uint8))
    categories = [{"id": i, "name": n} for i, n in zip(COCO_IDS, names, strict=True)]
    def box(category, bbox, crowd=0):
        return {"image_id": 1, "category_id": category, "bbox": bbox, "iscrowd": crowd}

    annotations = [
        box(1, [20, 10, 40, 50]),          # person
        box(90, [180, 80, 40, 40]),        # toothbrush, spilling past the right edge
        box(3, [0, 0, 100, 100], crowd=1),  # crowd: skipped
        box(3, [5, 5, 0.5, 10]),           # under a pixel wide: skipped
    ]
    images = [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100},
              {"id": 2, "file_name": "b.jpg", "width": 200, "height": 100}]  # b: background
    (root / "annotations").mkdir(exist_ok=True)
    # categories listed backwards: the order in the file must not matter
    (root / "annotations" / f"instances_{split}.json").write_text(
        json.dumps({"images": images, "annotations": annotations,
                    "categories": list(reversed(categories))}))


def test_labels_follow_the_checkpoint_class_order(tmp_path):
    tool = _tool()
    names = tool.coco_names()
    assert len(names) == 80 and names[0] == "person" and names[79] == "toothbrush"
    for split in ("train2017", "val2017"):
        _coco(tmp_path, split, names)
    assert tool.main([str(tmp_path)]) == 0

    rows = (tmp_path / "labels" / "train2017" / "a.txt").read_text().split("\n")
    first, second = (list(map(float, r.split())) for r in rows[:2])
    assert first == pytest.approx([0, 40 / 200, 35 / 100, 40 / 200, 50 / 100])
    assert second[0] == 79                                      # id 90 is index 79
    assert second[1] + second[3] / 2 == pytest.approx(1.0)      # clipped at the right edge
    assert rows[2] == ""                                        # crowd and tiny left out
    assert (tmp_path / "labels" / "train2017" / "b.txt").read_text() == ""

    from easydetect.data.dataset import DetDataset

    ds = DetDataset(tmp_path / "data.yaml", "val", imgsz=64, augment=False)
    assert len(ds) == 2 and ds.nc == 80 and ds.names[79] == "toothbrush"
    _, target = ds[[f.name for f in ds.files].index("a.jpg")]
    assert target["labels"].tolist() == [0, 79]


def test_a_category_order_other_than_the_checkpoints_stops_the_conversion(tmp_path):
    tool = _tool()
    names = tool.coco_names()
    swapped = [names[1], names[0], *names[2:]]
    _coco(tmp_path, "val2017", swapped)
    with pytest.raises(SystemExit, match="differ"):
        tool.main([str(tmp_path), "--splits", "val2017"])
