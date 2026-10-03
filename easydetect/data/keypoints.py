# Apache-2.0
"""COCO 2017 person keypoints, as crops for the top-down keypoint network.

    <root>/annotations/person_keypoints_{train,val}2017.json
    <root>/images/{train,val}2017/*.jpg

(the layout tools/coco2yolo.py reads too). Each labelled person is one item:
the box is widened and padded as at inference (easydetect.pose.box_to_crop),
and in training moved about — scale, rotation, sometimes only the upper or
lower half of the body, a flip that also swaps left and right — then the crop
gets the detector's colour jitter and, half the time, a patch blanked out, so
a hidden limb is still placed from the rest of the body.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..pose import FLIP, INPUT, box_to_crop, crop, crop_matrix
from .augment import photometric

SCALE = (0.7, 1.3)  # crop size factor
ROTATE, ROTATE_P = 60.0, 0.6  # degrees either way, how often
HALF_BODY_P, HALF_BODY_MIN = 0.3, 8  # only with this many keypoints labelled
HALF_BODY_MIN_SIZE = 32.0  # pixels: a smaller half is blown up past recognition
ERASE_P, ERASE_MAX = 0.5, 0.4  # a blanked patch up to this share of each side
UPPER = set(range(11))  # face, shoulders, arms
KEPT = ("image_id", "bbox", "keypoints", "num_keypoints", "area", "iscrowd")

DESCRIPTION = [
    f"crop scale ×{SCALE[0]:g}–{SCALE[1]:g}, rotation ±{ROTATE:g}° (p={ROTATE_P})",
    f"half body: upper or lower keypoints only (p={HALF_BODY_P}, "
    f"at least {HALF_BODY_MIN_SIZE:g} px)",
    "horizontal flip with left/right swapped (p=0.5)",
    "photometric jitter (the detector's)",
    f"one blanked patch up to {ERASE_MAX:g} of each side (p={ERASE_P})",
]


def load_coco(root: Path, split: str) -> tuple[dict, dict]:
    """``images {id: file}``, ``people {image id: [annotation, ...]}`` for a split."""
    path = Path(root) / "annotations" / f"person_keypoints_{split}2017.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: unzip COCO's annotations_trainval2017.zip into {root}")
    data = json.loads(path.read_text())
    images = {im["id"]: Path(root) / "images" / f"{split}2017" / im["file_name"]
              for im in data["images"]}
    people = defaultdict(list)
    for ann in data["annotations"]:
        # only what training and scoring read: the outlines alone would be most
        # of the memory, and every DataLoader worker holds its own copy
        people[ann["image_id"]].append({k: ann[k] for k in KEPT if k in ann})
    return images, dict(people)


class KeypointDataset:
    """``ds[i] -> (crop (3, 256, 192) float RGB 0-255, keypoints (17, 2) crop
    pixels, weight (17,), index)``; ``ds.matrix(i)`` maps the crop back."""

    def __init__(self, root: str | Path, split: str = "train", augment: bool = True,
                 limit: int | None = None) -> None:
        self.images, self.people = load_coco(Path(root), split)
        self.items = [
            (image_id, ann) for image_id, anns in sorted(self.people.items()) for ann in anns
            if not ann.get("iscrowd") and ann.get("num_keypoints", 0) > 0
            and ann["bbox"][2] > 1 and ann["bbox"][3] > 1
        ]
        if limit:
            self.items = self.items[:limit]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _xyxy(ann) -> np.ndarray:
        x, y, w, h = ann["bbox"]
        return np.array([x, y, x + w, y + h], np.float64)

    def matrix(self, i: int) -> np.ndarray:
        """The plain (unaugmented) map from picture to crop for item ``i``."""
        center, size = box_to_crop(self._xyxy(self.items[i][1]))
        return crop_matrix(center[0], size[0])

    def _half_body(self, kpts: np.ndarray):
        labelled = np.where(kpts[:, 2] > 0)[0]
        if len(labelled) < HALF_BODY_MIN:
            return None
        upper = [k for k in labelled if k in UPPER]
        lower = [k for k in labelled if k not in UPPER]
        chosen = upper if (random.random() < 0.5 and len(upper) > 2) or len(lower) < 2 else lower
        if len(chosen) < 2:
            return None
        pts = kpts[chosen, :2]
        lo, hi = pts.min(0), pts.max(0)
        # a few keypoints a handful of pixels apart would fill the crop at up
        # to 150x (measured on COCO train): a blur with keypoints in it
        if (hi - lo).max() < HALF_BODY_MIN_SIZE:
            return None
        return np.concatenate([lo, hi])

    def __getitem__(self, i: int):
        image_id, ann = self.items[i]
        img = cv2.imread(str(self.images[image_id]))
        if img is None:
            raise FileNotFoundError(self.images[image_id])
        kpts = np.asarray(ann["keypoints"], np.float64).reshape(-1, 3)
        box = self._xyxy(ann)
        rotation, flip = 0.0, False
        if self.augment:
            if random.random() < HALF_BODY_P:
                half = self._half_body(kpts)
                if half is not None:
                    box = half
            center, size = box_to_crop(box)
            center, size = center[0], size[0] * random.uniform(*SCALE)
            if random.random() < ROTATE_P:
                rotation = random.uniform(-ROTATE, ROTATE)
            flip = random.random() < 0.5
        else:
            center, size = box_to_crop(box)
            center, size = center[0], size[0]
        m = crop_matrix(center, size, rotation)
        out = crop(img, m)
        xy = kpts[:, :2] @ m[:, :2].T + m[:, 2]
        weight = kpts[:, 2] > 0
        if flip:
            out = out[:, ::-1]
            xy[:, 0] = INPUT[1] - 1 - xy[:, 0]
            xy, weight = xy[list(FLIP)], weight[list(FLIP)]
        if self.augment:
            out = photometric(np.ascontiguousarray(out))
            if random.random() < ERASE_P:
                h, w = INPUT
                eh = int(h * random.uniform(0.1, ERASE_MAX))
                ew = int(w * random.uniform(0.1, ERASE_MAX))
                y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
                out[y0:y0 + eh, x0:x0 + ew] = np.random.randint(0, 256, 3, dtype=np.uint8)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < INPUT[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < INPUT[0])
        weight = (weight & inside).astype(np.float32)
        tensor = cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_BGR2RGB) \
            .transpose(2, 0, 1).astype(np.float32)
        return tensor, xy.astype(np.float32), weight, i
