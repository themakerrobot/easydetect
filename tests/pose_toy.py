# Apache-2.0
"""A COCO-keypoints-shaped folder of drawn stick figures, for the pose tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# a standing figure, 17 COCO keypoints, in a 0-1 box (x right, y down)
TEMPLATE = np.array([
    [0.50, 0.08], [0.45, 0.05], [0.55, 0.05], [0.40, 0.08], [0.60, 0.08],
    [0.30, 0.25], [0.70, 0.25], [0.20, 0.42], [0.80, 0.42], [0.15, 0.58], [0.85, 0.58],
    [0.38, 0.55], [0.62, 0.55], [0.36, 0.76], [0.64, 0.76], [0.35, 0.97], [0.65, 0.97],
])
LIMBS = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))


def figure(rng, w: int, h: int):
    """Keypoints (17, 3) and box of one figure placed at random in a w×h picture."""
    bh = rng.uniform(0.5, 0.9) * h
    bw = bh * rng.uniform(0.45, 0.6)
    x0, y0 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
    pts = TEMPLATE + rng.normal(0, 0.025, TEMPLATE.shape)
    if rng.random() < 0.5:  # arms up
        pts[[9, 10], 1] -= 0.45
    xy = pts * [bw, bh] + [x0, y0]
    kpts = np.concatenate([xy, np.full((17, 1), 2.0)], 1)
    return kpts, [float(x0), float(y0), float(bw), float(bh)]


def draw(img, kpts) -> None:
    import cv2

    for i, (a, b) in enumerate(LIMBS):
        colour = tuple(int(c) for c in np.array([60 + 13 * i, 255 - 15 * i, 120 + 9 * i]) % 256)
        cv2.line(img, tuple(int(v) for v in kpts[a, :2]), tuple(int(v) for v in kpts[b, :2]),
                 colour, 3)
    for k, (x, y, _) in enumerate(kpts):
        cv2.circle(img, (int(x), int(y)), 4, (255 - 14 * k, 40 + 12 * k, 200), -1)


def make(root: Path, train: int = 40, val: int = 10, seed: int = 0, empty: int = 0) -> Path:
    """A COCO-shaped keypoint set of drawn stick figures; ``empty`` more val
    pictures have nobody in them (about half of COCO's val2017 does)."""
    import cv2

    rng = np.random.default_rng(seed)
    for split, count in (("train", train), ("val", val)):
        folder = root / "images" / f"{split}2017"
        folder.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for i in range(count):
            w, h = 320, 240
            img = np.full((h, w, 3), rng.integers(20, 90), np.uint8)
            kpts, box = figure(rng, w, h)
            draw(img, kpts)
            name = f"{i:06d}.jpg"
            cv2.imwrite(str(folder / name), img)
            images.append({"id": i + 1, "file_name": name, "width": w, "height": h})
            anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": 1,
                         "bbox": box, "area": box[2] * box[3] * 0.6, "iscrowd": 0,
                         "num_keypoints": 17, "keypoints": kpts.reshape(-1).tolist()})
        for i in range(count, count + (empty if split == "val" else 0)):
            name = f"{i:06d}.jpg"
            cv2.imwrite(str(folder / name), np.full((240, 320, 3), 60, np.uint8))
            images.append({"id": i + 1, "file_name": name, "width": 320, "height": 240})
        (root / "annotations").mkdir(exist_ok=True)
        (root / "annotations" / f"person_keypoints_{split}2017.json").write_text(json.dumps(
            {"images": images, "annotations": anns,
             "categories": [{"id": 1, "name": "person"}]}))
    return root
