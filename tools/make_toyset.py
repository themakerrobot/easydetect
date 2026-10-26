#!/usr/bin/env python3
# Apache-2.0
"""A labelled dataset to train on when you have none: squares and circles.

    python tools/make_toyset.py toyset          # 300 images, 80/20 train/val
    python tools/make_toyset.py toyset 1000

Every box is exact, shapes barely overlap, backgrounds vary, so a run that
learns nothing here is broken rather than short of data. For trying the loop
(train, val, export, predict, the lab), not for judging accuracy.
"""
import pathlib
import random
import sys

import cv2
import numpy as np

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "toyset").resolve()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rng = random.Random(0)
W, H = 640, 480

for split, count in (("train", int(n * 0.8)), ("val", n - int(n * 0.8))):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        # a gradient with noise, not a flat colour
        base = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
        grad = np.linspace(0, rng.uniform(-80, 80), W, dtype=np.float32)[None, :, None]
        noise = np.random.default_rng(i).normal(0, 12, (H, W, 3))
        img = np.clip(base + grad + noise, 0, 255).astype(np.uint8)
        lines, placed = [], []
        for _ in range(rng.randint(1, 5)):
            cls = rng.randint(0, 1)
            for _try in range(30):                     # a spot clear of the other shapes
                w, h = rng.randint(40, 200), rng.randint(40, 200)
                if cls == 1:
                    h = w                              # circles are round
                x, y = rng.randint(0, W - w), rng.randint(0, H - h)
                if all(x + w < px or px + pw < x or y + h < py or py + ph < y
                       for px, py, pw, ph in placed):
                    break
            else:
                continue                               # no room: skip this one
            placed.append((x, y, w, h))
            color = tuple(rng.randint(0, 255) for _ in range(3))
            if cls == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            else:
                cv2.circle(img, (x + w // 2, y + h // 2), w // 2, color, -1)
                cv2.circle(img, (x + w // 2, y + h // 2), w // 2, (0, 0, 0), 2)
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        cv2.imwrite(str(root / "images" / split / f"{i:04d}.jpg"), img)
        (root / "labels" / split / f"{i:04d}.txt").write_text("\n".join(lines) + "\n")

(root / "data.yaml").write_text(
    "train: images/train\nval: images/val\nnames:\n  0: square\n  1: circle\n"
)
print(f"{n} images -> {root}  (data.yaml ready)")
