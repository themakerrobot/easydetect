#!/usr/bin/env python3
# Apache-2.0
"""Turn Pascal VOC 2007 into an easydetect dataset — a smaller, fixed subset if asked.

    python tools/voc2yolo.py VOCdevkit/VOC2007 voc --train 800 --val 1000

reads ``Annotations/*.xml`` and ``ImageSets/Main/trainval.txt`` and writes
``voc/images/{train,val}``, ``voc/labels/{train,val}`` (``cls cx cy w h``,
normalised) and ``voc/data.yaml``, the 20 VOC classes in their usual order.
The two splits are drawn without overlap from trainval with ``--seed``, so
every run of a comparison trains and scores on the same pictures. Objects
VOC marks "difficult" are left out, as VOC's own scoring ignores them.
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

NAMES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
         "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
         "sofa", "train", "tvmonitor")


def labels_of(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    rows = []
    for obj in root.iter("object"):
        if int(obj.findtext("difficult", "0")):
            continue
        name = obj.findtext("name").strip()
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1 = max(x1 - 1, 0.0), max(y1 - 1, 0.0)  # VOC counts pixels from 1
        rows.append(f"{NAMES.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return rows


def convert(voc: Path, out: Path, train: int | None, val: int | None, seed: int) -> Path:
    ids = (voc / "ImageSets" / "Main" / "trainval.txt").read_text().split()
    random.Random(seed).shuffle(ids)
    val_ids = ids[:val] if val else ids[: len(ids) // 5]
    rest = ids[len(val_ids):]
    train_ids = rest[:train] if train else rest
    for split, chosen in (("train", train_ids), ("val", val_ids)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in chosen:
            shutil.copyfile(voc / "JPEGImages" / f"{i}.jpg", out / "images" / split / f"{i}.jpg")
            rows = labels_of(voc / "Annotations" / f"{i}.xml")
            (out / "labels" / split / f"{i}.txt").write_text("\n".join(rows) + "\n" if rows else "")
    data = out / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images/train",
                                    "val": "images/val", "names": dict(enumerate(NAMES))}))
    print(f"{out}: {len(train_ids)} train, {len(val_ids)} val pictures")
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("voc", type=Path, help="VOCdevkit/VOC2007")
    p.add_argument("out", type=Path)
    p.add_argument("--train", type=int, help="pictures to train on (default: the rest)")
    p.add_argument("--val", type=int, help="pictures to score on (default: a fifth)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    convert(args.voc, args.out, args.train, args.val, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
