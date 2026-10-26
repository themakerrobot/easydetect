#!/usr/bin/env python3
# Apache-2.0
"""Turn COCO 2017 (images + instances JSON) into the layout easydetect trains on.

    python tools/coco2yolo.py ~/datasets/coco

expects, under that folder,

    images/train2017/*.jpg   images/val2017/*.jpg        (the unzipped image zips)
    annotations/instances_train2017.json  annotations/instances_val2017.json

and writes labels/train2017/*.txt, labels/val2017/*.txt (``cls cx cy w h``,
normalised) and data.yaml. Class indices follow COCO's category ids in order
(1 person → 0 … 90 toothbrush → 79), the order the original COCO head was trained
with; the script stops if the JSON's categories say otherwise, because a
shuffled order would train every class under another's name without an error.
Crowd regions (``iscrowd``) are skipped, as COCO evaluation ignores them. A
picture with no boxes gets an empty label file: a background picture.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def coco_names() -> list[str]:
    """The 80 names in the pretrained checkpoints' order (tools/convert_upstream.py)."""
    spec = importlib.util.spec_from_file_location("convert_upstream", HERE / "convert_upstream.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.COCO_NAMES)


def convert_split(root: Path, split: str, expected: list[str]) -> dict[str, int]:
    source = root / "annotations" / f"instances_{split}.json"
    if not source.exists():
        raise SystemExit(f"missing {source} — unzip annotations_trainval2017.zip into {root}")
    images_dir = root / "images" / split
    if not images_dir.is_dir():
        raise SystemExit(f"missing {images_dir} — unzip {split}.zip into {root / 'images'}")
    coco = json.loads(source.read_text(encoding="utf-8"))

    categories = sorted(coco["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in categories]
    if names != expected:
        raise SystemExit(f"{source.name}: categories differ from the checkpoints' order:\n"
                         f"  json:       {names}\n  checkpoint: {expected}")
    index = {c["id"]: i for i, c in enumerate(categories)}

    boxes = defaultdict(list)
    crowd = tiny = 0
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            crowd += 1
            continue
        boxes[a["image_id"]].append(a)

    out_dir = root / "labels" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    written = background = kept = 0
    for image in coco["images"]:
        w, h = image["width"], image["height"]
        lines = []
        for a in boxes.get(image["id"], []):
            x, y, bw, bh = a["bbox"]
            x1, y1 = max(x, 0.0), max(y, 0.0)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 - x1 < 1 or y2 - y1 < 1:
                tiny += 1
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            lines.append(f"{index[a['category_id']]} {cx:.6f} {cy:.6f}"
                         f" {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (out_dir / f"{Path(image['file_name']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
        background += not lines
        kept += len(lines)
    return {"images": written, "boxes": kept, "background": background,
            "crowd_skipped": crowd, "tiny_skipped": tiny}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="the COCO folder (images/, annotations/)")
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    names = coco_names()
    for split in args.splits:
        stats = convert_split(root, split, names)
        print(f"{split}: " + ", ".join(f"{k} {v}" for k, v in stats.items()))
    splits = {"train": "images/train2017", "val": "images/val2017"}
    lines = [f"path: {root}", f"train: {splits['train']}", f"val: {splits['val']}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
