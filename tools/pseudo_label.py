#!/usr/bin/env python3
# Apache-2.0
"""COCO's 80 classes boxed on other pictures by a large COCO model: more
pictures to train a small one on, with no Objects365 (whose licence we avoid).

    python tools/pseudo_label.py --images ~/datasets/openimages/images/train \\
        --coco ~/datasets/coco --out ~/datasets/oi-pseudo --limit 300000

The teacher (``ed-x`` by default, COCO 55.8) looks at each picture at 640 px;
every query whose best class scores ``--conf`` or more becomes a box. Pictures
with no box are left out. ``<out>/images/train`` is a link to the pictures'
folder, ``<out>/labels/train/<stem>.txt`` the boxes, ``<out>/train.txt`` the
pictures with boxes, and ``<out>/data.yaml`` trains on COCO's training
pictures and these together, scored on COCO val.
A rerun skips pictures already labelled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pictures(folder: Path, limit: int | None) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
    if limit and len(files) > limit:
        pick = np.random.default_rng(0).choice(len(files), limit, replace=False)
        files = [files[i] for i in sorted(pick)]
    return files


class Pictures:
    """Each picture squeezed to size x size (as easydetect trains), RGB 0..1."""

    def __init__(self, files: list[Path], size: int):
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import cv2
        import torch

        img = cv2.imread(str(self.files[i]))
        if img is None:
            return torch.zeros(3, self.size, self.size), i, False
        x = cv2.resize(img, (self.size, self.size))[:, :, ::-1].copy()
        return torch.from_numpy(x).permute(2, 0, 1).float().div_(255), i, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", required=True, help="a folder of pictures")
    parser.add_argument("--coco", default="~/datasets/coco", help="COCO (data.yaml beside)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher", default="ed-x", help="a COCO model name or .pt")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, help="this many pictures, picked at random")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    import torch
    import yaml

    from easydetect import Detector

    src = Path(args.images).expanduser().resolve()
    out = Path(args.out).expanduser()
    labels = out / "labels" / "train"
    labels.mkdir(parents=True, exist_ok=True)
    link = out / "images" / "train"
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, link, target_is_directory=True)

    files = pictures(src, args.limit)
    todo = [f for f in files if not (labels / f"{f.stem}.txt").exists()
            and not (labels / f"{f.stem}.none").exists()]
    print(f"{len(files)} pictures, {len(todo)} to label", flush=True)

    teacher = Detector(args.teacher, verbose=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = teacher._ensure_net().eval().to(device)
    names = {int(k): v for k, v in teacher.names.items()}
    loader = torch.utils.data.DataLoader(Pictures(todo, args.imgsz), batch_size=args.batch,
                                         num_workers=args.workers, pin_memory=True)
    boxes_seen = np.zeros(len(names), np.int64)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for n, (x, idx, ok) in enumerate(loader):
            o = net(x.to(device, non_blocking=True))
            prob = o["pred_logits"].float().sigmoid()
            score, cls = prob.max(-1)
            box = o["pred_boxes"].float().clamp(0, 1)
            for j in range(len(idx)):
                f = todo[int(idx[j])]
                keep = (score[j] >= args.conf) & bool(ok[j])
                if not keep.any():
                    (labels / f"{f.stem}.none").touch()   # looked at, nothing to box
                    continue
                c, b = cls[j][keep].cpu().numpy(), box[j][keep].cpu().numpy()
                np.add.at(boxes_seen, c, 1)
                rows = (f"{k} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n"
                        for k, (cx, cy, w, h) in zip(c, b, strict=True))
                (labels / f"{f.stem}.txt").write_text("".join(rows))
            if n % 200 == 0:
                print(f"{(n + 1) * args.batch}/{len(todo)}", flush=True)

    # the pictures with boxes, by way of the link (so their labels are ours):
    # a picture not looked at, or with nothing boxed, is not taught as empty
    boxed = sorted(p.stem for p in labels.glob("*.txt"))
    by_stem = {f.stem: f.name for f in files}
    (out / "train.txt").write_text("".join(f"{link / by_stem[s]}\n" for s in boxed
                                           if s in by_stem))
    coco = Path(args.coco).expanduser()
    cfg = yaml.safe_load((coco / "data.yaml").read_text())
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out),
        "train": [str(coco / cfg["train"]), str(out / "train.txt")],
        "val": str(coco / cfg["val"]),
        "names": names,
    }, allow_unicode=True, sort_keys=False))
    labelled = sum(1 for _ in labels.glob("*.txt"))
    (out / "pseudo.json").write_text(json.dumps({
        "teacher": args.teacher, "conf": args.conf, "imgsz": args.imgsz,
        "pictures_with_boxes": labelled,
        "boxes_this_run": {names[i]: int(v) for i, v in enumerate(boxes_seen) if v}}, indent=1))
    print(f"{labelled} pictures with boxes; {out / 'data.yaml'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
