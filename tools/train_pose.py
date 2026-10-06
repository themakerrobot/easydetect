#!/usr/bin/env python3
# Apache-2.0
"""Train easydetect's keypoint model on COCO 2017 person keypoints.

    pip install "easydetect[train]"
    python tools/train_pose.py --coco ~/datasets/coco            # roughly 4-5 h, one RTX 5090
    python tools/train_pose.py --coco ~/datasets/coco --resume runs/pose/s
    python tools/train_pose.py --export runs/pose/s/best.pt      # -> pose-s.onnx beside it
    python tools/train_pose.py --coco ~/datasets/coco --eval runs/pose/s/pose-s.onnx \
        --detector dfine-m                                       # the whole pipeline

``--coco`` holds ``images/{train,val}2017`` and
``annotations/person_keypoints_{train,val}2017.json`` (COCO's
annotations_trainval2017.zip). The network (easydetect/nn/posenet.py) starts
from the D-FINE detector's COCO backbone of the same size, so it already knows
what people look like. Every ``--val-every`` epochs the EMA weights are scored
on val2017 with the labelled boxes (OKS AP, easydetect.pose.KeypointAP); the
best is kept as ``best.pt`` and, at the end, exported to ``pose-<size>.onnx``
and checked against PyTorch on ONNX Runtime.

Only COCO's keypoint labels (CC BY 4.0) and its pictures go into this model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# the training loop, scoring and export live in the package (Detector.train
# uses them for keypoint datasets of your own); names kept here for scripts
from easydetect.pose_trainer import (  # noqa: E402, F401
    CSV_FIELDS,
    append_row,
    build,
    evaluate,
    export,
    lr_at,
    param_groups,
    train_keypoints,
)


def evaluate_pipeline(coco: str, onnx: Path, detector: str, device: str = "AUTO",
                      limit: int | None = None, flip: bool = True,
                      subpixel: bool = True) -> dict:
    """OKS AP on val2017 the way it is used: the detector's person boxes, then
    keypoints in each — every picture, people missed and false boxes included.
    A person's score is its box confidence times its mean keypoint confidence.

    ``detector="gt"`` takes the labelled boxes instead (confidence 1): the
    keypoint model alone, as training scores it, but through the exported
    model and the runtime's decoding (``flip``, ``subpixel``)."""
    import cv2

    from easydetect import Detector
    from easydetect.data.keypoints import load_coco
    from easydetect.pose import KeypointAP, KeypointEstimator, person_rows

    images, people = load_coco(Path(coco), "val")
    gt_boxes = detector == "gt"
    model = None if gt_boxes else Detector(detector, device=device, verbose=False)
    estimator = KeypointEstimator(onnx, device="CPU" if device == "AUTO" else device,
                                  flip=flip, subpixel=subpixel)
    ap = KeypointAP()
    ids = sorted(images)[:limit] if limit else sorted(images)
    started = time.time()
    for k, image_id in enumerate(ids):
        anns = people.get(image_id, [])
        if gt_boxes:
            labelled = [a for a in anns if not a.get("iscrowd") and a.get("num_keypoints", 0) > 0
                        and a["bbox"][2] > 1 and a["bbox"][3] > 1]
            if not labelled:
                continue
            img = cv2.imread(str(images[image_id]))
            boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in
                              (a["bbox"] for a in labelled)], np.float32)
            box_conf = np.ones(len(boxes), np.float32)
        else:
            img = cv2.imread(str(images[image_id]))
            r = model(img, conf=0.05, iou=0.7, max_det=100, verbose=False)[0]
            rows = person_rows(r.names, r.boxes.cls)
            boxes, box_conf = r.boxes.xyxy[rows], r.boxes.conf[rows]
        xy, conf = estimator(img, boxes)
        score = np.array([b * (c[c > 0.2].mean() if (c > 0.2).any() else 0.0)
                          for b, c in zip(box_conf, conf, strict=True)])
        gt = np.array([a["keypoints"] for a in anns], np.float64).reshape(len(anns), 17, 3)
        ap.add(xy, score, gt, np.array([a["area"] for a in anns]),
               np.array([a.get("iscrowd", 0) for a in anns], bool))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(ids)}  {time.time() - started:.0f}s", flush=True)
    result = ap.compute()
    how = ("flip, " if flip else "") + ("sub-pixel" if subpixel else "argmax")
    print(f"{onnx.name} on {detector} boxes ({how}), {len(ids)} pictures: "
          + ", ".join(f"{k} {v:.4f}" for k, v in result.items()))
    return result



def train(args) -> None:
    """COCO's person keypoints, through easydetect.pose_trainer."""
    from easydetect.data.keypoints import KeypointDataset

    out = Path(args.resume or Path(args.out) / args.size)
    train_ds = KeypointDataset(args.coco, "train", augment=True, limit=args.limit)
    val_ds = KeypointDataset(args.coco, "val", augment=False, limit=args.val_limit)
    train_keypoints(train_ds, val_ds, out, size=args.size, init=args.init, epochs=args.epochs,
                    batch=args.batch, lr=args.lr, weight_decay=args.weight_decay,
                    warmup=args.warmup, clip=args.clip, workers=args.workers,
                    val_every=args.val_every, freeze=args.freeze, hours=args.hours,
                    device=args.device, amp=not args.no_amp, seed=args.seed,
                    resume=bool(args.resume), log=lambda m: print(m, flush=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coco", help="COCO 2017 folder (images/, annotations/)")
    p.add_argument("--size", default="s", choices=["s", "m", "l"],
                   help="backbone and starting weights of dfine-s, dfine-m or dfine-l")
    p.add_argument("--init", help="dfine-s / dfine-m (default: same size), imagenet, none, "
                                  "a D-FINE .pt, or a keypoint best.pt to train further")
    p.add_argument("--epochs", type=int, default=210)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1000, help="steps")
    p.add_argument("--clip", type=float, default=3.0,
                   help="gradient norm cap: typical steps measure 1.4-3.4 (COCO, batch 64), "
                        "so 3 trims the rare spikes and leaves ordinary steps alone")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--freeze", type=int, default=0,
                   help="keep the stem and this many backbone stages as they start")
    p.add_argument("--hours", type=float,
                   help="stop after the last epoch that ends within this time (resume later)")
    p.add_argument("--device")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/pose")
    p.add_argument("--resume", help="a run folder to continue (its last.pt)")
    p.add_argument("--limit", type=int, help="train on the first N people (a smoke test)")
    p.add_argument("--val-limit", type=int, help="score on the first N people")
    p.add_argument("--export", type=Path, help="only export this best.pt to ONNX")
    p.add_argument("--eval", type=Path, help="only score this .onnx behind a detector")
    p.add_argument("--detector", default="dfine-m",
                   help="the detector --eval runs first; gt: the labelled boxes")
    p.add_argument("--no-flip", action="store_true",
                   help="--eval: read each crop once, not also mirrored (as pose_flip=False)")
    p.add_argument("--no-subpixel", action="store_true",
                   help="--eval: decode to the peak bin only")
    args = p.parse_args(argv)
    if args.export:
        export(args.export)
        return 0
    if args.eval:
        if not args.coco:
            p.error("--coco is required to evaluate")
        evaluate_pipeline(args.coco, args.eval, args.detector, args.device or "AUTO",
                          args.val_limit, flip=not args.no_flip, subpixel=not args.no_subpixel)
        return 0
    if not args.coco:
        p.error("--coco is required to train")
    args.init = args.init or f"dfine-{args.size}"
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
