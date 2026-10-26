#!/usr/bin/env python3
# Apache-2.0
"""Lighter exports side by side: fewer queries or decoder layers, INT8.

    python tools/eval_exports.py --data ~/datasets/coco/data.yaml --model ed-n --imgsz 320
    python tools/eval_exports.py --data data.yaml --model runs/train/weights/best.pt

Each config (``layers:queries[:int8]``, ``all`` for every layer) is exported
from the same checkpoint (a mirror name's COCO .pt, or yours) and scored on
the same val pictures: mAP50-95 / mAP50 with every box (as ``val`` does),
precision and recall of what ``predict`` shows by default (conf 0.5, NMS 0.7),
and the median inference time. INT8 calibrates on the first ``--calib`` val
pictures, which are then left out of the scoring for every config alike.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from easydetect import Detector, downloads
from easydetect.data.dataset import DetDataset
from easydetect.predictor import drop_duplicates
from easydetect.validator import Evaluator

DEFAULT_CONFIGS = ["all:300", "all:100", "2:300", "1:300", "1:100", "all:300:int8", "all:100:int8"]


def box_iou(x, y):
    tl = np.maximum(x[:, None, :2], y[None, :, :2])
    br = np.minimum(x[:, None, 2:], y[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    ax = np.prod(x[:, 2:] - x[:, :2], axis=1)
    ay = np.prod(y[:, 2:] - y[:, :2], axis=1)
    return inter / np.maximum(ax[:, None] + ay[None, :] - inter, 1e-9)


def shown_and_correct(r, gt, gk, w, h):
    """Boxes predict() shows by default (conf 0.5, NMS 0.7) and how many match a truth box."""
    keep = r.boxes.conf >= 0.5
    xyxy, cls = r.boxes.xyxy[keep].astype(np.float32), r.boxes.cls[keep].astype(int)
    if len(xyxy) > 1:
        kept = drop_duplicates(xyxy, 0.7)
        xyxy, cls = xyxy[kept], cls[kept]
    if not len(xyxy) or not len(gt):
        return len(xyxy), 0
    ious = box_iou(xyxy / [w, h, w, h], gt)
    taken = np.zeros(len(gt), bool)
    correct = 0
    for i in range(len(xyxy)):
        cand = np.where((gk == cls[i]) & ~taken & (ious[i] >= 0.5))[0]
        if len(cand):
            taken[cand[ious[i, cand].argmax()]] = True
            correct += 1
    return len(xyxy), correct


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="ed-n", help="a mirror name (its COCO .pt) or a .pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help="layers:queries[:int8]; 'all' = every decoder layer")
    parser.add_argument("--calib", type=int, default=300, help="pictures to calibrate INT8 on")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--limit", type=int, help="score only the first N pictures")
    parser.add_argument("--out", default="exports", help="where the exported variants go")
    args = parser.parse_args()

    pt = Path(args.model)
    if pt.suffix != ".pt":
        pt = downloads.download_checkpoint(args.model)
    ds = DetDataset(Path(args.data).expanduser(), "val", 64, augment=False)
    calib, files = ds.files[:args.calib], ds.files[args.calib:]
    files = files[:args.limit] if args.limit else files
    truth = []
    for f in files:
        lab = ds._load_labels(f)
        cx, cy, bw, bh = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
        truth.append((np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1),
                      lab[:, 0].astype(int)))
    n_truth = sum(len(t[1]) for t in truth)

    rows = []
    for config in args.configs:
        layers, queries, *rest = config.split(":")
        int8 = rest == ["int8"]
        out = Path(args.out) / f"{pt.stem}-{args.imgsz}-{config.replace(':', '-')}"
        xml = Detector(str(pt), verbose=False).export(
            format="openvino", imgsz=args.imgsz, out_dir=out,
            layers=None if layers == "all" else int(layers), queries=int(queries),
            int8=int8, data=calib if int8 else None, calib=args.calib, verbose=False)
        model = Detector(str(xml), device=args.device, verbose=False)
        for _ in range(3):                   # compile and warm up before the clock runs
            model(cv2.imread(str(files[0])), verbose=False)
        ev, shown, correct, times = Evaluator(ds.nc), 0, 0, []
        for k, (f, (gt, gk)) in enumerate(zip(files, truth, strict=True)):
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            r = model(img, conf=0.001, iou=None, max_det=300, verbose=False)[0]
            times.append(r.speed["inference"])
            ev.add(r.boxes.xyxy / [w, h, w, h], r.boxes.conf, r.boxes.cls.astype(int), gt, gk)
            s, c = shown_and_correct(r, gt, gk, w, h)
            shown, correct = shown + s, correct + c
            if (k + 1) % 1000 == 0:
                print(f"  {config}: {k + 1}/{len(files)}", flush=True)
        m = ev.compute()
        rows.append((config, m["map"], m["map50"], correct / max(shown, 1),
                     correct / max(n_truth, 1), float(np.median(times))))
        print(f"  {config}: mAP50-95 {m['map']:.4f}, {rows[-1][-1]:.1f} ms", flush=True)

    print(f"\n{pt.stem} at {args.imgsz}, {len(files)} pictures, {args.device}")
    print(f"{'config':<16}{'mAP50-95':>9}{'mAP50':>8}{'P@0.5':>8}{'R@0.5':>8}{'ms':>8}")
    for config, m, m50, precision, recall, ms in rows:
        print(f"{config:<16}{m:9.4f}{m50:8.4f}{precision:8.1%}{recall:8.1%}{ms:8.1f}")


if __name__ == "__main__":
    main()
