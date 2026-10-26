#!/usr/bin/env python3
# Apache-2.0
"""A YOLO .onnx and a ed checkpoint on the same val images, scored by one evaluator.

    python tools/compare_yolo.py --data ~/datasets/coco/data.yaml \
        --yolo yolo11s.onnx --ed runs/ed-n-320/weights/best.pt

Both run on OpenVINO (CPU by default), each with the preprocessing it was
trained for: YOLO letterboxed (aspect kept, grey padding) then class-wise NMS
at IoU 0.7; ed plain-resized, no NMS. Boxes from both go through
easydetect's Evaluator (COCO-style mAP50 / mAP50-95, the same code as
`easydetect val`), at conf 0.001 and up to 300 boxes a picture.

The YOLO side reads the exported ONNX's published layouts — boxes as centre,
width, height in letterboxed input pixels then one score per class (YOLOv8,
YOLO11), or, from an end-to-end model (YOLO26), up to 300 rows of corners,
score and class with no NMS to run; it is written here from that
description, with no Ultralytics code. Its numbers
can differ a little from Ultralytics' own `val`, which batches pictures of
similar shape; the point is that both models are measured the same way.
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) / 2, (size - nh) / 2
    out = cv2.copyMakeBorder(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR),
                             round(py - 0.1), round(py + 0.1), round(px - 0.1), round(px + 0.1),
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, px, py


class Yolo:
    """An exported YOLO ONNX: input 1x3xSxS RGB 0..1, output 1 x (4+classes) x anchors."""

    # multi_label: every class above conf counts, each box once per class (how
    # Ultralytics' own validator scores); otherwise one class a box, as predict gives
    multi_label = False

    def __init__(self, path, device, size=320):
        import openvino as ov

        core = ov.Core()
        model = core.read_model(path)
        shape = model.input(0).get_partial_shape()
        if shape[2].is_static:                   # exported at a fixed size: that is the size
            self.size = int(shape[2].get_length())
        else:                                    # exported with dynamic=True: pin it here
            self.size = size
            model.reshape({model.input(0): [1, 3, size, size]})
        self.compiled = core.compile_model(model, device)
        self.request = self.compiled.create_infer_request()

    def __call__(self, img, conf=0.001, iou=0.7, max_det=300):
        h, w = img.shape[:2]
        boxed, r, px, py = letterbox(img, self.size)
        x = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        t = time.perf_counter()
        out = self.request.infer({0: x})[self.compiled.output(0)][0]
        infer_ms = (time.perf_counter() - t) * 1000
        # end-to-end (YOLO26, YOLOv10): rows of x1 y1 x2 y2 score class, no NMS to run
        if out.ndim == 2 and out.shape[1] == 6:
            out = out[out[:, 4] >= conf][:max_det]
            xyxy = np.stack([(out[:, 0] - px) / r, (out[:, 1] - py) / r,
                             (out[:, 2] - px) / r, (out[:, 3] - py) / r], 1)
            xyxy = np.clip(xyxy, 0, [w, h, w, h])
            return xyxy / [w, h, w, h], out[:, 4], out[:, 5].astype(int), infer_ms
        scores = out[4:].T                                                # (anchors, nc)
        if self.multi_label:
            anchor, cls = np.nonzero(scores >= conf)
            best = scores[anchor, cls]
        else:
            cls = scores.argmax(1)
            best = scores[np.arange(len(cls)), cls]
            anchor = np.nonzero(best >= conf)[0]
            best, cls = best[anchor], cls[anchor]
        cx, cy, bw, bh = out[:4, anchor]
        xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
        picked = cv2.dnn.NMSBoxesBatched(xywh.tolist(), best.tolist(), cls.tolist(), conf, iou)
        picked = np.asarray(picked, int).reshape(-1)[:max_det]
        xywh, best, cls = xywh[picked], best[picked], cls[picked]
        xyxy = np.stack([(xywh[:, 0] - px) / r, (xywh[:, 1] - py) / r,
                         (xywh[:, 0] + xywh[:, 2] - px) / r, (xywh[:, 1] + xywh[:, 3] - py) / r], 1)
        xyxy = np.clip(xyxy, 0, [w, h, w, h])
        return xyxy / [w, h, w, h], best, cls, infer_ms


class EdRunner:
    # multi_label: the 300 best (query, class) pairs, as the original evaluation
    # takes them; otherwise one class a query, as predict gives
    multi_label = False

    def __init__(self, path, device):
        from easydetect import Detector

        self.model = Detector(path, device=device, verbose=False)
        self.model(np.zeros((64, 64, 3), np.uint8))                          # build the IR once
        self.size = self.model.predictor.imgsz

    def __call__(self, img, conf=0.001, max_det=300):
        h, w = img.shape[:2]
        if self.multi_label:
            from easydetect.predictor import _looks_like_logits
            from easydetect.utils.ops import cxcywh2xyxy_np

            predictor = self.model.predictor
            tensor = predictor.preprocess(img)
            t = time.perf_counter()
            boxes, scores = predictor.infer(tensor)
            ms = (time.perf_counter() - t) * 1000
            if _looks_like_logits(scores):
                scores = 1.0 / (1.0 + np.exp(-scores))
            nc = scores.shape[-1]
            flat = scores.reshape(-1)
            top = np.argsort(-flat)[:max_det]
            top = top[flat[top] >= conf]
            xyxy = np.clip(cxcywh2xyxy_np(boxes[top // nc].astype(np.float32)), 0, 1)
            return xyxy, flat[top], (top % nc).astype(int), ms
        r = self.model.predict(img, conf=conf, max_det=max_det, iou=None, verbose=False)[0]
        return (r.boxes.xyxy / [w, h, w, h], r.boxes.conf, r.boxes.cls.astype(int),
                r.speed["inference"])


def evaluate(name, runner, files, truth, nc):
    from easydetect.validator import Evaluator

    ev, times = Evaluator(nc), []
    for i, (f, (gt_xyxy, gt_cls)) in enumerate(zip(files, truth, strict=True)):
        boxes, conf, cls, ms = runner(cv2.imread(str(f)))
        times.append(ms)
        ev.add(boxes, conf, cls, gt_xyxy, gt_cls)
        if (i + 1) % 1000 == 0:
            print(f"  {name}: {i + 1}/{len(files)}", flush=True)
    m = ev.compute()
    return m["map50"], m["map"], statistics.median(times[5:] or times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--yolo", help="yolo .onnx")
    parser.add_argument("--ed", nargs="*", default=[], help="ed .pt / .xml (one or more)")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--size", type=int, default=320,
                        help="YOLO input size when the ONNX was exported with a dynamic one")
    parser.add_argument("--limit", type=int, help="only the first N pictures (a quick look)")
    args = parser.parse_args()

    from pathlib import Path

    from easydetect.data.dataset import DetDataset

    ds = DetDataset(Path(args.data).expanduser(), "val", 64, augment=False)
    files = ds.files[:args.limit] if args.limit else ds.files
    truth = []
    for f in files:
        lab = ds._load_labels(f)
        cx, cy, bw, bh = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
        truth.append((np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1),
                      lab[:, 0].astype(int)))
    print(f"{len(files)} pictures, {sum(len(t[1]) for t in truth)} boxes, device {args.device}")

    rows = []
    if args.yolo:
        runner = Yolo(args.yolo, args.device, args.size)
        rows.append((Path(args.yolo).name, runner.size,
                     *evaluate("yolo", runner, files, truth, ds.nc)))
    for path in args.ed:
        runner = EdRunner(path, args.device)
        rows.append((str(Path(path)), runner.size, *evaluate("ed", runner, files, truth, ds.nc)))

    print(f"\n{'model':<40} {'input':>5} {'mAP50':>7} {'mAP50-95':>9} {'model ms':>9}")
    for name, size, m50, m, ms in rows:
        print(f"{name[-40:]:<40} {size:>5} {m50:7.3f} {m:9.3f} {ms:9.1f}")


if __name__ == "__main__":
    main()
