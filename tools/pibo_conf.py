#!/usr/bin/env python3
# Apache-2.0
"""The score threshold a Pibo model should use, picked on validation pictures
and written into the .onnx (``easydetect.conf``, which openpibo reads).

    python tools/pibo_conf.py model.onnx --data ~/datasets/coco/data.yaml

Pictures go through what Pibo does (squeezed to the model's square, each
query's best class, boxes overlapping a better one by IoU > 0.7 dropped, any
class); the threshold kept is the one with the best F1 (a box right when it
meets an unmatched true box of its class at IoU 0.5 or more), among 0.25,
0.30 ... 0.70, the higher on a tie.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STEPS = [round(0.25 + 0.05 * i, 2) for i in range(10)]


def _xyxy(c):
    return np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2,
                     c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], 1)


def _iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def pick(onnx: Path, data: str, limit: int | None = 2000, threads: int = 4) -> tuple[float, dict]:
    """(best threshold, {threshold: (precision, recall, F1)})."""
    import cv2
    import onnxruntime as ort

    from easydetect.data.dataset import DetDataset

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = threads
    s = ort.InferenceSession(str(onnx), opt, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    size = int(inp.shape[2])
    outs = [o.name for o in s.get_outputs()]
    ds = DetDataset(data, "val", 64, augment=False)
    files = ds.files[:limit] if limit else ds.files
    tp = {c: 0 for c in STEPS}
    found = {c: 0 for c in STEPS}
    truths = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        x = cv2.resize(img, (size, size))[:, :, ::-1].transpose(2, 0, 1)[None]
        res = dict(zip(outs, s.run(None, {inp.name: x.astype(np.float32) / 255}), strict=True))
        boxes, scores = res["boxes"][0], res["scores"][0]
        if scores.min() < 0 or scores.max() > 1:
            scores = 1 / (1 + np.exp(-scores))
        lab = ds._load_labels(f)
        gt, gc = _xyxy(lab[:, 1:5]), lab[:, 0].astype(int)
        truths += len(gt)
        cls, best = scores.argmax(-1), scores.max(-1)
        order = np.argsort(-best)
        b, cls, best = _xyxy(boxes)[order], cls[order], best[order]
        for c in STEPS:
            keep = best >= c
            bb, cc = b[keep], cls[keep]
            if len(bb) > 1:
                m = _iou(bb, bb)
                alive = np.ones(len(bb), bool)
                for j in range(len(bb)):
                    if alive[j]:
                        alive[j + 1:] &= ~(m[j, j + 1:] > 0.7)
                bb, cc = bb[alive], cc[alive]
            found[c] += len(bb)
            if not len(bb) or not len(gt):
                continue
            m = _iou(bb, gt)
            used = np.zeros(len(gt), bool)
            for j in range(len(bb)):
                cand = (m[j] >= 0.5) & (gc == cc[j]) & ~used
                if cand.any():
                    used[np.argmax(np.where(cand, m[j], -1))] = True
                    tp[c] += 1
    table = {c: (tp[c] / max(found[c], 1), tp[c] / max(truths, 1),
                 2 * tp[c] / max(found[c] + truths, 1)) for c in STEPS}
    best_c = max(STEPS, key=lambda c: (round(table[c][2], 6), c))
    return best_c, table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("onnx")
    parser.add_argument("--data", required=True, help="data.yaml with a val split")
    parser.add_argument("--limit", type=int, default=2000, help="val pictures used (0: all)")
    parser.add_argument("--dry-run", action="store_true", help="show it, write nothing")
    args = parser.parse_args()
    from easydetect.exporter import embed_metadata

    best, table = pick(Path(args.onnx), args.data, args.limit or None)
    for c, (p, r, f1) in table.items():
        print(f"conf {c:.2f}  precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}"
              + ("  <-" if c == best else ""))
    if not args.dry_run:
        embed_metadata(args.onnx, {"easydetect.conf": f"{best:.2f}"})
        print(f"easydetect.conf = {best:.2f} written into {args.onnx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
