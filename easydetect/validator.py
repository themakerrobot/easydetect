# Apache-2.0
"""Validation: mAP@50 and mAP@50-95 as COCO computes them, no pycocotools needed.

Like the original evaluation: each picture's 300 best (query, class) pairs —
a query may stand for two classes — scored with COCO's 101-point precision
over up to 100 boxes a picture."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import DetDataset, worker_context
from .utils.ops import cxcywh2xyxy


def _iou_np(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None] - inter, 1e-9, None)


def _ap(recall, precision):
    """COCO's: the best precision at or beyond each of 101 recall points, averaged."""
    if not len(precision):
        return 0.0
    p = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0.0).mean())


class Evaluator:
    """Accumulates (pred, gt) pairs per image; computes mAP over IoU 0.5:0.95.

    Up to ``max_det`` boxes a picture count, the highest-scoring ones (COCO's
    maxDets 100)."""

    def __init__(self, num_classes, max_det=100):
        self.nc = num_classes
        self.max_det = max_det
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.records = []  # per image: (pred xyxy, pred conf, pred cls, gt xyxy, gt cls)

    def add(self, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls):
        # copies, never views: an array that views a tensor from a DataLoader
        # worker keeps that worker's shared-memory file open, two per image —
        # 10,000 for COCO val, past Linux's usual limit of 1,024 descriptors
        conf = np.array(pred_conf, np.float32).reshape(-1)
        top = np.argsort(-conf, kind="stable")[: self.max_det]
        self.records.append(
            (
                np.array(pred_boxes, np.float32).reshape(-1, 4)[top],
                conf[top],
                np.array(pred_cls, np.int64).reshape(-1)[top],
                np.array(gt_boxes, np.float32).reshape(-1, 4),
                np.array(gt_cls, np.int64).reshape(-1),
            )
        )

    def compute(self):
        aps = np.zeros((self.nc, len(self.iouv)))
        seen = np.zeros(self.nc, bool)
        for c in range(self.nc):
            scores, matches, n_gt = [], [], 0
            for pb, pc, pk, gb, gk in self.records:
                gsel = gb[gk == c]
                n_gt += len(gsel)
                sel = pk == c
                pbc, pcc = pb[sel], pc[sel]
                order = np.argsort(-pcc)
                pbc, pcc = pbc[order], pcc[order]
                iou = _iou_np(pbc, gsel)
                m = np.zeros((len(pbc), len(self.iouv)), bool)
                for t, thr in enumerate(self.iouv):
                    used = np.zeros(len(gsel), bool)
                    for i in range(len(pbc)):
                        if not len(gsel):
                            break
                        j = int(np.argmax(np.where(used, -1.0, iou[i])))
                        if iou[i, j] >= thr and not used[j]:
                            used[j] = True
                            m[i, t] = True
                scores.append(pcc)
                matches.append(m)
            if n_gt == 0:
                continue
            seen[c] = True
            scores = np.concatenate(scores) if scores else np.zeros(0)
            matches = np.concatenate(matches) if matches else np.zeros((0, len(self.iouv)), bool)
            order = np.argsort(-scores)
            matches = matches[order]
            for t in range(len(self.iouv)):
                tp = np.cumsum(matches[:, t])
                fp = np.cumsum(~matches[:, t])
                recall = tp / n_gt
                precision = tp / np.clip(tp + fp, 1e-9, None)
                aps[c, t] = _ap(recall, precision)
        per_class = {c: float(aps[c].mean()) for c in range(self.nc) if seen[c]}
        if not seen.any():
            return {"map50": 0.0, "map": 0.0, "per_class": {}}
        return {
            "map50": float(aps[seen, 0].mean()),
            "map": float(aps[seen].mean()),
            "per_class": per_class,
        }


def top_pairs(logits: torch.Tensor, boxes: torch.Tensor, k: int = 300):
    """The original post-processing: the ``k`` best (query, class) pairs of one
    picture — ``logits (Q, C)``, ``boxes (Q, 4)`` — as boxes, scores, classes."""
    scores = logits.sigmoid().flatten()
    top, index = scores.topk(min(k, scores.numel()))
    nc = logits.shape[-1]
    return boxes[index // nc], top, index % nc


@torch.no_grad()
def validate_torch(net, data_yaml, imgsz=640, batch=8, conf=0.0, device=None, workers=2):
    """Evaluate a torch EDNet on the val split. Returns metrics dict.

    On a GPU the network runs in bfloat16 (float16 where there is none) —
    scoring, not training, so the few thousandths it may move are noise."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = DetDataset(data_yaml, "val", imgsz, augment=False)
    if len(ds) <= batch * 4:
        workers = 0     # starting workers takes longer than reading a few pictures
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=DetDataset.collate,
                    pin_memory=device.type == "cuda",
                    multiprocessing_context=worker_context() if workers > 0 else None)
    net = net.to(device).eval()
    ev = Evaluator(net.num_classes)
    half = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16)
    for imgs, targets in dl:
        imgs = DetDataset.as_float(imgs.to(device, non_blocking=True))
        with torch.autocast(device.type, dtype=half, enabled=device.type == "cuda"):
            out = net(imgs)
        logits, boxes = out["pred_logits"].float(), cxcywh2xyxy(out["pred_boxes"].float())
        for b in range(imgs.shape[0]):
            bx, sc, cl = top_pairs(logits[b], boxes[b])
            keep = sc >= conf
            ev.add(
                bx[keep].cpu().numpy(),
                sc[keep].cpu().numpy(),
                cl[keep].cpu().numpy(),
                cxcywh2xyxy(targets[b]["boxes"]).numpy(),
                targets[b]["labels"].numpy(),
            )
    return ev.compute()
