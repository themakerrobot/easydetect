# Apache-2.0
"""Keypoints for the people a detector found: easydetect's own top-down model.

Each person box is widened to the network's 3:4 shape with a margin, cut out
and resized to 256x192, and the network (easydetect/nn/posenet.py, trained by
tools/train_pose.py on COCO 2017 keypoints) places 17 keypoints in the crop;
they are mapped back to the picture. This module needs only NumPy and OpenCV
plus a runtime (OpenVINO or ONNX Runtime): the crop geometry, decoding and the
OKS evaluator are shared with training.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

INPUT = (256, 192)  # crop height, width
SPLIT = 2  # the network places a keypoint to 1/SPLIT of a crop pixel
SIGMA = (5.66, 4.9)  # spread of the training targets in those bins (y, x)
PAD = 1.25  # margin around the box: limbs often stick out of it

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
#: index of each keypoint's mirror image, for horizontal flips
FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)
#: limbs to draw, as keypoint index pairs (COCO's skeleton)
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
)
#: COCO's per-keypoint OKS falloff: how far off a keypoint may be, relative to the
#: person's size, for each body part (an eye must be closer than a hip)
OKS_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                       1.07, 1.07, .87, .87, .89, .89], np.float64) / 10.0


# ---------------------------------------------------------------- crop geometry

def box_to_crop(xyxy: np.ndarray, pad: float = PAD) -> tuple[np.ndarray, np.ndarray]:
    """Boxes ``(N, 4)`` → crop centres ``(N, 2)`` and sizes ``(N, 2)`` (w, h) in
    picture pixels, widened to the network's aspect ratio and padded."""
    xyxy = np.asarray(xyxy, np.float64).reshape(-1, 4)
    center = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    w = np.maximum(xyxy[:, 2] - xyxy[:, 0], 1.0)
    h = np.maximum(xyxy[:, 3] - xyxy[:, 1], 1.0)
    aspect = INPUT[1] / INPUT[0]
    w, h = np.where(w > aspect * h, w, h * aspect), np.where(w > aspect * h, w / aspect, h)
    return center, np.stack([w, h], 1) * pad


def crop_matrix(center, size, rotation: float = 0.0) -> np.ndarray:
    """The 2x3 affine map from picture pixels to crop pixels: ``size`` (w, h)
    around ``center`` fills the crop, turned by ``rotation`` degrees."""
    out_h, out_w = INPUT
    a = np.deg2rad(rotation)
    cos, sin = np.cos(a), np.sin(a)
    sx, sy = out_w / size[0], out_h / size[1]
    m = np.array([[sx * cos, sx * sin, 0.0], [-sy * sin, sy * cos, 0.0]])
    m[:, 2] = np.array([out_w / 2, out_h / 2]) - m[:, :2] @ np.asarray(center, np.float64)
    return m


def invert(m: np.ndarray) -> np.ndarray:
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    return np.linalg.inv(full)[:2]


def apply(m: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Points ``(..., 2)`` through a 2x3 affine map."""
    return xy @ m[:, :2].T + m[:, 2]


def crop(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.warpAffine(img, m, (INPUT[1], INPUT[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ---------------------------------------------------------------- decoding

def _peak(sigma: float, bins: int) -> float:
    """The largest value a perfectly confident (target-shaped) distribution has."""
    d = np.arange(bins) - bins / 2
    g = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return float(g.max() / g.sum())


_PEAK_X = _peak(SIGMA[1], INPUT[1] * SPLIT)
_PEAK_Y = _peak(SIGMA[0], INPUT[0] * SPLIT)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def _refine(logits: np.ndarray) -> np.ndarray:
    """The peak bin, moved by a parabola through the log-probabilities of it and
    its two neighbours: exact for the Gaussian the network is trained to put
    out, and never more than half a bin from the peak."""
    peak = logits.argmax(-1)
    n = logits.shape[-1]
    inner = np.clip(peak, 1, n - 2)
    left = np.take_along_axis(logits, (inner - 1)[..., None], -1)[..., 0]
    mid = np.take_along_axis(logits, inner[..., None], -1)[..., 0]
    right = np.take_along_axis(logits, (inner + 1)[..., None], -1)[..., 0]
    curve = left - 2 * mid + right
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(curve < 0, 0.5 * (left - right) / curve, 0.0)
    offset = np.where((peak == inner), np.clip(offset, -0.5, 0.5), 0.0)
    return peak + offset


def decode(x_logits: np.ndarray, y_logits: np.ndarray,
           subpixel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Network output → keypoints in crop pixels ``(N, K, 2)`` and confidences
    ``(N, K)`` in 0-1: how close each distribution is to a confident one.
    ``subpixel`` places each keypoint between bins (a parabola through the
    peak) rather than on the peak's bin, half a crop pixel wide."""
    x_logits, y_logits = x_logits.astype(np.float32), y_logits.astype(np.float32)
    px, py = _softmax(x_logits), _softmax(y_logits)
    if subpixel:
        xy = np.stack([_refine(x_logits), _refine(y_logits)], -1).astype(np.float32) / SPLIT
    else:
        xy = np.stack([px.argmax(-1), py.argmax(-1)], -1).astype(np.float32) / SPLIT
    conf = np.minimum(px.max(-1) / _PEAK_X, py.max(-1) / _PEAK_Y)
    return xy, np.clip(conf, 0.0, 1.0).astype(np.float32)


def unflip(x_logits: np.ndarray, y_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Output for a mirrored crop → output in the original crop's terms: left
    and right keypoints swap, and x bin ``b`` of the mirror is the original's
    ``(W - 1) * SPLIT - b`` (pixel centres, as training flips them)."""
    x = x_logits[:, list(FLIP)][..., ::-1]
    x = np.concatenate([x[..., 1:], x[..., -1:]], -1)  # reversed is one bin off
    return x, y_logits[:, list(FLIP)]


# ---------------------------------------------------------------- the runtime

class KeypointEstimator:
    """``estimator(img, xyxy) -> (keypoints (N, 17, 2) picture pixels, conf (N, 17))``."""

    def __init__(self, model: str | Path, device: str = "CPU", backend: str | None = None,
                 batch: int = 16, flip: bool = False, subpixel: bool = True) -> None:
        """``flip`` also reads each crop mirrored and averages the two (twice the
        work); ``subpixel`` decodes finer than the network's half-pixel bins."""
        from .predictor import installed

        self.backend = backend or ("openvino" if installed("openvino") else "onnxruntime")
        self.batch = batch
        self.flip, self.subpixel = flip, subpixel
        self._local = threading.local()
        if self.backend == "openvino":
            import openvino as ov

            self._model = ov.Core().compile_model(str(model), device)
        else:
            import onnxruntime as ort

            self._model = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def _infer(self, crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.backend == "onnxruntime":
            x, y = self._model.run(None, {"image": crops})
            return x, y
        if not hasattr(self._local, "request"):
            self._local.request = self._model.create_infer_request()
        out = self._local.request.infer({"image": crops})
        return out[self._model.outputs[0]], out[self._model.outputs[1]]

    @staticmethod
    def preprocess(img: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """BGR picture + boxes → ``(N, 3, 256, 192)`` RGB 0-255 crops and their maps."""
        import cv2

        centers, sizes = box_to_crop(xyxy)
        maps = [crop_matrix(c, s) for c, s in zip(centers, sizes, strict=True)]
        crops = np.stack([cv2.cvtColor(crop(img, m), cv2.COLOR_BGR2RGB) for m in maps])
        return crops.transpose(0, 3, 1, 2).astype(np.float32), maps

    def __call__(self, img: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        if not len(xyxy):
            return np.zeros((0, len(KEYPOINT_NAMES), 2), np.float32), \
                np.zeros((0, len(KEYPOINT_NAMES)), np.float32)
        crops, maps = self.preprocess(img, xyxy)
        xs, ys = [], []
        for i in range(0, len(crops), self.batch):
            part = crops[i:i + self.batch]
            x, y = self._infer(np.ascontiguousarray(part))
            if self.flip:
                fx, fy = unflip(*self._infer(np.ascontiguousarray(part[..., ::-1])))
                x, y = (x + fx) / 2, (y + fy) / 2
            xs.append(x)
            ys.append(y)
        xy, conf = decode(np.concatenate(xs), np.concatenate(ys), subpixel=self.subpixel)
        back = [apply(invert(m), p) for m, p in zip(maps, xy, strict=True)]
        return np.stack(back).astype(np.float32), conf


def default_estimator(device: str = "CPU", backend: str | None = None) -> KeypointEstimator:
    """The mirror's keypoint model, downloaded once into the cache."""
    from .downloads import download_pose

    return KeypointEstimator(download_pose(), device=device, backend=backend)


def person_rows(names: dict[int, str], cls: np.ndarray) -> np.ndarray:
    """Which boxes get keypoints: those named "person" — or every box, when the
    model has no such class (a model trained on one kind of person, "worker")."""
    person = [k for k, v in names.items() if str(v).lower() == "person"]
    if not person:
        return np.ones(len(cls), bool)
    return np.isin(cls.astype(int), person)


# ---------------------------------------------------------------- evaluation

def oks(pred: np.ndarray, gt: np.ndarray, area: float) -> float:
    """Object keypoint similarity of one prediction ``(K, 2)`` to one labelled
    person ``(K, 3)`` (x, y, visibility) of segment ``area``, as COCO defines it."""
    vis = gt[:, 2] > 0
    if not vis.any():
        return 0.0
    d2 = ((pred[:, :2] - gt[:, :2]) ** 2).sum(1)
    e = d2 / ((2 * OKS_SIGMAS) ** 2) / (area + np.spacing(1)) / 2
    return float(np.exp(-e[vis]).mean())


class KeypointAP:
    """COCO-style keypoint AP (OKS 0.50:0.95, 101-point precision, up to 20 people
    a picture, people without labelled keypoints ignored). Written to the
    published definition for easydetect; within a few tenths of pycocotools,
    which also gives crowd regions their own treatment."""

    THRESHOLDS = np.linspace(0.5, 0.95, 10)

    def __init__(self, max_det: int = 20) -> None:
        self.max_det = max_det
        self.scores: list[np.ndarray] = []
        self.matched: list[np.ndarray] = []  # (thresholds, dets): 1 tp, 0 fp, -1 ignored
        self.positives = 0

    def add(self, pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, areas: np.ndarray,
            ignore: np.ndarray | None = None) -> None:
        """One picture: ``pred (D, K, 2)``, ``scores (D,)``, ``gt (G, K, 3)``,
        ``areas (G,)``; ``ignore`` marks crowds (and people with no keypoints
        labelled are ignored anyway)."""
        gt = np.asarray(gt, np.float64).reshape(-1, len(OKS_SIGMAS), 3)
        ig = (gt[:, :, 2] > 0).sum(1) == 0
        if ignore is not None:
            ig |= np.asarray(ignore, bool)
        order_g = np.argsort(ig, kind="stable")  # people that count first
        gt, areas, ig = gt[order_g], np.asarray(areas)[order_g], ig[order_g]
        order = np.argsort(-np.asarray(scores), kind="stable")[: self.max_det]
        pred, scores = np.asarray(pred)[order], np.asarray(scores)[order]
        sims = np.array([[oks(p, g, a) for g, a in zip(gt, areas, strict=True)] for p in pred]) \
            .reshape(len(pred), len(gt))
        out = np.zeros((len(self.THRESHOLDS), len(pred)), np.int8)
        for t, thr in enumerate(self.THRESHOLDS):
            taken = np.zeros(len(gt), bool)
            for d in range(len(pred)):
                best, best_sim = -1, min(thr, 1 - 1e-10)
                for g in range(len(gt)):
                    if taken[g]:
                        continue
                    if best > -1 and not ig[best] and ig[g]:
                        break  # the rest are ignored people: a real match is better
                    if sims[d, g] < best_sim:
                        continue
                    best, best_sim = g, sims[d, g]
                if best >= 0:
                    taken[best] = True
                    out[t, d] = -1 if ig[best] else 1
        self.scores.append(scores.astype(np.float64))
        self.matched.append(out)
        self.positives += int((~ig).sum())

    def compute(self) -> dict[str, float]:
        if not self.scores or not self.positives:
            return {"ap": 0.0, "ap50": 0.0, "ap75": 0.0}
        scores = np.concatenate(self.scores)
        matched = np.concatenate(self.matched, 1)
        order = np.argsort(-scores, kind="mergesort")
        matched = matched[:, order]
        recall_points = np.linspace(0, 1, 101)
        aps = []
        for row in matched:
            row = row[row >= 0]
            tp = np.cumsum(row == 1)
            fp = np.cumsum(row == 0)
            recall = tp / self.positives
            precision = tp / np.maximum(tp + fp, 1)
            if len(precision):  # the best precision at this recall or any higher one
                precision = np.maximum.accumulate(precision[::-1])[::-1]
            idx = np.searchsorted(recall, recall_points, side="left")
            sampled = np.array([precision[i] if i < len(precision) else 0.0 for i in idx])
            aps.append(sampled.mean())
        return {"ap": float(np.mean(aps)), "ap50": float(aps[0]), "ap75": float(aps[5])}
