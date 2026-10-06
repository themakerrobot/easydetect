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
#: keypoint model sizes, on the backbones of dfine-s, dfine-m and dfine-l
SIZES = ("s", "m", "l")
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

#: the ONNX metadata key a keypoint model keeps its KeypointSpec under
ONNX_SPEC_KEY = "easydetect.keypoints"


class KeypointSpec:
    """What a keypoint model places: the names of its keypoints, which swap on
    a horizontal flip, the lines drawn between them, how far off each may be
    for OKS, and which detector classes get them (``None``: boxes named
    "person", or every box when there is no such class).

    COCO's 17 body keypoints are ``COCO``; a model trained on a dataset of its
    own carries its spec inside its ``.onnx``."""

    def __init__(self, names, flip=None, skeleton=(), sigmas=None, classes=None) -> None:
        self.names = tuple(str(n) for n in names)
        k = len(self.names)
        self.flip = tuple(int(i) for i in (flip if flip is not None else range(k)))
        if sorted(self.flip) != list(range(k)):
            raise ValueError(f"flip_idx must reorder all {k} keypoints, not {list(self.flip)}")
        self.skeleton = tuple((int(a), int(b)) for a, b in skeleton)
        if any(not (0 <= a < k and 0 <= b < k) for a, b in self.skeleton):
            raise ValueError(f"skeleton names keypoints outside 0-{k - 1}")
        # without COCO's measured falloffs every keypoint gets the same: what
        # Ultralytics uses for a keypoint set of its own
        self.sigmas = np.asarray(sigmas if sigmas is not None else np.full(k, 1.0 / k),
                                 np.float64)
        if len(self.sigmas) != k:
            raise ValueError(f"{len(self.sigmas)} OKS sigmas for {k} keypoints")
        self.classes = None if classes is None else tuple(int(c) for c in classes)

    def __len__(self) -> int:
        return len(self.names)

    def __eq__(self, other) -> bool:
        return isinstance(other, KeypointSpec) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"KeypointSpec({len(self)} keypoints: {', '.join(self.names[:4])}…)"

    @property
    def is_coco(self) -> bool:
        return self.names == KEYPOINT_NAMES

    def to_dict(self) -> dict:
        return {"names": list(self.names), "flip": list(self.flip),
                "skeleton": [list(e) for e in self.skeleton],
                "sigmas": [round(float(v), 6) for v in self.sigmas],
                "classes": None if self.classes is None else list(self.classes)}

    @classmethod
    def from_dict(cls, d: dict) -> KeypointSpec:
        return cls(d["names"], d.get("flip"), d.get("skeleton", ()), d.get("sigmas"),
                   d.get("classes"))

    @classmethod
    def from_data(cls, cfg: dict) -> KeypointSpec:
        """The spec a keypoint data.yaml describes (``load_data_yaml``'s dict):
        ``kpt_shape``, ``flip_idx``, and optionally ``kpt_names`` (a list, or
        Ultralytics' ``{class: [names]}``) and ``skeleton``. Keypoints go on
        the dataset's classes."""
        k = int(cfg["kpt_shape"][0])
        names = cfg.get("kpt_names")
        if isinstance(names, dict):  # one list per class: they share a keypoint set here
            names = next(iter(names.values()), None)
        if not names or len(names) != k:
            names = [f"kp{i}" for i in range(k)]
        if k == len(KEYPOINT_NAMES) and tuple(names) == KEYPOINT_NAMES:
            return cls(KEYPOINT_NAMES, FLIP, SKELETON, OKS_SIGMAS, sorted(cfg["names"]))
        return cls(names, cfg.get("flip_idx"), cfg.get("skeleton") or (), None,
                   sorted(cfg["names"]))


#: COCO's 17 body keypoints, on boxes named "person"
COCO = KeypointSpec(KEYPOINT_NAMES, FLIP, SKELETON, OKS_SIGMAS)


def spec_of(onnx_path: str | Path) -> KeypointSpec:
    """The keypoint set a model's ``.onnx`` carries; COCO's for one without."""
    import json

    from .predictor import onnx_metadata

    raw = onnx_metadata(onnx_path).get(ONNX_SPEC_KEY)
    return KeypointSpec.from_dict(json.loads(raw)) if raw else COCO


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


def unflip(x_logits: np.ndarray, y_logits: np.ndarray,
           flip=FLIP) -> tuple[np.ndarray, np.ndarray]:
    """Output for a mirrored crop → output in the original crop's terms: left
    and right keypoints swap (``flip``, the model's), and x bin ``b`` of the
    mirror is the original's ``(W - 1) * SPLIT - b`` (pixel centres, as
    training flips them)."""
    x = x_logits[:, list(flip)][..., ::-1]
    x = np.concatenate([x[..., 1:], x[..., -1:]], -1)  # reversed is one bin off
    return x, y_logits[:, list(flip)]


# ---------------------------------------------------------------- the runtime

class KeypointEstimator:
    """``estimator(img, xyxy) -> (keypoints (N, 17, 2) picture pixels, conf (N, 17))``."""

    def __init__(self, model: str | Path, device: str = "CPU", backend: str | None = None,
                 batch: int = 16, flip: bool = True, subpixel: bool = True) -> None:
        """``flip`` also reads each crop mirrored and averages the two: +2.5 OKS
        AP on COCO val2017 for about 65% more time a person; ``subpixel``
        decodes finer than the network's half-pixel bins."""
        from .predictor import installed

        self.backend = backend or ("openvino" if installed("openvino") else "onnxruntime")
        self.batch = batch
        self.flip, self.subpixel = flip, subpixel
        #: the keypoint set this model places (its names, flip pairs, skeleton)
        self.spec = spec_of(model)
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
            return np.zeros((0, len(self.spec), 2), np.float32), \
                np.zeros((0, len(self.spec)), np.float32)
        crops, maps = self.preprocess(img, xyxy)
        xs, ys = [], []
        step = max(self.batch // 2, 1) if self.flip else self.batch
        for i in range(0, len(crops), step):
            part = crops[i:i + step]
            if self.flip:  # each crop and its mirror in one call: one call's overhead
                x, y = self._infer(np.ascontiguousarray(np.concatenate([part, part[..., ::-1]])))
                fx, fy = unflip(x[len(part):], y[len(part):], self.spec.flip)
                x, y = (x[:len(part)] + fx) / 2, (y[:len(part)] + fy) / 2
            else:
                x, y = self._infer(np.ascontiguousarray(part))
            xs.append(x)
            ys.append(y)
        xy, conf = decode(np.concatenate(xs), np.concatenate(ys), subpixel=self.subpixel)
        back = [apply(invert(m), p) for m, p in zip(maps, xy, strict=True)]
        return np.stack(back).astype(np.float32), conf


def default_estimator(device: str = "CPU", backend: str | None = None,
                      flip: bool = True, size: str = "s") -> KeypointEstimator:
    """The mirror's keypoint model of ``size`` (one of ``SIZES``), downloaded
    once into the cache."""
    from .downloads import download_pose

    return KeypointEstimator(download_pose(size), device=device, backend=backend, flip=flip)


def person_rows(names: dict[int, str], cls: np.ndarray) -> np.ndarray:
    """Which boxes get keypoints: those named "person" — or every box, when the
    model has no such class (a model trained on one kind of person, "worker")."""
    person = [k for k, v in names.items() if str(v).lower() == "person"]
    if not person:
        return np.ones(len(cls), bool)
    return np.isin(cls.astype(int), person)


def keypoint_rows(spec: KeypointSpec, names: dict[int, str], cls: np.ndarray) -> np.ndarray:
    """Which boxes get keypoints: the classes the model was trained for, or, for
    COCO's model, the people (``person_rows``)."""
    if spec.classes is None:
        return person_rows(names, cls)
    return np.isin(cls.astype(int), spec.classes)


# ---------------------------------------------------------------- evaluation

def oks(pred: np.ndarray, gt: np.ndarray, area: float, sigmas=OKS_SIGMAS) -> float:
    """Object keypoint similarity of one prediction ``(K, 2)`` to one labelled
    person ``(K, 3)`` (x, y, visibility) of segment ``area``, as COCO defines it."""
    vis = gt[:, 2] > 0
    if not vis.any():
        return 0.0
    d2 = ((pred[:, :2] - gt[:, :2]) ** 2).sum(1)
    e = d2 / ((2 * np.asarray(sigmas)) ** 2) / (area + np.spacing(1)) / 2
    return float(np.exp(-e[vis]).mean())


class KeypointAP:
    """COCO-style keypoint AP (OKS 0.50:0.95, 101-point precision, up to 20 people
    a picture, people without labelled keypoints ignored). Written to the
    published definition for easydetect; within a few tenths of pycocotools,
    which also gives crowd regions their own treatment."""

    THRESHOLDS = np.linspace(0.5, 0.95, 10)

    def __init__(self, max_det: int = 20, sigmas=OKS_SIGMAS) -> None:
        self.max_det = max_det
        self.sigmas = np.asarray(sigmas, np.float64)
        self.scores: list[np.ndarray] = []
        self.matched: list[np.ndarray] = []  # (thresholds, dets): 1 tp, 0 fp, -1 ignored
        self.positives = 0

    def add(self, pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, areas: np.ndarray,
            ignore: np.ndarray | None = None) -> None:
        """One picture: ``pred (D, K, 2)``, ``scores (D,)``, ``gt (G, K, 3)``,
        ``areas (G,)``; ``ignore`` marks crowds (and people with no keypoints
        labelled are ignored anyway)."""
        gt = np.asarray(gt, np.float64).reshape(-1, len(self.sigmas), 3)
        ig = (gt[:, :, 2] > 0).sum(1) == 0
        if ignore is not None:
            ig |= np.asarray(ignore, bool)
        order_g = np.argsort(ig, kind="stable")  # people that count first
        gt, areas, ig = gt[order_g], np.asarray(areas)[order_g], ig[order_g]
        order = np.argsort(-np.asarray(scores), kind="stable")[: self.max_det]
        pred, scores = np.asarray(pred)[order], np.asarray(scores)[order]
        sims = np.array([[oks(p, g, a, self.sigmas) for g, a in zip(gt, areas, strict=True)]
                         for p in pred]) \
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
