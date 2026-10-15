# Apache-2.0
"""Inference for an exported D-FINE model, on OpenVINO or ONNX Runtime.

OpenVINO runs an IR (``.xml``) or an ``.onnx`` on a CPU, an Intel GPU or an
Intel NPU. ONNX Runtime runs the ``.onnx`` only, and is the lighter install
(67 MB against 180 MB) for a CPU-only box such as a Raspberry Pi. Both share
the same preprocessing and decoding, so the boxes come out the same.

The exported graph is ``images -> (boxes cxcywh 0..1, scores)``. The scores it
emits are **already sigmoid'd** — squashing them a second time silently turns a
0.95 detection into 0.72 and breaks every conf threshold, so the decoder only
applies a sigmoid when the tensor is clearly still logits.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .utils.ops import cxcywh2xyxy_np


def preprocess_image(img: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR HWC uint8 -> NCHW float32 RGB 0..1, plain-resized to imgsz.

    One pass, straight into a contiguous NCHW buffer: the colour swap, the
    transpose and the scaling as views and one multiply, instead of an RGB
    copy, a float copy and a transposed view the runtime then copies again
    (1080p: 6.9 -> 1.7 ms, the same numbers to 6e-8)."""
    import cv2

    resized = cv2.resize(img, (imgsz, imgsz))
    out = np.empty((1, 3, imgsz, imgsz), np.float32)
    np.multiply(resized[:, :, ::-1].transpose(2, 0, 1), np.float32(1 / 255), out=out[0],
                casting="unsafe")
    return out


def ov_core():
    """An OpenVINO Core that keeps compiled models in easydetect's cache: a
    model's second load in a new process compiles from there, about three
    times faster (0.8-1.2 s -> 0.3-0.4 s). ``$EASYDETECT_OV_CACHE=0`` turns it off."""
    import os

    import openvino as ov

    core = ov.Core()
    if os.environ.get("EASYDETECT_OV_CACHE", "1") != "0":
        try:
            from .downloads import cache_dir

            folder = cache_dir() / "openvino"
            folder.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(folder)})
        except OSError:
            pass  # a read-only home: compile every time, as before
    return core


def _looks_like_logits(scores: np.ndarray) -> bool:
    """True when the head still emits raw logits (anything outside 0..1)."""
    return bool(scores.size) and (scores.min() < 0.0 or scores.max() > 1.0)


#: The ONNX metadata key easydetect keeps the class names under.
ONNX_NAMES_KEY = "easydetect.names"


def _varint(f) -> int | None:
    shift = value = 0
    while True:
        byte = f.read(1)
        if not byte:
            return None
        value |= (byte[0] & 0x7F) << shift
        if byte[0] < 0x80:
            return value
        shift += 7


def onnx_metadata(path: Path) -> dict[str, str]:
    """The ``metadata_props`` of an ONNX file, read without the ``onnx`` package.

    ModelProto keeps them as field 14, each a (key = 1, value = 2) pair. Only
    the top level of the file is walked; the graph and its weights are skipped
    over with a seek, so a large model costs no more than a small one.
    """
    props: dict[str, str] = {}
    with open(path, "rb") as f:
        while (tag := _varint(f)) is not None:
            field, wire = tag >> 3, tag & 7
            if wire == 0:
                _varint(f)
            elif wire == 1:
                f.seek(8, 1)
            elif wire == 5:
                f.seek(4, 1)
            elif wire == 2:
                size = _varint(f) or 0
                if field != 14:
                    f.seek(size, 1)
                    continue
                entry, key, value, i = f.read(size), "", "", 0
                while i < len(entry):
                    sub_tag, n = entry[i], 0
                    i += 1
                    shift = 0
                    while True:                       # the length, a varint
                        n |= (entry[i] & 0x7F) << shift
                        i += 1
                        if entry[i - 1] < 0x80:
                            break
                        shift += 7
                    text = entry[i:i + n].decode("utf-8", "replace")
                    i += n
                    if sub_tag >> 3 == 1:
                        key = text
                    elif sub_tag >> 3 == 2:
                        value = text
                props[key] = value
            else:                                     # not an ONNX file after all
                break
    return props


def read_names(model_path: Path) -> dict[int, str]:
    """Class names: ``<stem>.names.json`` beside the model, then the names an
    ``.onnx`` carries inside it, then ``labels.txt`` in its folder.

    The names inside the file come before a ``labels.txt`` because they belong
    to that file: a lone ``best.onnx`` downloaded into a folder that holds
    another model's ``labels.txt`` still names its own classes.
    """
    model_path = Path(model_path)
    sidecar = model_path.with_suffix(".names.json")
    if sidecar.exists():
        table = json.loads(sidecar.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in table.items()}
    if model_path.suffix.lower() == ".onnx" and model_path.is_file():
        try:
            inside = onnx_metadata(model_path).get(ONNX_NAMES_KEY)
            if inside:
                return {int(k): str(v) for k, v in json.loads(inside).items()}
        except (OSError, ValueError, IndexError):
            pass
    labels = model_path.parent / "labels.txt"
    if labels.exists():
        lines = [ln.strip() for ln in labels.read_text(encoding="utf-8").splitlines()]
        return {i: n for i, n in enumerate(lines) if n}
    return {}


#: Default for ``iou``: above this IoU, two boxes are one object.
IOU = 0.7

#: Default for ``conf``. D-FINE gives an unsure box 0.3-0.5 where YOLO would
#: give it under 0.25, so YOLO's 0.25 shows mostly mistakes here: dfine-m on
#: COCO val2017 (NMS 0.7) shows boxes 32% right at 0.25, 70% at 0.5, while
#: still finding 67% of the objects. See docs/performance.md.
CONF = 0.5


def drop_duplicates(xyxy: np.ndarray, iou: float = IOU) -> np.ndarray:
    """Indices of the boxes to keep, from boxes sorted best first.

    A DETR is trained to give each object one query, and mostly does; but
    below its confident answers a second query can land on the same object,
    often under a neighbouring class — one vehicle as both ``truck`` 0.83 and
    ``car`` 0.57 (IoU 0.93). Measured on dfine-s at ``conf=0.25``, every pair
    overlapping by more than 0.7 was one object twice, and several pairs
    were two classes, so this ignores the class: a box that covers a better
    one by more than ``iou`` goes. Two separate objects that overlap this
    much in a picture are rare; pass ``iou=None`` to keep every box.
    """
    x1, y1, x2, y2 = xyxy.T
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    alive = np.ones(len(xyxy), bool)
    for i in range(len(xyxy)):
        if not alive[i]:
            continue
        rest = np.nonzero(alive[i + 1:])[0] + i + 1
        if not len(rest):
            break
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        overlap = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        alive[rest[overlap > iou]] = False
    return np.nonzero(alive)[0]


#: How much less sure an enclosing box may be than the box inside it and still
#: count as the whole object rather than a loose box around several.
CONTAIN_MARGIN = 0.1


def drop_contained(xyxy: np.ndarray, scores: np.ndarray, cls: np.ndarray,
                   contain: float) -> np.ndarray:
    """Indices of the boxes to keep when a box lies inside another of its class.

    IoU cannot see a part inside a whole: a chair half hidden behind a person
    can come back as the whole chair (0.64) plus its two visible pieces (0.67,
    0.61), and each piece overlaps the whole by an IoU of only its share of the
    area. Measured as the overlap over the *smaller* box instead, a piece sits
    near 1.0, so a pair of one class sharing at least ``contain`` of the
    smaller box is one object, and one box goes:

    * the inner one, when the enclosing box is about as sure (within
      ``CONTAIN_MARGIN``) — the whole object and a piece of it;
    * the enclosing one, when it is much less sure than the box inside — a
      loose box around a group, which must not erase the members.

    NMS's "keep the best" would not do here: the best box is a piece, and once
    it has removed the whole, the other piece no longer overlaps anything and
    stays — two half chairs. Same class only: a cup on a table is not part of
    the table. The cost is a real object inside another of its class, a child
    held by an adult, so this is off unless asked for.
    """
    x1, y1, x2, y2 = xyxy.T
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    gone = np.zeros(len(xyxy), bool)
    for i in range(len(xyxy)):
        for j in range(i + 1, len(xyxy)):
            if cls[i] != cls[j]:
                continue
            iw = max(0.0, min(x2[i], x2[j]) - max(x1[i], x1[j]))
            ih = max(0.0, min(y2[i], y2[j]) - max(y1[i], y1[j]))
            if iw * ih < contain * max(min(area[i], area[j]), 1e-9):
                continue
            big, small = (i, j) if area[i] >= area[j] else (j, i)
            if scores[big] >= scores[small] - CONTAIN_MARGIN:
                gone[small] = True
            else:
                gone[big] = True
    return np.nonzero(~gone)[0]


def _split_outputs(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(boxes, scores)`` for the one image in a batch, whatever order they come in."""
    boxes = next((a for a in arrays if a.shape[-1] == 4), None)
    scores = next((a for a in arrays if a is not boxes), None)
    if boxes is None or scores is None:
        shapes = [a.shape for a in arrays]
        raise ValueError(f"unexpected model outputs {shapes}; expected boxes + scores")
    return boxes[0], scores[0]


def installed(backend: str) -> bool:
    import importlib.util

    module = {"openvino": "openvino", "onnxruntime": "onnxruntime"}[backend]
    return importlib.util.find_spec(module) is not None


BACKENDS = ("openvino", "onnxruntime")


def pick_backend(requested: str | None = None, model_path: str | Path | None = None,
                 device: str = "AUTO") -> str:
    """Which runtime to use: the one asked for, else OpenVINO when it is there.

    An ``.xml`` is OpenVINO's own format, and an Intel GPU or NPU is only
    reachable through OpenVINO; anything else falls back to ONNX Runtime
    when OpenVINO is not installed. ``$EASYDETECT_BACKEND`` sets the default.
    """
    import os

    requested = (requested or os.environ.get("EASYDETECT_BACKEND") or "").lower() or None
    if requested and requested not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {requested!r}")
    suffix = Path(model_path).suffix.lower() if model_path else ""
    intel_only = str(device).upper().split(".")[0] in ("GPU", "NPU")
    if requested == "onnxruntime" and suffix == ".xml":
        raise ValueError("ONNX Runtime cannot read an OpenVINO .xml; pass the .onnx exported "
                         "beside it, or use backend='openvino'")
    if requested == "onnxruntime" and intel_only:
        raise ValueError(f"device {device!r} is an OpenVINO device; ONNX Runtime runs on the CPU")
    backend = requested or ("openvino" if installed("openvino") or suffix == ".xml" or intel_only
                            else "onnxruntime" if installed("onnxruntime") else "openvino")
    if not installed(backend):
        why = (" (an .xml needs it)" if suffix == ".xml" else
               f" (device {device} needs it)" if intel_only else "")
        raise ImportError(f"{backend} is not installed{why}: pip install {backend} "
                          f"(pip install easydetect brings both runtimes)")
    return backend


class Predictor:
    """What every runtime shares: resize in, boxes out. Subclasses run the graph."""

    model_path: Path
    device: str
    precision: str | None
    backend: str
    names: dict[int, str]
    imgsz: int

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(boxes (Q,4) cxcywh 0..1, scores (Q,K))`` for one image — per runtime."""
        raise NotImplementedError

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 -> NCHW float32 RGB 0..1, plain-resized to imgsz."""
        return preprocess_image(img, self.imgsz)

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        orig_shape: tuple[int, int],
        conf: float = CONF,
        max_det: int = 300,
        classes: list[int] | None = None,
        iou: float | None = IOU,
        contain: float | None = None,
    ) -> np.ndarray:
        """-> ``(N, 6)`` array of ``x1 y1 x2 y2 conf cls`` in pixels.

        ``iou`` drops a box that covers a higher-scoring one by more than that
        IoU, whatever the two classes (``None`` keeps them all): see
        :func:`drop_duplicates`. ``contain`` (off by default) also drops a box
        lying mostly inside another of its class: see
        :func:`drop_contained`.
        """
        if _looks_like_logits(scores):
            scores = 1.0 / (1.0 + np.exp(-scores))
        if classes is not None:
            # each query's best class among those asked for, not its best class
            # overall: a query sure of "person" at 0.6 but of "rider" at 0.7
            # is still a person when only people are wanted
            wanted = np.asarray(classes, np.int64)
            wanted = wanted[(wanted >= 0) & (wanted < scores.shape[-1])]
            among = scores[:, wanted] if len(wanted) else scores[:, :0]
            cls = wanted[among.argmax(-1)] if len(wanted) else np.zeros(len(scores), np.int64)
            best = among.max(-1) if len(wanted) else np.zeros(len(scores), scores.dtype)
        else:
            cls = scores.argmax(-1)
            best = scores.max(-1)
        keep = best >= conf
        boxes, best, cls = boxes[keep], best[keep], cls[keep]
        if len(best) > max_det:
            top = np.argsort(-best)[:max_det]
            boxes, best, cls = boxes[top], best[top], cls[top]
        order = np.argsort(-best)
        boxes, best, cls = boxes[order], best[order], cls[order]

        h, w = orig_shape
        xyxy = cxcywh2xyxy_np(boxes.astype(np.float32)) * np.array([w, h, w, h], np.float32)
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)
        if iou is not None and len(xyxy) > 1:
            kept = drop_duplicates(xyxy, iou)
            xyxy, best, cls = xyxy[kept], best[kept], cls[kept]
        if contain is not None and len(xyxy) > 1:
            kept = drop_contained(xyxy, best, cls, contain)
            xyxy, best, cls = xyxy[kept], best[kept], cls[kept]
        return np.concatenate(
            [xyxy, best.astype(np.float32)[:, None], cls.astype(np.float32)[:, None]], axis=1
        )

    def __call__(
        self,
        img: np.ndarray,
        conf: float = CONF,
        max_det: int = 300,
        classes: list[int] | None = None,
        iou: float | None = IOU,
        contain: float | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run one image; returns ``(detections (N,6), speed dict in ms)``."""
        import time

        t0 = time.perf_counter()
        tensor = self.preprocess(img)
        t1 = time.perf_counter()
        boxes, scores = self.infer(tensor)
        t2 = time.perf_counter()
        det = self.postprocess(boxes, scores, img.shape[:2], conf, max_det, classes, iou,
                               contain)
        t3 = time.perf_counter()
        speed = {
            "preprocess": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "postprocess": (t3 - t2) * 1e3,
        }
        return det, speed


class OVPredictor(Predictor):
    """Compiles an IR once, then runs images through it."""

    backend = "openvino"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "AUTO",
        imgsz: int | None = None,
        names: dict[int, str] | None = None,
        precision: str | None = None,
    ) -> None:
        """``precision`` overrides what OpenVINO runs the graph in.

        Left alone, OpenVINO picks: on a CPU that supports it that means
        bfloat16, which is much faster but shifts scores slightly. Pass
        ``"f32"`` when you want output that matches PyTorch exactly.
        """
        self.model_path = Path(model_path)
        self.device = device
        self.precision = precision
        core = ov_core()
        model = core.read_model(str(self.model_path))
        config = {"INFERENCE_PRECISION_HINT": precision} if precision else {}
        self.compiled = core.compile_model(model, device, config)
        self.input = self.compiled.input(0)
        # CompiledModel.__call__ reuses one InferRequest, so a second thread
        # calling it gets "Infer Request is busy". One request per thread.
        self._local = threading.local()
        self.names = names if names else read_names(self.model_path)
        self.imgsz = imgsz or self._input_size()

    def _input_size(self) -> int:
        shape = self.input.partial_shape
        try:
            h, w = shape[2], shape[3]
            if h.is_static and w.is_static:
                return int(max(h.get_length(), w.get_length()))
        except Exception:  # dynamic or unusual layout
            pass
        return 640

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(boxes (Q,4) cxcywh 0..1, scores (Q,K))`` for one image."""
        request = getattr(self._local, "request", None)
        if request is None:
            request = self._local.request = self.compiled.create_infer_request()
        request.infer({0: tensor})
        # copy: the tensor data is the request's own buffer, overwritten next call
        arrays = [
            np.array(request.get_output_tensor(i).data)
            for i in range(len(self.compiled.outputs))
        ]
        return _split_outputs(arrays)


def ort_providers(device: str, available: list[str]) -> list[str]:
    """ONNX Runtime's execution providers for ``device``, the CPU always last.

    ``"AUTO"`` takes a GPU when this onnxruntime build has one — CUDA
    (onnxruntime-gpu) or DirectML (onnxruntime-directml, any GPU on Windows);
    ``"CUDA"``/``"GPU0"`` asks for CUDA, ``"TENSORRT"`` for TensorRT (its
    first run builds an engine, minutes); ``"CPU"`` stays on the CPU."""
    want = str(device).upper()
    order = {"AUTO": ["CUDAExecutionProvider", "DmlExecutionProvider"],
             "CUDA": ["CUDAExecutionProvider"], "GPU0": ["CUDAExecutionProvider"],
             "GPU": ["CUDAExecutionProvider", "DmlExecutionProvider"],
             "TENSORRT": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
             "DML": ["DmlExecutionProvider"]}.get(want, [])
    return [p for p in order if p in available] + ["CPUExecutionProvider"]


class ORTPredictor(Predictor):
    """An ``.onnx`` on ONNX Runtime: the light CPU runtime, no OpenVINO needed."""

    backend = "onnxruntime"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "CPU",
        imgsz: int | None = None,
        names: dict[int, str] | None = None,
        precision: str | None = None,
        threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(f"ONNX Runtime runs .onnx files, not {self.model_path.name}")
        self.device = device
        self.precision = precision
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        providers = ort_providers(device, ort.get_available_providers())
        # InferenceSession.run may be called from several threads at once
        self.session = ort.InferenceSession(str(self.model_path), options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.names = names if names else read_names(self.model_path)
        self.imgsz = imgsz or self._input_size()

    def _input_size(self) -> int:
        shape = self.session.get_inputs()[0].shape
        h, w = (shape[2], shape[3]) if len(shape) == 4 else (None, None)
        if isinstance(h, int) and isinstance(w, int):
            return max(h, w)
        return 640

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(boxes (Q,4) cxcywh 0..1, scores (Q,K))`` for one image."""
        return _split_outputs(self.session.run(None, {self.input_name: tensor}))
