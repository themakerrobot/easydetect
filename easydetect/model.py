# Apache-2.0
"""The one object users touch: ``Detector``.

    from easydetect import Detector

    model = Detector("dfine-s")            # mirrored weights, downloaded on demand
    model = Detector("best.pt")               # or your own checkpoint
    results = model("bus.jpg", conf=0.5)    # list[Results]
    model.train(data="data.yaml", epochs=100)
    model.val(data="data.yaml").box.map50
    model.export(format="openvino", half=True)

Inference runs on OpenVINO, training on PyTorch. A ``.pt`` checkpoint is
exported to an IR (cached) the first time you predict with it, so the two
halves of the workflow meet without you having to think about it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from . import downloads
from .errors import ModelNotFoundError
from .metrics import DetMetrics
from .results import Results

#: What a Detector can add to its boxes: nothing, a mask each, or a person's keypoints.
TASKS = ("detect", "segment", "pose")

_TORCH_HINT = (
    "this needs the training extra: pip install 'easydetect[train]' "
    "(torch, torchvision, scipy, onnx)"
)


def increment_path(path: str | Path) -> Path:
    """``runs/detect/predict`` -> ``runs/detect/predict2`` when it already exists."""
    path = Path(path)
    if not path.exists():
        return path
    i = 2
    while (candidate := path.with_name(f"{path.name}{i}")).exists():
        i += 1
    return candidate


def _torch_device(device: Any) -> str | None:
    """``0`` -> ``"cuda:0"``; ``"cpu"``/``"cuda"``/``None`` pass through."""
    if device is None or device == "":
        return None
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return f"cuda:{int(device)}"
    return str(device)


class Detector:
    """Train, validate, export, track and predict with one object."""

    def __init__(
        self,
        model: str | Path = "dfine-s",
        device: str = "AUTO",
        verbose: bool = True,
        precision: str | None = None,
        pretrained: bool = True,
        backend: str | None = None,
        task: str = "detect",
    ) -> None:
        """``pretrained=False`` starts training from an ImageNet backbone
        instead of the mirror's COCO weights — for domains COCO says nothing
        about, or for a clean baseline.

        ``backend`` picks the runtime: ``"openvino"`` (CPU, Intel GPU, NPU;
        reads ``.xml`` and ``.onnx``) or ``"onnxruntime"`` (CPU; ``.onnx``
        only, the lighter install). Left out, OpenVINO is used when it is
        installed and ONNX Runtime otherwise.

        ``task="segment"`` also outlines what is inside every box
        (``r.masks``), with MobileSAM prompted by the box — any class, no mask
        labels needed; it downloads once and adds about 150 ms a picture plus
        25 ms a box on a 4-core CPU.

        ``task="pose"`` places 17 body keypoints on every person
        (``r.keypoints``) with easydetect's own keypoint model, trained on COCO
        — on boxes named "person", or on every box when the model has no such
        class."""
        self.model_name = str(model)
        self.device = device
        self.precision = precision
        self.backend = backend
        self.pretrained = pretrained
        self.verbose = verbose
        if task not in TASKS:
            raise ValueError(f"task must be one of {', '.join(TASKS)}")
        self.task = task
        self.segmenter = None  # BoxSegmenter for task="segment" (lazy)
        self.pose = None  # KeypointEstimator for task="pose" (lazy)
        #: task="pose" reads each person mirrored too and averages: +2.5 OKS AP,
        #: about 65% more keypoint time. False before the first prediction: faster.
        self.pose_flip = True
        #: the keypoint model for task="pose": "s" (HGNetv2-B0, the default), "m"
        #: (B2), "l" (B4) — larger and slower in turn — or the path of a pose
        #: .onnx of your own. Set it before the first prediction.
        self.pose_model = "s"
        self.names: dict[int, str] = {}
        self.net = None  # torch DFINENet (lazy)
        self.ckpt: dict | None = None
        self.variant: str | None = None
        self.ckpt_path: Path | None = None
        self.predictor = None  # OVPredictor or ORTPredictor (lazy)
        self.ir_path: Path | None = None
        self.tracker = None

        path = Path(self.model_name)
        suffix = path.suffix.lower()
        if suffix == ".pt":
            if not path.exists():
                raise FileNotFoundError(f"checkpoint not found: {path}")
            self._load_checkpoint(path)
        elif suffix in (".xml", ".onnx"):
            if not path.exists():
                raise FileNotFoundError(f"model not found: {path}")
            self.ir_path = path
            self.variant = downloads.normalize_name(path.stem).rsplit("-", 1)[-1]
        elif downloads.is_model_name(self.model_name):
            self.model_name = downloads.normalize_name(self.model_name)
            self.variant = self.model_name.rsplit("-", 1)[-1]
        else:
            raise ModelNotFoundError(downloads._unknown_name_message(self.model_name))

    # ------------------------------------------------------------------ predict

    def __call__(self, source: Any = None, **kwargs: Any) -> Any:
        return self.predict(source, **kwargs)

    def predict(
        self,
        source: Any = None,
        conf: float = 0.5,
        imgsz: int | None = None,
        device: str | None = None,
        max_det: int = 300,
        classes: list[int] | None = None,
        iou: float | None = 0.7,
        contain: float | None = None,
        stream: bool = False,
        save: bool = False,
        show: bool = False,
        project: str = "runs",
        name: str = "predict",
        vid_stride: int = 1,
        verbose: bool | None = None,
        line_width: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run detection on any supported source.

        Returns a ``list[Results]``, or a generator when ``stream=True`` (the
        only sane option for a long video or a live camera). ``conf=0.5``
        is where D-FINE's boxes are mostly right (70% on COCO, against 32% at
        YOLO's usual 0.25). ``iou`` drops a box covering a higher-scoring one
        by more than that IoU — the same object found twice, sometimes under
        two classes; ``None`` keeps all. ``contain=0.8`` also merges a box
        with 80% of its area inside another of its class — the visible pieces
        of a half hidden object beside the whole of it — into one box; off by
        default, as on COCO it removes more real objects than it fixes.
        """
        if source is None:
            raise ValueError("predict() needs a source (image, folder, video, url, camera index)")
        if kwargs.pop("track", False):
            raise TypeError("use model.track(...) instead of predict(track=True)")
        for unexpected in kwargs:
            raise TypeError(f"unexpected predict() argument: {unexpected!r}")

        gen = self._run(
            source,
            conf=conf,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            classes=classes,
            iou=iou,
            contain=contain,
            save=save,
            show=show,
            project=project,
            name=name,
            vid_stride=vid_stride,
            verbose=self.verbose if verbose is None else verbose,
            line_width=line_width,
            tracker=None,
        )
        return _Stream(gen, "predict") if stream else list(gen)

    def track(
        self,
        source: Any = None,
        conf: float = 0.5,
        iou: float | None = 0.7,
        contain: float | None = None,
        match_iou: float = 0.3,
        max_age: int = 30,
        persist: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Predict, and keep an id on each box across frames (``boxes.id``).

        ``iou`` and ``contain`` filter duplicates, as in :meth:`predict`. ``match_iou``
        is how much a box must overlap a track in the previous frame to carry
        its id on; ``max_age`` is how many frames a track waits for its object.
        """
        from .tracker import IoUTracker

        if self.tracker is None or not persist:
            self.tracker = IoUTracker(iou=match_iou, max_age=max_age)
        kwargs.setdefault("name", "track")
        gen = self._run(source, conf=conf, iou=iou, contain=contain, tracker=self.tracker,
                        **self._track_kwargs(kwargs))
        return _Stream(gen, "track") if stream else list(gen)

    def _ensure_segmenter(self, device: str | None = None):
        if self.segmenter is None:
            from .segment import default_segmenter

            # the same runtime and device the detector runs on
            backend = self.predictor.backend if self.predictor is not None else self.backend
            self.segmenter = default_segmenter(device=device or self.device, backend=backend)
        return self.segmenter

    def _ensure_pose(self, device: str | None = None):
        if self.pose is None:
            from .pose import default_estimator

            backend = self.predictor.backend if self.predictor is not None else self.backend
            from .pose import SIZES

            model = str(self.pose_model)
            if model in SIZES:
                self.pose = default_estimator(device=device or self.device, backend=backend,
                                              flip=self.pose_flip, size=model)
            elif model.endswith(".onnx") and Path(model).is_file():
                from .pose import KeypointEstimator

                self.pose = KeypointEstimator(model, device=device or self.device,
                                              backend=backend, flip=self.pose_flip)
            else:
                raise ValueError(f"pose_model must be one of {', '.join(SIZES)} or a pose "
                                 f".onnx file, not {self.pose_model!r}")
        return self.pose

    def _keypoints(self, img, det, names, device):
        """``(N, 17, 3)`` x, y, confidence for each box; zeros for boxes that are
        not people."""
        from .pose import KEYPOINT_NAMES, person_rows

        out = np.zeros((len(det), len(KEYPOINT_NAMES), 3), np.float32)
        rows = person_rows(names, det[:, -1]) if len(det) else np.zeros(0, bool)
        if rows.any():
            xy, conf = self._ensure_pose(device)(img, det[rows, :4])
            out[rows] = np.concatenate([xy, conf[..., None]], -1)
        return out

    def _track_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "imgsz", "device", "max_det", "classes", "save", "show",
            "project", "name", "vid_stride", "verbose", "line_width",
        }
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise TypeError(f"unexpected track() argument(s): {sorted(unexpected)}")
        kwargs.setdefault("verbose", self.verbose)
        return kwargs

    def _run(
        self,
        source: Any,
        conf: float = 0.5,
        imgsz: int | None = None,
        device: str | None = None,
        max_det: int = 300,
        classes: list[int] | None = None,
        iou: float | None = 0.7,
        contain: float | None = None,
        save: bool = False,
        show: bool = False,
        project: str = "runs",
        name: str = "predict",
        vid_stride: int = 1,
        verbose: bool = True,
        line_width: int | None = None,
        tracker: Any = None,
    ) -> Iterator[Results]:
        from .sources import SourceLoader

        predictor = self._ensure_predictor(device=device, imgsz=imgsz)
        loader = SourceLoader(source, vid_stride=vid_stride)
        save_dir = increment_path(Path(project) / "detect" / name) if save else None
        writer = _OutputWriter(save_dir) if save_dir else None
        if save_dir and verbose:
            print(f"results will be saved to {save_dir}")

        try:
            for frame in loader:
                det, speed = predictor(frame.img, conf=conf, max_det=max_det, classes=classes,
                                       iou=iou, contain=contain)
                if tracker is not None:
                    det = tracker.update(det)
                masks = None
                if self.task == "segment":
                    import time

                    started = time.perf_counter()
                    masks, _ = self._ensure_segmenter(device)(frame.img, det[:, :4])
                    speed["segment"] = (time.perf_counter() - started) * 1e3
                names = self.names or predictor.names
                keypoints = None
                if self.task == "pose":
                    import time

                    started = time.perf_counter()
                    keypoints = self._keypoints(frame.img, det, names, device)
                    speed["pose"] = (time.perf_counter() - started) * 1e3
                result = Results(
                    frame.img,
                    path=frame.path,
                    names=names,
                    boxes=det,
                    speed=speed,
                    masks=masks,
                    keypoints=keypoints,
                )
                if verbose:
                    total = sum(speed.values())
                    print(
                        f"{frame.prefix()}{predictor.imgsz}x{predictor.imgsz} "
                        f"{result.verbose()}{total:.1f}ms"
                    )
                if writer is not None:
                    writer.write(result, frame, line_width=line_width)
                if show and not _display(result, frame, line_width):
                    break  # the viewer asked to stop (q / Esc / window closed)
                yield result
        finally:
            if writer is not None:
                writer.close()
            if show:
                _close_windows()

    def _ensure_predictor(self, device: str | None = None, imgsz: int | None = None):
        """Load the model on its runtime (downloading or exporting first, if needed)."""
        from .predictor import ORTPredictor, OVPredictor, installed, pick_backend

        device = device or self.device
        if (self.ir_path is not None and self.ir_path.suffix.lower() == ".xml"
                and self.backend != "openvino" and not installed("openvino")
                and self.ir_path.with_suffix(".onnx").exists()):
            # every export writes the .onnx beside the .xml: without OpenVINO, run that
            self.ir_path = self.ir_path.with_suffix(".onnx")
        backend = pick_backend(self.backend, self.ir_path, device)
        if (
            self.predictor is not None
            and self.predictor.backend == backend
            and self.predictor.device == device
            and self.predictor.precision == self.precision
            and (imgsz is None or self.predictor.imgsz == imgsz)
        ):
            return self.predictor

        wanted = ".xml" if backend == "openvino" else ".onnx"
        if self.ir_path is not None and self.ir_path.suffix.lower() not in (wanted, ".onnx"):
            self.ir_path = None             # an .xml met a runtime that cannot read it
        if self.ir_path is None:
            if self.net is not None:
                fmt = "openvino" if backend == "openvino" else "onnx"
                self.ir_path = self.export(format=fmt, out_dir=self._cache_export_dir(),
                                           verbose=self.verbose)
            elif backend == "openvino":
                self.ir_path = downloads.download_ir(self.model_name)
            else:
                self.ir_path = downloads.download_onnx(self.model_name)
        if backend == "openvino":
            self.predictor = OVPredictor(self.ir_path, device=device, imgsz=imgsz,
                                         names=self.names or None, precision=self.precision)
        else:
            self.predictor = ORTPredictor(self.ir_path, device=device, imgsz=imgsz,
                                          names=self.names or None, precision=self.precision)
        if not self.names:
            self.names = self.predictor.names
        return self.predictor

    def _cache_export_dir(self) -> Path:
        stem = self.ckpt_path.stem if self.ckpt_path else (self.variant or "s")
        out = downloads.cache_dir() / "exported" / stem
        out.mkdir(parents=True, exist_ok=True)
        return out

    # --------------------------------------------------------------- checkpoint

    def _load_checkpoint(self, path: Path) -> None:
        torch = _import_torch()
        from .nn import DFINENet

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.ckpt = ckpt
        self.ckpt_path = Path(path)
        self.variant = ckpt["variant"]
        self.names = {int(k): str(v) for k, v in ckpt.get("names", {}).items()}
        self.net = DFINENet(ckpt["variant"], ckpt["num_classes"], pretrained_backbone=False)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        self.ir_path = None
        self.predictor = None

    def _ensure_net(self):
        """Torch weights for train/val/export — from the checkpoint or the mirror."""
        if self.net is not None:
            return self.net
        _import_torch()
        if downloads.is_model_name(self.model_name):
            self._load_checkpoint(downloads.download_checkpoint(self.model_name))
            return self.net
        raise RuntimeError(
            f"'{self.model_name}' has no torch weights — load a .pt checkpoint "
            f"(Detector('best.pt')) or a pretrained name to do this."
        )

    # ----------------------------------------------------------------- training

    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 8,
        device: Any = None,
        workers: int = 4,
        project: str = "runs",
        name: str = "train",
        resume: bool = False,
        patience: int = 50,
        lr0: float | None = None,
        seed: int = 0,
        val: bool = True,
        amp: bool = True,
        freeze: Any = None,
        augment: bool = True,
        on_epoch_end: Any = None,
        on_progress: Any = None,
        **kwargs: Any,
    ) -> Path:
        """Train on a data.yaml dataset. Returns the path of ``best.pt``.

        ``freeze="backbone"`` trains about twice as fast on CPU and helps when
        the dataset is small. ``lr0=None`` sets the learning rate from the
        batch size, ``1e-4 × √(batch / 4)``, so a bigger batch is not a run
        that learned less; a number sets it outright. ``augment=True`` trains on zoomed-out, cropped
        and colour-jittered pictures (D-FINE's recipe) and on plain ones for
        the last tenth of the epochs; ``False`` keeps only the flip.
        ``on_epoch_end`` is called with a dict of the epoch's numbers — the
        same row that lands in ``results.csv``.
        ``on_progress`` hears about once a second inside an epoch —
        ``{"phase": "train", "epoch", "epochs", "step", "steps", "seconds"}`` —
        and once more with ``"phase": "val"`` before validation starts, so a
        long epoch is not a silent one.
        """
        _import_torch()
        from .data.dataset import load_data_yaml
        from .nn import DFINENet
        from .trainer import Trainer
        from .validator import validate_torch

        cfg = load_data_yaml(data)
        self.names = cfg["names"]
        # where the weights came from, for the run's record
        origin = None
        if self.net is not None:
            origin = ({"kind": "checkpoint", "path": str(self.ckpt_path)} if self.ckpt_path
                      else {"kind": "custom"})
        if self.net is None and self.pretrained and downloads.is_model_name(self.model_name):
            self._try_pretrained_start(cfg["nc"])
            if self.net is not None:
                origin = {"kind": "coco", "weights": self.model_name}
        if self.net is None:
            self.net = DFINENet(self.variant or "s", cfg["nc"])
            loaded = getattr(self.net.backbone, "imagenet_loaded", False)
            origin = {"kind": "imagenet" if loaded else "scratch"}
        elif self.net.num_classes != cfg["nc"]:
            self.net = self._reheaded_net(cfg["nc"])

        device = _torch_device(device)
        trainer = Trainer(
            self.net,
            data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr=lr0,
            device=device,
            workers=workers,
            project=project,
            name=name,
            resume=resume,
            patience=patience,
            seed=seed,
            amp=amp,
            freeze=freeze,
            augment=augment,
            on_epoch_end=on_epoch_end,
            on_progress=on_progress,
            origin=origin,
            **kwargs,
        )
        val_fn = None
        if val:
            def val_fn(net):  # noqa: E306 — tiny closure, reads better inline
                return validate_torch(net, data, imgsz=imgsz, batch=batch, device=device)["map"]

        best = trainer.train(val_fn=val_fn)
        self._load_checkpoint(best)
        return best

    def _try_pretrained_start(self, nc: int) -> None:
        """Warm-start from the mirror's COCO weights, quietly falling back."""
        try:
            checkpoint = downloads.download_checkpoint(self.model_name)
        except ModelNotFoundError as exc:
            if self.verbose:
                print(f"{exc}", file=sys.stderr)
            return
        self._load_checkpoint(checkpoint)
        if self.net.num_classes != nc:
            self.net = self._reheaded_net(nc)

    def _reheaded_net(self, nc: int):
        """Keep backbone/encoder/decoder weights, re-init the class heads for ``nc``."""
        from .nn import DFINENet

        fresh = DFINENet(self.variant or "s", nc, pretrained_backbone=False)
        transferred = transfer_weights(fresh, self.net.state_dict())
        if self.verbose:
            print(
                f"transferred {transferred} tensors from the "
                f"{self.net.num_classes}-class checkpoint; class heads re-initialised for {nc}"
            )
        return fresh

    def val(
        self,
        data: str,
        imgsz: int = 640,
        batch: int = 8,
        conf: float = 0.01,
        device: Any = None,
        workers: int = 2,
    ) -> DetMetrics:
        """COCO-style mAP on the ``val`` split of a data.yaml."""
        self._ensure_net()
        from .data.dataset import load_data_yaml
        from .validator import validate_torch

        cfg = load_data_yaml(data)
        raw = validate_torch(
            self.net,
            data,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            device=_torch_device(device),
            workers=workers,
        )
        metrics = DetMetrics(
            raw["map50"], raw["map"], per_class=raw.get("per_class"), names=cfg["names"]
        )
        if self.verbose:
            print(f"mAP50 {metrics.box.map50:.4f}   mAP50-95 {metrics.box.map:.4f}")
        return metrics

    def export(
        self,
        format: str = "openvino",
        imgsz: int | None = None,
        half: bool = False,
        out_dir: str | Path | None = None,
        verbose: bool | None = None,
        layers: int | None = None,
        queries: int | None = None,
        int8: bool = False,
        data: Any = None,
        calib: int = 300,
    ) -> Path:
        """Export to OpenVINO IR (default) or ONNX. Returns the file written.

        ``int8=True`` (OpenVINO only) quantizes the IR to 8 bits, calibrated
        on ``calib`` pictures from ``data`` — a data.yaml (its train split), a
        folder or a list of files — and writes ``<name>_int8.xml``.

        ``layers`` (decoder layers to run, of 3-6 by size) and ``queries`` (of
        300) export a faster, slightly less accurate model; dfine-n at 320 on
        a 4-core CPU: 19.9 ms, 13.2 ms with ``queries=100``.
        """
        format = format.lower()
        if format not in ("openvino", "onnx"):
            raise ValueError("supported formats: 'openvino', 'onnx'")
        self._ensure_net()
        from .exporter import export_onnx, export_openvino

        if out_dir is None:
            out_dir = self.ckpt_path.parent if self.ckpt_path else Path(".")
        imgsz = imgsz or (self.ckpt or {}).get("imgsz", 640)
        stem = self.ckpt_path.stem if self.ckpt_path else f"easydetect-{self.variant}"
        verbose = self.verbose if verbose is None else verbose
        if int8:
            if format != "openvino":
                raise ValueError("int8 is an OpenVINO export: format='openvino'")
            return export_openvino(
                self.net, self.names, imgsz=imgsz, out_dir=out_dir, fname=f"{stem}_int8",
                verbose=verbose, layers=layers, queries=queries,
                int8=_calibration_images(data, calib),
            )
        exporter = export_openvino if format == "openvino" else export_onnx
        return exporter(
            self.net, self.names, imgsz=imgsz, out_dir=out_dir, fname=stem,
            half=half, verbose=verbose, layers=layers, queries=queries,
        )

    # -------------------------------------------------------------------- misc

    def info(self) -> str:
        """One line about what is loaded."""
        where = self.ckpt_path or self.ir_path or self.model_name
        nc = self.net.num_classes if self.net is not None else len(self.names) or "?"
        return f"D-FINE-{(self.variant or '?').upper()} — {nc} classes — {where}"

    def __repr__(self) -> str:
        return f"Detector({self.model_name!r}, device={self.device!r})"


def _calibration_images(data: Any, count: int) -> list[Path]:
    """Up to ``count`` pictures spread over a data.yaml's train split, a folder or a list."""
    from .data.dataset import list_images, load_data_yaml

    if data is None:
        raise ValueError("int8=True needs pictures to calibrate on: data='data.yaml' or a folder")
    if isinstance(data, (list, tuple)):
        files = [Path(f) for f in data]
    else:
        path = Path(data).expanduser()
        if path.suffix in (".yaml", ".yml"):
            cfg = load_data_yaml(path)
            files = list_images(cfg["root"], cfg["train"], cfg["yaml_dir"])
        elif path.is_dir():
            files = list_images(path, ".")
        else:
            raise ValueError(f"{data}: give a data.yaml, a folder of pictures or a list")
    if not files:
        raise ValueError(f"no pictures found in {data}")
    step = max(len(files) // max(int(count), 1), 1)
    return files[::step][:count]


class _Stream:
    """The generator ``stream=True`` hands back, plus a nudge when it is dropped.

    ``model.predict(0, stream=True, show=True)`` on its own looks like it should
    open a camera, but a generator nobody iterates never runs a single frame —
    the call just returns and the program exits in silence. Saying so is much
    kinder than letting people debug an empty window that never appeared.
    """

    def __init__(self, generator, verb: str) -> None:
        self._generator = generator
        self._verb = verb
        self._started = False

    def __iter__(self):
        self._started = True
        return self._generator

    def __next__(self):
        self._started = True
        return next(self._generator)

    def close(self) -> None:
        self._started = True
        self._generator.close()

    def __del__(self) -> None:
        if self._started:
            return
        try:
            print(
                f"easydetect: {self._verb}(stream=True) returned a generator that was never "
                f"iterated, so nothing ran. Use it in a loop:\n"
                f"    for r in model.{self._verb}(source, stream=True, ...):\n"
                f"        ...\n"
                f"or drop stream=True to get a list back.",
                file=sys.stderr,
            )
        except Exception:  # interpreter shutdown - nothing useful to say
            pass


def _display(result: Results, frame, line_width: int | None = None) -> bool:
    """Show one frame without blocking the stream. False means "stop"."""
    import cv2

    window = Path(frame.path).name or "easydetect"
    cv2.imshow(window, result.plot(line_width=line_width))
    key = cv2.waitKey(1 if frame.kind != "image" else 0) & 0xFF
    if key in (ord("q"), 27):  # q or Esc
        return False
    return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1


def _close_windows() -> None:
    import cv2

    cv2.destroyAllWindows()
    cv2.waitKey(1)  # let the window manager actually take the close down


class _OutputWriter:
    """Writes annotated images as .jpg and annotated video frames as .mp4."""

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.videos: dict[str, Any] = {}

    def write(self, result: Results, frame, line_width: int | None = None) -> None:
        import cv2

        img = result.plot(line_width=line_width)
        if frame.kind == "image":
            stem = Path(frame.path).stem or "image"
            out = self.save_dir / f"{stem}.jpg"
            if out.exists():
                out = self.save_dir / f"{stem}_{frame.index}.jpg"
            cv2.imwrite(str(out), img)
            return
        writer = self.videos.get(frame.path)
        if writer is None:
            stem = Path(frame.path).stem or "stream"
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(
                str(self.save_dir / f"{stem}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                frame.fps or 25.0,  # keep the source's timing, not a guess
                (w, h),
            )
            self.videos[frame.path] = writer
        writer.write(img)

    def close(self) -> None:
        for writer in self.videos.values():
            writer.release()
        self.videos.clear()


def transfer_weights(net, state_dict) -> int:
    """Copy every tensor whose name *and* shape match. Returns how many landed."""
    target = net.state_dict()
    matched = {
        k: v for k, v in state_dict.items() if k in target and target[k].shape == v.shape
    }
    target.update(matched)
    net.load_state_dict(target)
    return len(matched)


def _import_torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise ImportError(_TORCH_HINT) from exc


__all__ = ["Detector", "increment_path", "transfer_weights"]
