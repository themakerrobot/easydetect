# Apache-2.0
"""``easydetect <mode> key=value ...`` — the command line for this package.

    easydetect predict model=dfine-s source=bus.jpg conf=0.5
    easydetect train   model=dfine-s data=data.yaml epochs=100 imgsz=640
    easydetect val     model=best.pt data=data.yaml
    easydetect export  model=best.pt format=openvino half=true
  easydetect serve   model=best.onnx host=0.0.0.0 port=8000
    easydetect track   model=best.pt source=video.mp4
"""

from __future__ import annotations

import sys
from typing import Any

MODES = ("predict", "track", "train", "val", "export", "serve")

HELP = """easydetect — real-time object detection (D-FINE), Apache-2.0

Usage:  easydetect <mode> key=value ...

Modes:
  predict   run detection on an image / folder / video / url / camera
  track     predict and keep an id on each box across frames
  train     train on a data.yaml dataset (images/ + labels/*.txt)
  val       COCO-style mAP50 / mAP50-95 on the val split
  export    write an OpenVINO IR (or ONNX) next to the checkpoint
  serve     answer POST /predict with detections as JSON (an HTTP server)

Common keys:
  model=dfine-n|s|m|l|x|best.pt|model.xml   source=bus.jpg|dir|video.mp4|0|url
  data=data.yaml  epochs=100  imgsz=640  batch=8  conf=0.5  iou=0.7|none  device=0|cpu|AUTO
  contain=0.8  task=segment|pose  project=runs  name=predict  save=true  show=false
  pose_model=s|m|my.onnx  pose_flip=false
  format=openvino  half=true  int8=true  layers=1  queries=100

Examples:
  easydetect predict model=dfine-s source=bus.jpg conf=0.5
  easydetect train   model=dfine-s data=data.yaml epochs=100
  easydetect val     model=best.pt data=data.yaml
  easydetect export  model=best.pt format=openvino half=true
  easydetect serve   model=best.onnx host=0.0.0.0 port=8000
"""


def parse_value(text: str) -> Any:
    """``true`` -> True, ``3`` -> 3, ``0.5`` -> 0.5, ``0,1`` -> [0, 1], else str."""
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    stripped = text.strip("[]")
    if "," in stripped:
        return [parse_value(part.strip()) for part in stripped.split(",") if part.strip()]
    return text


def parse_args(argv: list[str]) -> tuple[str, dict[str, Any]]:
    """-> ``(mode, overrides)``. The mode may be positional or ``mode=predict``."""
    mode = ""
    overrides: dict[str, Any] = {}
    for arg in argv:
        if "=" in arg:
            key, _, value = arg.partition("=")
            key = key.strip().lstrip("-")
            if key == "mode":
                mode = str(value)
            else:
                overrides[key] = parse_value(value)
        elif not mode:
            mode = arg
        else:
            raise SystemExit(f"unexpected argument {arg!r} — arguments are key=value pairs")
    return mode, overrides


def _take(overrides: dict[str, Any], key: str, default: Any = None) -> Any:
    return overrides.pop(key, default)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        return 0
    if argv[0] in ("-v", "--version", "version"):
        from . import __version__

        print(__version__)
        return 0

    mode, overrides = parse_args(argv)
    if mode not in MODES:
        print(f"unknown mode {mode!r}. Modes: {', '.join(MODES)}\n", file=sys.stderr)
        print(HELP, file=sys.stderr)
        return 2

    if mode == "serve":
        from .serve import serve

        keys = ("model", "host", "port", "device", "task", "backend", "pose_model", "pose_flip")
        unknown = set(overrides) - set(keys)
        if unknown:
            print(f"serve takes {', '.join(keys)} — not {', '.join(sorted(unknown))}",
                  file=sys.stderr)
            return 2
        serve(model=str(overrides.get("model", "dfine-s")),
              host=str(overrides.get("host", "127.0.0.1")), port=int(overrides.get("port", 8000)),
              device=str(overrides.get("device", "AUTO")), task=overrides.get("task", "detect"),
              backend=overrides.get("backend"), pose_model=str(overrides.get("pose_model", "s")),
              pose_flip=bool(overrides.get("pose_flip", True)))
        return 0

    from .model import Detector

    model_name = _take(overrides, "model", "dfine-s")
    device = _take(overrides, "device")
    verbose = _take(overrides, "verbose", True)

    if mode in ("predict", "track"):
        task = _take(overrides, "task", "detect")
        model = Detector(model_name, device=device or "AUTO", verbose=verbose, task=task)
        model.pose_model = _take(overrides, "pose_model", model.pose_model)
        model.pose_flip = bool(_take(overrides, "pose_flip", model.pose_flip))
        source = _take(overrides, "source")
        if source is None:
            print(
                "predict needs source=… (image, folder, video, url, camera index)",
                file=sys.stderr,
            )
            return 2
        overrides.setdefault("save", True)
        overrides["stream"] = False
        runner = model.predict if mode == "predict" else model.track
        results = runner(source, **overrides)
        print(f"{len(results)} result(s)")
        return 0

    model = Detector(model_name, verbose=verbose)
    if mode == "train":
        data = _take(overrides, "data")
        if data is None:
            print("train needs data=path/to/data.yaml", file=sys.stderr)
            return 2
        best = model.train(data=data, device=device, **overrides)
        print(best)
        return 0
    if mode == "val":
        data = _take(overrides, "data")
        if data is None:
            print("val needs data=path/to/data.yaml", file=sys.stderr)
            return 2
        metrics = model.val(data=data, device=device, **overrides)
        if not verbose:  # model.val() already printed the line when verbose
            print(f"mAP50 {metrics.box.map50:.4f}  mAP50-95 {metrics.box.map:.4f}")
        return 0

    path = model.export(**overrides)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
