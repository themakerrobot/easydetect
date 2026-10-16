# Apache-2.0
"""How fast each model runs here: ``easydetect benchmark model=a.xml,b.onnx``.

Made for the machine the model will run on (a Raspberry Pi, a mini PC), so it
needs only the inference dependencies, no PyTorch. Every model is run on every
runtime that can read it (OpenVINO reads .xml and .onnx, ONNX Runtime .onnx),
on one picture, ``runs`` times after ``warmup``; the medians of preprocessing,
inference and decoding are printed as a table, fastest first.

``model=`` takes files, folders (every .xml / .onnx inside, at any depth),
model names (``dfine-n``: the mirror's ready-made 640 px IR and ONNX), or a
mix. With no
``source=`` the picture is a synthetic 640×480 frame, the size a Pi camera
usually delivers; the time a model takes does not depend on what it sees.
"""

from __future__ import annotations

import os
import platform
import statistics
from pathlib import Path

import numpy as np


def _model_files(spec) -> list[Path]:
    items = spec if isinstance(spec, (list, tuple)) else [spec]
    files: list[Path] = []
    for item in items:
        path = Path(str(item)).expanduser()
        if not path.exists() and _is_name(str(item)):
            # a model name (dfine-n): the mirror's ready-made IR and ONNX
            from . import downloads

            for fetch in (downloads.download_ir, downloads.download_onnx):
                try:
                    files.append(fetch(str(item)))
                except Exception:  # noqa: BLE001 — one format missing is fine
                    pass
            continue
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in (".xml", ".onnx"))
        elif path.exists():
            files.append(path)
        else:
            raise FileNotFoundError(f"{path}: no such file or folder")
    if not files:
        raise FileNotFoundError(f"no .xml or .onnx models in {spec}")
    return files


def _is_name(text: str) -> bool:
    from . import downloads

    return downloads.is_model_name(text)


def _label(path: Path, spec) -> str:
    """A model's name in the table: its folder too when the folder says more
    (pi_kit.py writes ``dfine-n-320-q100/dfine-n.xml``)."""
    return f"{path.parent.name}/{path.name}" if path.parent.name not in ("", ".") else path.name


def _picture(source) -> np.ndarray:
    if source is None:
        rng = np.random.default_rng(0)
        import cv2

        noise = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        return cv2.GaussianBlur(noise, (0, 0), 3)
    import cv2

    img = cv2.imread(str(source))
    if img is None:
        raise FileNotFoundError(f"cannot read {source}")
    return img


def _predictor(path: Path, backend: str, device: str, threads: int | None):
    from .predictor import ORTPredictor, OVPredictor

    if backend == "openvino":
        return OVPredictor(path, device=device, threads=threads)
    return ORTPredictor(path, device=device, threads=threads)


def benchmark(model, source=None, runs: int = 50, warmup: int = 5,
              backend: str | list[str] | None = None, device: str = "CPU",
              threads: int | None = None, verbose: bool = True) -> list[dict]:
    """Time every model on every runtime; returns one dict a (model, runtime)."""
    from .predictor import installed

    files = _model_files(model)
    img = _picture(source)
    wanted = backend if isinstance(backend, (list, tuple)) else \
        [backend] if backend else ["openvino", "onnxruntime"]
    rows = []
    for path in files:
        for name in wanted:
            if not installed(name) or (name == "onnxruntime" and path.suffix != ".onnx"):
                continue
            try:
                predictor = _predictor(path, name, device, threads)
            except Exception as exc:  # noqa: BLE001 — one model failing must not stop the rest
                rows.append({"model": _label(path, model), "backend": name,
                             "error": str(exc)[:120]})
                continue
            for _ in range(warmup):
                predictor(img, conf=0.25)
            speeds = [predictor(img, conf=0.25)[1] for _ in range(runs)]
            row = {"model": _label(path, model), "backend": name, "imgsz": predictor.imgsz}
            for key in ("preprocess", "inference", "postprocess"):
                row[key] = statistics.median(s[key] for s in speeds)
            row["total"] = row["preprocess"] + row["inference"] + row["postprocess"]
            row["fps"] = 1000.0 / row["total"]
            rows.append(row)
    rows.sort(key=lambda r: r.get("total", float("inf")))
    if verbose:
        print(report(rows, threads))
    return rows


def report(rows: list[dict], threads: int | None = None) -> str:
    """The table ``benchmark`` prints, with the machine it ran on."""
    versions = []
    for module in ("openvino", "onnxruntime"):
        try:
            versions.append(f"{module} {__import__(module).__version__}")
        except ImportError:
            pass
    head = (f"{platform.machine()} {platform.system()}, {os.cpu_count()} cores"
            f"{f', {threads} threads' if threads else ''}; {', '.join(versions)}")
    lines = [head, f"{'model':40s} {'runtime':12s} {'size':>5s} {'pre':>7s} {'infer':>8s} "
                   f"{'post':>6s} {'total ms':>9s} {'fps':>6s}"]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['model']:40s} {r['backend']:12s}  failed: {r['error']}")
            continue
        lines.append(f"{r['model']:40s} {r['backend']:12s} {r['imgsz']:5d} {r['preprocess']:7.1f} "
                     f"{r['inference']:8.1f} {r['postprocess']:6.1f} {r['total']:9.1f} "
                     f"{r['fps']:6.1f}")
    return "\n".join(lines)
