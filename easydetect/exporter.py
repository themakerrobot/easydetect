# Apache-2.0
"""Export: EDNet -> ONNX (opset 17) -> OpenVINO IR, static shape, FP16 optional."""

from __future__ import annotations

import copy
import json
import threading
import warnings
from pathlib import Path

from easydetect.utils.imageio import imread  # any path, Hangul included

#: torch.onnx keeps global state while exporting ("in_onnx_export"), so two
#: exports at once fail an assertion deep inside it. Serialise them — the
#: whole export, names and conversion included, so that two threads exporting
#: into one folder cannot interleave their files (re-entrant: the OpenVINO
#: export runs the ONNX one inside it).
_EXPORT_LOCK = threading.RLock()


def _write_labels(out_dir: Path, fname: str, names) -> dict[int, str]:
    """Write ``labels.txt`` (one class per line) + ``<fname>.names.json``.

    labels.txt beside the IR is how downstream runtimes (ovkit among them)
    discover class names, so a freshly trained model answers "my-class 0.91"
    with no extra wiring.
    """
    table = {int(k): str(v) for k, v in (names or {}).items()}
    (out_dir / f"{fname}.names.json").write_text(
        json.dumps(table, ensure_ascii=False), encoding="utf-8"
    )
    if table:
        lines = [table.get(i, f"class_{i}") for i in range(max(table) + 1)]
        (out_dir / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def export_onnx(net, names, imgsz=640, out_dir=".", fname="easydetect", half=False, verbose=True,
                layers=None, queries=None):
    """Write ``<out_dir>/<fname>.onnx`` (plus labels). Returns the .onnx path.

    ``layers`` / ``queries`` export a lighter decoder: see ``EDNet.trim``.
    """
    import torch

    from .nn import DeployWrapper

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{fname}.onnx"

    # the deploy form fuses the re-parameterised blocks and drops the training
    # heads; a copy, so the caller's network can keep training
    deployed = copy.deepcopy(net).cpu()
    if layers is not None or queries is not None:
        deployed.trim(layers=layers, queries=queries)
    deployed = deployed.deploy() if hasattr(deployed, "deploy") else deployed.eval()
    wrapper = DeployWrapper(deployed).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with _EXPORT_LOCK:
        _export_onnx_locked(wrapper, dummy, onnx_path, out_dir, fname, names, half)
    if verbose:
        print(f"[easydetect] exported: {onnx_path}" + (" (FP16)" if half else ""))
    return onnx_path


def _export_onnx_locked(wrapper, dummy, onnx_path, out_dir, fname, names, half):
    import torch

    with warnings.catch_warnings():
        # The graph is exported at a fixed input size on purpose, so the tracer
        # baking shape-dependent constants (the query count) in is what we want.
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)   # shape-inference chatter
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["images"],
            output_names=["boxes", "scores"],
            opset_version=17,
            dynamo=False,
        )
    if half:
        _to_fp16(onnx_path)
    table = _write_labels(out_dir, fname, names)
    _embed_names(onnx_path, table)


def _to_fp16(onnx_path: Path) -> None:
    """FP16 weights and arithmetic, float32 in and out (``half=True`` for an
    ONNX: half the file, faster on GPUs; on a CPU, little). Needs
    ``onnxconverter-common``; ``half`` used to be ignored here silently."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise ImportError("half=True for ONNX needs onnxconverter-common: "
                          "pip install onnxconverter-common (the OpenVINO export "
                          "does FP16 without it)") from exc
    model = float16.convert_float_to_float16(onnx.load(str(onnx_path)), keep_io_types=True)
    onnx.save(model, str(onnx_path))


def _embed_names(onnx_path: Path, table: dict[int, str]) -> None:
    """Keep the class names inside the .onnx too, so the file works on its own —
    one download from a Hugging Face page, no labels.txt to fetch beside it."""
    from .predictor import ONNX_NAMES_KEY

    embed_metadata(onnx_path, {ONNX_NAMES_KEY: json.dumps(
        {str(k): v for k, v in table.items()}, ensure_ascii=False)})


def embed_metadata(onnx_path: str | Path, entries: dict[str, str | None]) -> None:
    """Set (or, with None, remove) metadata entries inside an .onnx. Any ONNX
    runtime reads them back: onnxruntime's ``get_modelmeta().custom_metadata_map``."""
    import onnx

    model = onnx.load(str(onnx_path))
    kept = [p for p in model.metadata_props if p.key not in entries]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    for key, value in entries.items():
        if value is not None:
            entry = model.metadata_props.add()
            entry.key, entry.value = key, str(value)
    onnx.save(model, str(onnx_path))


def export_openvino(net, names, imgsz=640, out_dir=".", fname="easydetect", half=False,
                    verbose=True, layers=None, queries=None, int8=None):
    """Write ``<out_dir>/<fname>.xml`` (+ .bin, + labels). Returns the .xml path.

    ``int8`` is a list of image paths to calibrate an 8-bit model on (NNCF's
    post-training quantization; a few hundred pictures like the ones it will
    see). It runs about as fast as bfloat16 where the CPU has that, and over
    twice as fast as float32 where it does not — most laptop CPUs.
    """
    import openvino as ov

    out_dir = Path(out_dir)
    xml_path = out_dir / f"{fname}.xml"
    with _EXPORT_LOCK:
        onnx_path = export_onnx(
            net, names, imgsz=imgsz, out_dir=out_dir, fname=fname, verbose=False,
            layers=layers, queries=queries,
        )
        model = ov.convert_model(str(onnx_path))
        agreement = None
        if int8:
            fp32 = model
            model = _quantize(model, int8, imgsz)
            agreement = _int8_agreement(fp32, model, list(int8), imgsz)
        ov.save_model(model, str(xml_path), compress_to_fp16=half and not int8)
    if verbose:
        kind = "INT8" if int8 else "FP16" if half else "FP32"
        print(f"[easydetect] exported: {xml_path} ({kind})")
    if agreement is not None:
        found, total = agreement
        share = found / total if total else 1.0
        print(f"[easydetect] INT8 against FP32 on {min(len(int8), AGREEMENT_PICTURES)} of the "
              f"calibration pictures: {found} of {total} boxes at conf 0.5 found again "
              f"({share:.0%})")
        if share < 0.9:
            warnings.warn(f"the INT8 model misses {total - found} of {total} of the FP32 "
                          "model's boxes at conf 0.5; check it on your validation pictures "
                          "before using it", stacklevel=2)
    return xml_path


#: Pictures the INT8 check runs (from the calibration set).
AGREEMENT_PICTURES = 50


def _int8_agreement(fp32, int8, images, imgsz) -> tuple[int, int]:
    """How many of the FP32 model's boxes (conf 0.5) the INT8 one finds again
    (same class, IoU 0.5) on some of the calibration pictures: NNCF quantizes
    without checking, and a model that loses objects should say so."""
    import numpy as np
    import openvino as ov

    from .predictor import Predictor, _split_outputs, preprocess_image

    core = ov.Core()
    runs = [core.compile_model(m, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            for m in (fp32, int8)]
    decode = Predictor()
    step = max(len(images) // AGREEMENT_PICTURES, 1)
    found = total = 0
    for path in images[::step][:AGREEMENT_PICTURES]:
        img = imread(str(path))
        if img is None:
            continue
        x = preprocess_image(img, imgsz)
        a, b = (decode.postprocess(*_split_outputs(list(r(x).values())), img.shape[:2])
                for r in runs)
        total += len(a)
        for box in a:
            same = b[b[:, 5] == box[5]]
            if len(same):
                lt = np.maximum(box[:2], same[:, :2])
                rb = np.minimum(box[2:4], same[:, 2:4])
                inter = np.prod(np.clip(rb - lt, 0, None), 1)
                union = (np.prod(box[2:4] - box[:2]) + np.prod(same[:, 2:4] - same[:, :2], 1)
                         - inter)
                found += bool((inter / np.maximum(union, 1e-9) >= 0.5).any())
    return found, total


def _quantize(model, images, imgsz):
    """8-bit weights and activations, calibrated on ``images`` (paths)."""
    import nncf

    from .predictor import preprocess_image

    images = [str(p) for p in images]
    if not images:
        raise ValueError("int8 needs pictures to calibrate on: give data=")

    def transform(path):
        img = imread(path)
        if img is None:
            raise ValueError(f"cannot read {path}")
        return preprocess_image(img, imgsz)

    return nncf.quantize(model, nncf.Dataset(images, transform), subset_size=len(images),
                         model_type=nncf.ModelType.TRANSFORMER)
