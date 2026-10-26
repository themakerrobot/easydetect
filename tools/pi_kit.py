#!/usr/bin/env python3
# Apache-2.0
"""Export the models worth trying on a small CPU (a Raspberry Pi), with their accuracy.

    python tools/pi_kit.py --models ed-n ed-s --sizes 320 416 640 \\
        --data data.yaml --out pi-kit
    # copy pi-kit/ to the Pi, then there:
    easydetect benchmark model=pi-kit/ threads=4

On the PC (it needs PyTorch): every model at every input size, as OpenVINO
IR, and, with ``--data``, as an 8-bit IR calibrated on its train pictures;
``--queries`` adds lighter decoders (fewer queries). An ONNX copy of the
smallest variant of each lets ONNX Runtime be timed too. With ``--data`` each
one is scored on the val split (COCO mAP50-95, OpenVINO on this CPU in float32;
``--limit`` pictures), and ``accuracy.md`` beside the files lists them, so
the Pi's timings and these scores make one table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def score(path: Path, data: str, limit: int | None, backend: str | None = None) -> dict:
    """mAP of one exported model on the val split, as tools/compare_yolo.py scores.
    ``backend="onnxruntime"`` scores an .onnx the way the Pibo robot runs it."""
    import cv2
    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    from compare_yolo import evaluate

    from easydetect import Detector
    from easydetect.data.dataset import DetDataset

    ds = DetDataset(data, "val", 64, augment=False)
    files = ds.files[:limit] if limit else ds.files
    truth = []
    for f in files:
        lab = ds._load_labels(f)
        cx, cy, bw, bh = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
        truth.append((np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1),
                      lab[:, 0].astype(int)))
    model = Detector(str(path), device="CPU", verbose=False,
                     **({"backend": backend} if backend else {}))
    model(np.zeros((32, 32, 3), np.uint8))

    class Runner:
        def __call__(self, img, conf=0.001, max_det=300):
            h, w = img.shape[:2]
            r = model.predict(img, conf=conf, max_det=max_det, iou=None, verbose=False)[0]
            return (r.boxes.xyxy / [w, h, w, h], r.boxes.conf, r.boxes.cls.astype(int),
                    r.speed["inference"])

    del cv2
    m50, m, ms = evaluate(path.name, Runner(), files, truth, ds.nc)
    return {"map50": round(m50, 4), "map": round(m, 4), "ms_here": round(ms, 1)}


def ort_int8(onnx: Path, out: Path, pictures: list, size: int) -> Path:
    """An 8-bit copy of an .onnx for ONNX Runtime (what a Pi runs), calibrated
    on ``pictures``: the convolutions and matrix products only, per channel,
    as QDQ — the box decoding and the top-k selection stay in float."""
    import cv2
    import numpy as np
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from easydetect.utils.imageio import imread

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.files = iter(pictures)

        def get_next(self):
            for f in self.files:
                img = imread(str(f))
                if img is None:
                    continue
                x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))
                return {"images": (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]}
            return None

    out.parent.mkdir(parents=True, exist_ok=True)
    pre = out.with_name(out.stem + "-pre.onnx")
    quant_pre_process(str(onnx), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["Conv", "MatMul"], per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)
    names = onnx.with_name(onnx.stem + ".names.json")   # the classes travel beside it
    if names.exists():
        out.with_name(out.stem + ".names.json").write_text(names.read_text())
    return out


def _ir(xml) -> Path:
    """An exported IR, without the float ONNX the export went through (a
    ``*_int8.onnx`` beside an 8-bit IR is that float model, not 8-bit)."""
    xml = Path(xml)
    xml.with_suffix(".onnx").unlink(missing_ok=True)
    return xml


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["ed-n", "ed-s"],
                        help="model names (ed-n …) or your best.pt files")
    parser.add_argument("--sizes", nargs="+", type=int, default=[320, 416, 640])
    parser.add_argument("--queries", nargs="*", type=int, default=[100],
                        help="also export with these many decoder queries (of 300)")
    parser.add_argument("--data", help="data.yaml: calibrates the 8-bit models and scores all")
    parser.add_argument("--limit", type=int, default=500, help="val pictures scored")
    parser.add_argument("--calib", type=int, default=300, help="pictures to calibrate 8-bit on")
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--out", default="pi-kit")
    args = parser.parse_args()

    from easydetect import Detector

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name in args.models:
        stem = Path(name).stem if name.endswith(".pt") else name
        for size in args.sizes:
            for queries in [None, *args.queries]:
                tag = f"{stem}-{size}" + (f"-q{queries}" if queries else "")
                model = Detector(name, verbose=False)
                xml = model.export(format="openvino", imgsz=size, queries=queries,
                                   out_dir=out / tag, verbose=False)
                made.append(_ir(xml))
                if args.data and not args.no_int8:
                    q8 = model.export(format="openvino", imgsz=size, queries=queries,
                                      int8=True, data=args.data, calib=args.calib,
                                      out_dir=out / f"{tag}-int8", verbose=False)
                    made.append(_ir(q8))
                if queries is None and size == min(args.sizes):
                    made.append(Path(model.export(format="onnx", imgsz=size,
                                                  out_dir=out / f"{tag}-onnx", verbose=False)))
                print(f"[pi-kit] {tag} done", flush=True)

    rows = {}
    if args.data:
        for f in made:
            key = f"{f.parent.name}/{f.name}"
            rows[key] = score(f, args.data, args.limit)
            print(f"[pi-kit] {key}: {rows[key]}", flush=True)
        (out / "accuracy.json").write_text(json.dumps(rows, indent=2))
        lines = ["| model file | mAP50-95 | mAP50 | ms on this PC |", "| --- | --- | --- | --- |"]
        lines += [f"| {k} | {v['map']:.3f} | {v['map50']:.3f} | {v['ms_here']} |"
                  for k, v in rows.items()]
        (out / "accuracy.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    print(f"\ncopy {out}/ to the Pi, then: easydetect benchmark model={out.name}/ threads=4")


if __name__ == "__main__":
    main()
