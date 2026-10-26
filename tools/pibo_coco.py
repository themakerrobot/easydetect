#!/usr/bin/env python3
# Apache-2.0
"""One .onnx file for the Pibo robot: a COCO detector in place of its yolo11s.onnx.

    python tools/pibo_coco.py --weights ~/longrun/coco/ed-sq-320/weights/best.pt \\
        --out ~/pibo-ed-s --name ed-s-320 --fast

Writes ``<out>/<name>.onnx`` at 320 px with the 80 COCO names inside it
(metadata ``easydetect.names``), and with ``--fast`` ``<name>-fast.onnx``
with 2 decoder layers and 50 queries: lighter, its accuracy not yet measured.
The file is all the robot needs; ``easydetect-lab/pibo/detect.py`` runs it
on its own (``python3 detect.py photo.jpg --model <name>.onnx``) and is what a
reader of these files on the robot must match (see
``easydetect-lab/docs/pibo-integration.md``).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build(weights: Path, target: Path, size: int, conf: float | None = None, **trim) -> Path:
    from easydetect import Detector
    from easydetect.exporter import embed_metadata
    from easydetect.predictor import ONNX_CONF_KEY

    target.parent.mkdir(parents=True, exist_ok=True)
    work = target.parent / f".{target.stem}-export"
    written = Path(Detector(str(weights), verbose=False).export(
        format="onnx", imgsz=size, out_dir=work, verbose=False, **trim))
    # the names are inside already; a conf only when one was picked for this model
    embed_metadata(written, {ONNX_CONF_KEY: None if conf is None else f"{conf:.2f}"})
    shutil.move(str(written), target)
    shutil.rmtree(work, ignore_errors=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help="a COCO checkpoint (.pt) or a model name")
    parser.add_argument("--out", default="pibo-coco")
    parser.add_argument("--name", help="file name without .onnx (default: from the weights)")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, help="score a box needs (default: the reader's 0.5)")
    parser.add_argument("--fast", action="store_true",
                        help="also <name>-fast.onnx with 2 decoder layers and 50 queries")
    args = parser.parse_args()

    weights = Path(args.weights).expanduser()
    name = args.name or (weights.parent.parent.name if weights.parent.name == "weights"
                         else weights.stem)
    out = Path(args.out).expanduser()
    made = [build(weights, out / f"{name}.onnx", args.imgsz, args.conf)]
    if args.fast:
        made.append(build(weights, out / f"{name}-fast.onnx", args.imgsz, args.conf,
                          layers=2, queries=50))
    for f in made:
        print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
