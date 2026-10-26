#!/usr/bin/env python3
# Apache-2.0
"""Time easydetect on every OpenVINO device this machine has.

    python tools/bench.py                          # ed-s, every device
    python tools/bench.py --models ed-n ed-s ed-m
    python tools/bench.py --models runs/train/weights/best.pt --devices CPU NPU

The input size is the model's own (640 for the mirror's IRs). To time 320,
export an IR at that size and pass its .xml:
``Detector("ed-s").export(imgsz=320, out_dir="ir320")``.

Whole pipeline — resize, inference, decode — median of ``--runs`` calls after
five warm-ups, on one frame of OpenCV's sample street clip (or ``--image``).
Each device's boxes are checked against the CPU's, so a fast device that gets
the answer wrong shows up as such.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"


def frame(path: str | None):
    import cv2

    from easydetect.downloads import cache_dir, download

    if path:
        image = cv2.imread(path)
        if image is None:
            raise SystemExit(f"could not read {path}")
        return image
    cap = cv2.VideoCapture(str(download(SAMPLE, cache_dir() / "samples" / "vtest.avi")))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read the sample clip")
    return image


def agreement(ref, got) -> str:
    """Mean and worst IoU of each reference box with its best same-class match."""
    import numpy as np

    if not len(ref.boxes):
        return "no boxes to compare"
    ious = []
    for box, cls in zip(ref.boxes.xyxy, ref.boxes.cls, strict=True):
        best = 0.0
        for other, ocls in zip(got.boxes.xyxy, got.boxes.cls, strict=True):
            if ocls != cls:
                continue
            x1, y1 = np.maximum(box[:2], other[:2])
            x2, y2 = np.minimum(box[2:], other[2:])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            union = np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]) - inter
            best = max(best, float(inter / union) if union > 0 else 0.0)
        ious.append(best)
    return f"IoU vs CPU {np.mean(ious):.3f} (worst {min(ious):.3f})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["ed-s"])
    parser.add_argument("--devices", nargs="+", help="default: CPU, GPU, NPU — whichever exist")
    parser.add_argument("--image")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args(argv)

    import openvino as ov

    from easydetect import Detector

    image = frame(args.image)
    available = ov.Core().available_devices
    devices = args.devices or [d for d in ("CPU", "GPU", "NPU")
                               if any(a.startswith(d) for a in available)]
    print(f"OpenVINO {ov.__version__}; devices {available}; "
          f"frame {image.shape[1]}x{image.shape[0]}\n")
    print(f"{'model':<12} {'imgsz':>5} {'device':<6} {'latency':>9} {'FPS':>6} {'boxes':>5}  check")

    for name in args.models:
        reference = None
        for device in devices:
            try:
                model = Detector(name, device=device, verbose=False)
                for _ in range(5):
                    result = model(image, conf=args.conf)[0]
                times = []
                for _ in range(args.runs):
                    t = time.perf_counter()
                    result = model(image, conf=args.conf)[0]
                    times.append((time.perf_counter() - t) * 1000)
            except Exception as exc:  # noqa: BLE001 — report and carry on
                first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
                print(f"{Path(name).stem:<12} {'':>5} {device:<6} FAILED: {first}")
                continue
            ms = statistics.median(times)
            if reference is None:
                reference, check = result, "reference"
            else:
                check = agreement(reference, result)
            imgsz = model.predictor.imgsz if model.predictor else ""
            print(f"{Path(name).stem:<12} {imgsz:>5} {device:<6} {ms:7.1f}ms {1000 / ms:6.1f} "
                  f"{len(result.boxes):>5}  {check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
