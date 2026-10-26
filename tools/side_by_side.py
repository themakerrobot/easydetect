#!/usr/bin/env python3
# Apache-2.0
"""Several detectors on the same pictures, side by side: what each finds, how fast.

    python tools/side_by_side.py --models yolo11s.onnx ed-n ed-s \\
        --source photos/ --imgsz 320 --out side
    python tools/side_by_side.py --models yolo11s.onnx runs/ed-nq-320/weights/best.pt 0

A model is a ed name (ed-n …), a ed checkpoint or export (.pt,
.onnx, .xml), or a YOLO .onnx (YOLOv8/YOLO11, or an end-to-end YOLO26),
told apart by its outputs. ``--source`` is a picture, a folder or a camera
number. Every model runs on OpenVINO on this PC's CPU (all its cores), YOLO with
its own preprocessing (letterboxed) and NMS, ed resized and without; each
panel says how many boxes it kept at its ``conf`` and how long the network
took. Pictures are written to ``--out``, one row of panels each; a camera is
shown in a window (q quits). The mean times close the run.
"""

from __future__ import annotations

import argparse
import ast
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _is_yolo(path: str) -> bool:
    """A YOLO export has one output; a ed one gives boxes and scores apart."""
    if not path.endswith(".onnx"):
        return False
    import onnx

    return len(onnx.load(path, load_external_data=False).graph.output) == 1


def _yolo_names(path: str) -> dict[int, str]:
    import onnx

    meta = {p.key: p.value for p in onnx.load(path, load_external_data=False).metadata_props}
    try:
        return {int(k): str(v) for k, v in ast.literal_eval(meta.get("names", "{}")).items()}
    except (ValueError, SyntaxError):
        return {}


class YoloModel:
    def __init__(self, path, imgsz):
        from compare_yolo import Yolo

        self.label = Path(path).name
        self.run = Yolo(path, "CPU", imgsz)
        self.names = _yolo_names(path)
        self.conf = 0.25                    # YOLO's own default

    def __call__(self, img):
        h, w = img.shape[:2]
        xyxy, score, cls, ms = self.run(img, conf=self.conf)
        return xyxy * [w, h, w, h], score, cls, ms


class EdModel:
    def __init__(self, name, imgsz):
        from easydetect import Detector

        path = Path(name)
        # runs/<run>/weights/best.pt says its run; any other file its own name
        self.label = (path.parent.parent.name if path.parent.name == "weights"
                      else path.stem if path.suffix else name)
        self.model = Detector(name, device="CPU", verbose=False)
        self.imgsz = imgsz
        self.conf = 0.5                     # the lab's default
        self.model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=imgsz, verbose=False)
        self.names = self.model.names or {}

    def __call__(self, img):
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return r.boxes.xyxy, r.boxes.conf, r.boxes.cls.astype(int), r.speed["inference"]


def panel(img, found, model, height=360):
    xyxy, score, cls, ms = found
    out = img.copy()
    for (x1, y1, x2, y2), s, c in zip(np.asarray(xyxy), np.asarray(score), np.asarray(cls),
                                      strict=False):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (40, 200, 255), 2)
        name = str(model.names.get(int(c), int(c)))
        if not name.isascii():
            name = f"#{int(c)}"             # OpenCV draws no Hangul
        cv2.putText(out, f"{name} {s:.2f}", (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 200, 255), 1, cv2.LINE_AA)
    scale = height / out.shape[0]
    out = cv2.resize(out, (int(out.shape[1] * scale), height))
    bar = np.full((28, out.shape[1], 3), 30, np.uint8)
    title = f"{model.label}  {len(np.asarray(score))} boxes  {ms:.0f} ms  conf {model.conf}"
    cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("source", nargs="?", default=None, help="picture, folder or camera (0)")
    parser.add_argument("--source", dest="source_opt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--out", default="side-by-side")
    args = parser.parse_args()
    source = args.source_opt or args.source
    if source is None:
        parser.error("a picture, a folder or a camera number")
    models = [YoloModel(m, args.imgsz) if _is_yolo(m) else EdModel(m, args.imgsz)
              for m in args.models]
    times = {m.label: [] for m in models}

    def row(img):
        panels = []
        for m in models:
            found = m(img)
            times[m.label].append(found[3])
            panels.append(panel(img, found, m))
        return np.hstack(panels)

    if source.isdigit():
        cam = cv2.VideoCapture(int(source))
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            cv2.imshow("side by side (q quits)", row(frame))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        cam.release()
        cv2.destroyAllWindows()
    else:
        from easydetect.data.dataset import IMG_EXT
        from easydetect.utils.imageio import imread, imwrite

        src = Path(source)
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXT) \
            if src.is_dir() else [src]
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            img = imread(str(f))
            if img is None:
                continue
            imwrite(str(out / f"{f.stem}.jpg"), row(img))
        print(f"{len(files)} pictures -> {out}/")
    print("| model | mean ms (network only) | pictures |")
    print("| --- | --- | --- |")
    for label, ms in times.items():
        if ms:
            # the first call compiles and warms up: left out of the mean
            kept = ms[1:] or ms
            print(f"| {label} | {statistics.mean(kept):.1f} | {len(ms)} |")


if __name__ == "__main__":
    main()
