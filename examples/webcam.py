# Apache-2.0
"""Live detection from a webcam (or any stream), drawn in an OpenCV window.

    python examples/webcam.py                        # camera 0, ed-s
    python examples/webcam.py --source 1 --track     # camera 1, with track ids
    python examples/webcam.py --model best.pt --conf 0.4
    python examples/webcam.py --source rtsp://cam/live --save

Press q (or Esc) to quit.
"""

from __future__ import annotations

import argparse
import time

import cv2

from easydetect import Detector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ed-s", help="name, .pt, or .xml")
    parser.add_argument("--source", default="0", help="camera index, video file, or stream url")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="AUTO", help="OpenVINO device: AUTO, CPU, GPU, NPU")
    parser.add_argument("--track", action="store_true", help="keep an id on each box")
    parser.add_argument("--save", action="store_true", help="also write runs/detect/predict*/")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = Detector(args.model, device=args.device, verbose=False)
    run = model.track if args.track else model.predict

    fps, last = 0.0, time.perf_counter()
    for result in run(source, conf=args.conf, stream=True, save=args.save):
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)  # smoothed, not jumpy
        last = now

        frame = result.plot()
        cv2.putText(
            frame,
            f"{fps:4.1f} FPS   {result.verbose().rstrip(', ')}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("easydetect", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
