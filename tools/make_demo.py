#!/usr/bin/env python3
# Apache-2.0
"""Render the README's demo image with this package.

    python tools/make_demo.py --model ed-s --out docs/assets/demo.jpg

The default source is a frame of OpenCV's `vtest.avi` sample (Apache-2.0, from
the opencv/opencv repository), so the result can ship inside this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

VTEST = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"


def main(argv: list[str] | None = None) -> int:
    import cv2

    from easydetect import Detector
    from easydetect.downloads import cache_dir, download

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ed-s")
    parser.add_argument("--source", help="image path (default: an OpenCV sample frame)")
    parser.add_argument("--frame", type=int, default=0, help="frame index for a video source")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--out", default="docs/assets/demo.jpg")
    parser.add_argument("--quality", type=int, default=88)
    args = parser.parse_args(argv)

    if args.source:
        img = cv2.imread(args.source)
        if img is None:
            raise SystemExit(f"could not read {args.source}")
    else:
        clip = download(VTEST, cache_dir() / "samples" / "vtest.avi")
        cap = cv2.VideoCapture(str(clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("could not read a frame from the sample clip")

    result = Detector(args.model, verbose=False)(img, conf=args.conf)[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result.plot(), [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB)  {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
