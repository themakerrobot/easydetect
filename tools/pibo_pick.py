#!/usr/bin/env python3
# Apache-2.0
"""The best model for Pibo in a kit folder: the highest COCO mAP50-95
(official way) among the files fast enough on the Pi. Standard library only,
so it runs on the robot.

    python3 pi_profile.py ~/pi-kit-kd/            # writes times.json
    python3 pibo_pick.py ~/pi-kit-kd/ --ms 500    # copies the best to ed-s-pibo.onnx

``scores.json`` (written on the GPU machine by tools/long_run.py) holds each
file's accuracy, ``times.json`` its milliseconds on this Pi.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kit", help="a folder with scores.json and times.json")
    parser.add_argument("--ms", type=float, default=500.0, help="slowest allowed, 4 threads")
    parser.add_argument("--out", default="ed-s-pibo.onnx", help="where the best goes")
    args = parser.parse_args()
    kit = Path(args.kit).expanduser()
    scores = json.loads((kit / "scores.json").read_text())
    times = json.loads((kit / "times.json").read_text())
    rows = []
    for key, v in scores.items():
        ms = times.get(Path(v["file"]).parent.name)
        acc = v.get("map_official", v["map"])
        rows.append((acc, v["map"], ms, key, v["file"]))
    rows.sort(key=lambda r: -r[0])
    print(f"{'mAP50-95 official':>18} {'Pibo':>6} {'ms':>8}  model")
    for acc, pibo, ms, key, _ in rows:
        flag = "" if ms is None else ("  <= ok" if ms <= args.ms else "")
        print(f"{acc:18.3f} {pibo:6.3f} {ms if ms is not None else '?':>8}  {key}{flag}")
    fit = [r for r in rows if r[2] is not None and r[2] <= args.ms]
    if not fit:
        print(f"\nnothing at {args.ms:.0f} ms or less")
        return 1
    acc, pibo, ms, key, file = fit[0]
    out = Path(args.out).expanduser()
    shutil.copyfile(kit / file, out)
    print(f"\nbest: {key}: mAP50-95 {acc:.3f} (official), {pibo:.3f} (Pibo), {ms} ms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
