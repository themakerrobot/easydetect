#!/usr/bin/env python3
# Apache-2.0
"""Where an .onnx spends its time on this machine (a Raspberry Pi, say):
milliseconds a picture with 1-4 threads, then ONNX Runtime's own profile
by part of the network and by operator. Needs only numpy and onnxruntime.

    python3 pi_profile.py /home/pi/.model/object/ed-s-320.onnx
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort


def session(path: str, threads: int, profile: bool = False):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.enable_profiling = profile
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def timed(s, feed, n: int) -> float:
    for _ in range(3):
        s.run(None, feed)
    t = time.perf_counter()
    for _ in range(n):
        s.run(None, feed)
    return 1000 * (time.perf_counter() - t) / n


def part(name: str) -> str:
    for key in ("backbone", "encoder/encoder", "fpn_blocks", "pan_blocks", "cross_attn",
                "self_attn", "decoder/decoder", "enc_output", "encoder", "decoder"):
        if key in name:
            return key
    return "(rest)"


def main() -> int:
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    s = session(path, 4)
    inp = s.get_inputs()[0]
    shape = [d if isinstance(d, int) else 320 for d in inp.shape]
    feed = {inp.name: np.random.rand(*shape).astype(np.float32)}
    print(f"onnxruntime {ort.__version__}, {os.cpu_count()} cores, input {shape}")
    for threads in (1, 2, 3, 4):
        print(f"threads {threads}: {timed(session(path, threads), feed, n):.1f} ms")
    s = session(path, 4, profile=True)
    timed(s, feed, n)
    events = json.load(open(s.end_profiling()))
    ops, parts = collections.Counter(), collections.Counter()
    for e in events:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            ops[e["args"]["op_name"]] += e["dur"]
            parts[part(e["name"])] += e["dur"]
    total = sum(ops.values())
    print("\nby part:")
    for k, v in parts.most_common():
        print(f"  {100 * v / total:5.1f}%  {k}")
    print("by operator:")
    for k, v in ops.most_common(12):
        print(f"  {100 * v / total:5.1f}%  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
