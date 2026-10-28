#!/usr/bin/env python3
# Apache-2.0
"""A COCO fine-tune at a smaller input as a mirror start checkpoint.

    python tools/mirror_start.py --weights ~/longrun/coco/ed-nq-320/weights/best.pt \\
        --name ed-n --out ~/mirror-start
    hf upload leeyunjai/easydetect ~/mirror-start . --repo-type=model

Writes ``<out>/<name>/<name>-<imgsz>.pt``: the EMA weights and what loading
them needs, without the optimizer state. ``train(imgsz=320, queries=100)``
starts from it (``downloads.SIZED_STARTS``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

KEEP = ("model", "variant", "num_classes", "decoder_act", "queries", "denoising", "names",
        "imgsz")


def build(weights: Path, out: Path, name: str) -> Path:
    import torch

    from easydetect.downloads import SIZED_STARTS, normalize_name
    from easydetect.utils.checkpoint import load_checkpoint

    ckpt = load_checkpoint(weights)
    name = normalize_name(name)
    if f"ed-{ckpt['variant']}" != name:
        raise SystemExit(f"{weights} is ed-{ckpt['variant']}, not {name}")
    imgsz = int(ckpt.get("imgsz") or 0)
    if SIZED_STARTS.get(imgsz) != ckpt.get("queries"):
        raise SystemExit(f"{weights}: {imgsz} px with {ckpt.get('queries')} queries; "
                         f"a start must match SIZED_STARTS {SIZED_STARTS}")
    target = out / name / f"{name}-{imgsz}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: ckpt[k] for k in KEEP if k in ckpt}, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name", required=True, help="ed-n or ed-s")
    parser.add_argument("--out", default="mirror-start")
    args = parser.parse_args()
    target = build(Path(args.weights).expanduser(), Path(args.out).expanduser(), args.name)
    print(f"{target}  {target.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
