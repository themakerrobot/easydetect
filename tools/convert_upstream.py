#!/usr/bin/env python3
# Apache-2.0
"""Turn an official D-FINE COCO checkpoint into an easydetect one.

    python tools/convert_upstream.py --size s --out weights/ed-s.pt

With no --weights it downloads the matching Apache-2.0 COCO checkpoint from
the D-FINE release (https://github.com/Peterande/D-FINE) and caches it under
~/.easydetect/official/. Only the COCO-trained checkpoints are used: the
Objects365-pretrained ones may carry that dataset's terms.

This package's network matches the reference module for module, so the weights
load with ``strict=True``. The only tensors left behind are the anchor grid the
reference caches for a fixed 640 input — here it is computed from the input, so
any size works. The script adds our metadata (size, class names, imgsz) so
``Detector("x.pt")``, ``val()`` and ``export()`` work straight away.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Official COCO-only checkpoints, by size.
OFFICIAL = {size: f"dfine_{size}_coco.pth" for size in "nsmlx"}
RELEASE_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0"

#: Cached by the reference for a fixed 640 input; recomputed here instead.
DROPPED = ("decoder.anchors", "decoder.valid_mask")

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def official_state_dict(path: Path) -> dict:
    """Unwrap whichever container the release used (ema / model / bare)."""
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("ema", "model", "state_dict"):
        if isinstance(blob, dict) and key in blob:
            blob = blob[key]
            if isinstance(blob, dict) and "module" in blob:
                blob = blob["module"]
            break
    if not isinstance(blob, dict):
        raise SystemExit(f"{path} does not contain a state_dict")
    return {k: v for k, v in blob.items() if hasattr(v, "shape") and k not in DROPPED}


def resolve_weights(args) -> Path:
    from easydetect.downloads import cache_dir, download

    if args.weights:
        if str(args.weights).startswith(("http://", "https://")):
            name = str(args.weights).rsplit("/", 1)[-1]
            return download(str(args.weights), cache_dir() / "official" / name)
        return Path(args.weights)
    name = OFFICIAL[args.size]
    return download(f"{RELEASE_URL}/{name}", cache_dir() / "official" / name)


def convert(args: argparse.Namespace) -> int:
    import torch

    from easydetect.nn import EDNet

    weights = resolve_weights(args)
    state = official_state_dict(weights)
    names = _read_names(args.names)

    net = EDNet(args.size, num_classes=len(names), pretrained_backbone=False)
    missing, unexpected = net.load_state_dict(state, strict=not args.allow_partial)
    if missing or unexpected:
        print(f"warning: {len(missing)} missing, {len(unexpected)} unexpected tensors")
        for key in list(missing)[: args.report] + list(unexpected)[: args.report]:
            print(f"    {key}")
    else:
        print(f"loaded all {len(state)} tensors from {weights.name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "variant": args.size,
            "num_classes": len(names),
            "names": dict(enumerate(names)),
            "epoch": -1,
            "imgsz": args.imgsz,
            "source": f"Peterande/D-FINE {weights.name} (Apache-2.0)",
        },
        out,
    )
    print(f"wrote {out}")
    return 0


def _read_names(spec: str | None) -> list[str]:
    if not spec:
        return COCO_NAMES
    path = Path(spec)
    if path.suffix == ".json":
        table = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(table, dict):
            return [table[k] for k in sorted(table, key=int)]
        return list(table)
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="s", choices=tuple(OFFICIAL))
    parser.add_argument("--weights", help="official .pth path or URL (default: download it)")
    parser.add_argument("--out", default="ed-converted.pt")
    parser.add_argument("--names", help="labels.txt or names.json (default: COCO 80)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--allow-partial", action="store_true", help="load what matches instead of failing"
    )
    parser.add_argument("--report", type=int, default=10, help="print N example keys on mismatch")
    return convert(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
