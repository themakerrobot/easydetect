#!/usr/bin/env python3
# Apache-2.0
"""Turn an official D-FINE (or DEIM) COCO checkpoint into an easydetect one.

    python tools/convert_upstream.py --size s --out weights/ed-s.pt
    python tools/convert_upstream.py --size s --source deim --out weights/ed-s-deim.pt

DEIM (https://github.com/Intellindust-AI-Lab/DEIM, Apache-2.0) trained the
same network a better way (COCO: +0.2 to +0.7 mAP), with SiLU in the
decoder's feed-forward and box heads where the original has ReLU; the
checkpoint records it (``decoder_act``). Its COCO-only checkpoints sit on
Google Drive: downloading them needs ``pip install gdown``.

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
#: DEIM's COCO-only checkpoints (its model zoo; the Objects365 one is not used)
DEIM_DRIVE = {"n": "1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e", "s": "1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC",
              "m": "18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8", "l": "1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr",
              "x": "1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j"}

#: per source: (decoder activation, credit line)
SOURCES = {"dfine": ("relu", "Peterande/D-FINE {name} (Apache-2.0)"),
           "deim": ("silu", "Intellindust-AI-Lab/DEIM {name} (Apache-2.0)")}

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
    # DEIM trains every batch norm (the original freezes l's and x's backbone
    # ones): the same numbers, with a step counter the frozen ones lack
    return {k: v for k, v in blob.items() if hasattr(v, "shape") and k not in DROPPED
            and not k.endswith("num_batches_tracked")}


def resolve_weights(args) -> Path:
    from easydetect.downloads import cache_dir, download

    if args.weights:
        if str(args.weights).startswith(("http://", "https://")):
            name = str(args.weights).rsplit("/", 1)[-1]
            return download(str(args.weights), cache_dir() / "official" / name)
        return Path(args.weights)
    if args.source == "deim":
        dest = cache_dir() / "official" / f"deim_dfine_{args.size}_coco.pth"
        if not (dest.exists() and dest.stat().st_size):
            try:
                import gdown
            except ImportError:
                raise SystemExit("the DEIM checkpoints are on Google Drive: pip install gdown") \
                    from None
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not gdown.download(id=DEIM_DRIVE[args.size], output=str(dest), quiet=False):
                raise SystemExit(f"could not fetch DEIM's {args.size} checkpoint from Google Drive")
        return dest
    name = OFFICIAL[args.size]
    return download(f"{RELEASE_URL}/{name}", cache_dir() / "official" / name)


def convert(args: argparse.Namespace) -> int:
    import torch

    from easydetect.nn import EDNet

    weights = resolve_weights(args)
    state = official_state_dict(weights)
    names = _read_names(args.names)

    act, credit = SOURCES[args.source]
    net = EDNet(args.size, num_classes=len(names), pretrained_backbone=False, decoder_act=act)
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
            "decoder_act": act,
            "names": dict(enumerate(names)),
            "epoch": -1,
            "imgsz": args.imgsz,
            "source": credit.format(name=weights.name),
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
    parser.add_argument("--source", default="dfine", choices=tuple(SOURCES),
                        help="whose COCO weights (deim: the same network, trained better)")
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
