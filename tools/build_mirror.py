#!/usr/bin/env python3
# Apache-2.0
"""Build the weight-mirror layout that ``Detector("ed-s")`` downloads from.

    python tools/build_mirror.py --out mirror              # all five sizes
    python tools/build_mirror.py --out mirror --sizes s    # just one

For each size this downloads the Apache-2.0 COCO checkpoint from the upstream
release, loads it (strict — this package's layout matches), and writes::

    mirror/ed-s/ed-s.pt      torch weights, for train/val/export
    mirror/ed-s/ed-s.xml     OpenVINO IR, what predict downloads on OpenVINO
    mirror/ed-s/ed-s.bin
    mirror/ed-s/ed-s.onnx    the same network for ONNX Runtime (the light install)
    mirror/ed-s/labels.txt      COCO class names, one per line

The GitHub Actions workflow ``mirror.yml`` runs this and uploads the result.

Upload that tree to the mirror named by ``$EASYDETECT_ASSETS_URL`` (by default the
Hugging Face repo in easydetect/downloads.py), keeping the directory names:

    hf upload leeyunjai/easydetect mirror . --repo-type=model

(the `hf` command comes from ``pip install -U "huggingface_hub[cli]"``; it used
to be called ``huggingface-cli``)

Everything runs on CPU in a few minutes; there is no training involved.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert_upstream  # noqa: E402  (same directory)


def build(size: str, out_root: Path, imgsz: int, half: bool, onnx: bool) -> Path:
    from easydetect import Detector

    name = f"ed-{size}"
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = out / f"{name}.pt"
    convert_upstream.main(["--size", size, "--out", str(checkpoint), "--imgsz", str(imgsz)])

    model = Detector(str(checkpoint), verbose=False)
    xml = model.export(format="openvino", imgsz=imgsz, half=half, out_dir=out)
    if not onnx:     # the export writes it on the way to the IR
        (out / f"{name}.onnx").unlink(missing_ok=True)
    print(f"{name}: {', '.join(sorted(p.name for p in out.iterdir()))}")
    return xml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="mirror", help="output directory")
    parser.add_argument(
        "--sizes", nargs="+", default=list(convert_upstream.OFFICIAL), choices=tuple("nsmlx")
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 IR (smaller, same accuracy)")
    parser.add_argument("--no-onnx", action="store_true",
                        help="leave out the .onnx that ONNX Runtime downloads")
    args = parser.parse_args(argv)

    out_root = Path(args.out)
    for size in args.sizes:
        build(size, out_root, args.imgsz, args.half, not args.no_onnx)
    # the repo's front page travels with the weights
    shutil.copyfile(Path(__file__).with_name("hub_README.md"), out_root / "README.md")
    print(f"\nmirror ready at {out_root}/ — upload it keeping these directory names:")
    print('  pip install -U "huggingface_hub[cli]" && hf auth login')
    print(f"  hf upload leeyunjai/easydetect {out_root} . --repo-type=model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
