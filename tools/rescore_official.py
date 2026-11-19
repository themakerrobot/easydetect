#!/usr/bin/env python3
# Apache-2.0
"""Every COCO model long_run.py scored before, scored again the official way.

    python tools/rescore_official.py --root ~/longrun --coco ~/datasets/coco

Older entries in ``<root>/coco/scores.json`` and ``<root>/pi-kit-*/scores.json``
hold only the Pibo way (one class a query, up to 100 boxes). Each one whose
file is still there is scored again with ``pi_kit.score``, which adds
``map_official`` (the 300 best query x class pairs, as COCO tables count),
and the record is written back. YOLO entries are left as they are: their
evaluation already keeps up to 300 boxes a picture. The table of every
official number lands in ``<root>/official.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def records(root: Path) -> list[tuple[Path, Path]]:
    """(scores.json, folder its files are relative to)"""
    found = []
    coco = root / "coco" / "scores.json"
    if coco.exists():
        found.append((coco, root / "pi-kit-coco"))
    found += [(r, r.parent) for r in sorted(root.glob("pi-kit-*/scores.json"))]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="~/longrun")
    parser.add_argument("--coco", default="~/datasets/coco")
    parser.add_argument("--limit", type=int, help="val pictures (a quick try)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, str(HERE))
    from pi_kit import score

    root = Path(args.root).expanduser()
    coco_yaml = str(Path(args.coco).expanduser() / "data.yaml")
    lines = ["| record | model | COCO mAP50-95 (official) |", "| --- | --- | --- |"]
    for record, base in records(root):
        scores = json.loads(record.read_text())
        for key, entry in scores.items():
            if "map_official" not in entry and entry.get("file"):
                path = base / entry["file"]
                if not path.exists():
                    print(f"{key}: {path} is gone, skipped", flush=True)
                    continue
                backend = "onnxruntime" if path.suffix == ".onnx" else None
                entry.update(score(path, coco_yaml, args.limit, backend=backend))
                if not args.limit:
                    record.write_text(json.dumps(scores, indent=2))
                print(f"{key}: {entry['map_official']:.3f}", flush=True)
            value = entry.get("map_official", entry.get("map") if not entry.get("file") else None)
            if value is not None:
                lines.append(f"| {record.parent.name} | {key} | {value:.3f} |")
    (root / "official.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
