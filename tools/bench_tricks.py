#!/usr/bin/env python3
# Apache-2.0
"""The lab's 수업용 training on three small datasets, with and without a training option.

    pip install timm                       # only for the "-dino" variant
    python tools/bench_tricks.py --out runs/tricks --variants vfl,mal,vfl-ema200,vfl-dino

Variants (combine with "-"): ``vfl`` or ``mal`` (the class loss), ``-ema<N>``
(the weight average ramps up over N steps instead of 1,000), ``-dino`` (a
frozen DINOv2 ViT-S/14 as a training-only feature teacher). Each runs ed-n as
the 수업용 preset does — 320 px, batch 8, 100 queries, 30 denoising queries,
no zoom/crop augmentation, ``freeze="auto"``, 1,000 steps (800 when the
backbone trains) — on signature, medical-pills and 100 pictures of
african-wildlife, once per seed, and scores the best weights on the val split
with ``bench_finetune.py``'s evaluator. ``<out>/results.json`` keeps every
run (a run done is skipped on a rerun); ``<out>/results.md`` is the table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _options(variant: str) -> dict:
    parts = variant.split("-")
    options = {"cls_loss": parts[0]}
    for part in parts[1:]:
        if part.startswith("ema"):
            options["ema_warmups"] = int(part[3:])
        elif part == "dino":
            from easydetect.feature_teacher import FeatureTeacher

            options["extra_loss"] = FeatureTeacher()
        else:
            raise SystemExit(f"unknown variant part {part!r} in {variant!r}")
    return options


def _score(data: Path, best: Path, device: str) -> tuple[float, float]:
    import numpy as np
    from compare_yolo import EdRunner, evaluate

    from easydetect.data.dataset import DetDataset

    ds = DetDataset(data, "val", 64, augment=False)
    truth = []
    for f in ds.files:
        lab = ds._load_labels(f)
        cx, cy, bw, bh = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
        truth.append((np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1),
                      lab[:, 0].astype(int)))
    runner = EdRunner(str(best), device)
    runner.multi_label = True
    m50, m, _ = evaluate("ed", runner, ds.files, truth, ds.nc)
    return round(m, 4), round(m50, 4)


def table(results: dict, variants: list[str]) -> str:
    sets = sorted({k.split("/")[0] for k in results})
    rows = ["| dataset | " + " | ".join(variants) + " |",
            "| --- |" + " --- |" * len(variants)]
    for name in sets:
        cells = []
        for v in variants:
            runs = [r["map"] for k, r in results.items() if k.startswith(f"{name}/{v}/")]
            cells.append(f"{statistics.mean(runs):.3f} ± {statistics.pstdev(runs):.3f} "
                         f"({len(runs)})" if runs else "")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n\nmAP50-95 on each val split, mean ± spread (runs).\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="~/datasets/bench", help="where the datasets go")
    parser.add_argument("--out", default="runs/tricks")
    parser.add_argument("--variants", default="vfl,mal,vfl-ema200,vfl-dino")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default=None, help="training device (default: a GPU if any)")
    parser.add_argument("--score-device", default="CPU", help="OpenVINO device for scoring")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, str(HERE))
    import yaml
    from bench_finetune import fetch, subset
    from rf100 import classroom_epochs, coco_knows

    from easydetect import Detector
    from easydetect.data.dataset import DetDataset

    root, out = Path(args.root).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    record = out / "results.json"
    results = json.loads(record.read_text()) if record.exists() else {}
    variants = args.variants.split(",")
    sets = [("signature", fetch("signature", root)),
            ("medical-pills", fetch("medical-pills", root)),
            ("african-wildlife-100", subset(fetch("african-wildlife", root), 100))]
    for seed in (int(s) for s in args.seeds.split(",")):
        for name, data in sets:
            names = yaml.safe_load(Path(data).read_text())["names"]
            names = names if isinstance(names, dict) else dict(enumerate(names))
            pictures = len(DetDataset(data, "train", 64, augment=False).files)
            epochs = classroom_epochs(pictures, coco_knows(names))
            for variant in variants:
                key = f"{name}/{variant}/seed{seed}"
                if key in results:
                    continue
                started = time.time()
                best = Detector("ed-n", verbose=True).train(
                    data=str(data), epochs=epochs, imgsz=320, batch=8, queries=100,
                    denoising=30, freeze="auto", augment=False, patience=epochs, seed=seed,
                    workers=args.workers, device=args.device, project=str(out / "runs"),
                    name=key.replace("/", "-"), **_options(variant))
                m, m50 = _score(data, best, args.score_device)
                results[key] = {"map": m, "map50": m50, "epochs": epochs, "pictures": pictures,
                                "minutes": round((time.time() - started) / 60, 1)}
                record.write_text(json.dumps(results, indent=2))
                (out / "results.md").write_text(table(results, variants))
                print("RESULT", key, results[key], flush=True)
    print("\n" + table(results, variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
