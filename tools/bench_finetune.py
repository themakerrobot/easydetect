#!/usr/bin/env python3
# Apache-2.0
"""Fine-tune dfine-s and a YOLO on the same small datasets; score both the same way.

    pip install ultralytics          # only for this comparison; AGPL-3.0
    python tools/bench_finetune.py --root ~/datasets/bench --out runs/bench

COCO scores say little about the question most people have: with a few
hundred pictures of my own, which model ends up better? This answers it on
public datasets Ultralytics publishes in YOLO format (downloaded once into
``--root``):

=================  =======  ============  ========================
dataset            classes  train / val   what
=================  =======  ============  ========================
african-wildlife   4        1052 / 225    animals, large objects
construction-ppe   11       1132 / 143    small parts of people
brain-tumor        2        893 / 223     MRI, one subtle object
signature          1        143 / 35      documents, few pictures
medical-pills      1        92 / 23       many small, similar objects
=================  =======  ============  ========================

and, with ``--low-data N``, on african-wildlife cut to its first N training
pictures (the same val split).

Each model is fine-tuned from its COCO weights by its own trainer, with its
own defaults (augmentation, learning rate, schedule), at the same image size,
batch and number of epochs, with no early stop. Then both are exported (ONNX)
and scored by ``compare_yolo.py`` on the val split: one evaluator
(easydetect's COCO-style mAP, conf 0.001, up to 300 boxes), each model with
its own preprocessing, on OpenVINO on ``--device`` (CPU by default), which
also gives the model's median inference time a picture.

Ultralytics runs as its own ``yolo`` command, in a separate process; nothing
of it is imported here. Results land in ``<out>/results.json`` and as a
Markdown table in ``<out>/results.md``; a dataset already done is skipped,
so the script can be rerun after an interruption.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import yaml

ASSETS = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
DATASETS = {
    "african-wildlife": ["buffalo", "elephant", "rhino", "zebra"],
    "construction-ppe": ["helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
                         "no_helmet", "no_goggle", "no_gloves", "no_boots"],
    "brain-tumor": ["negative", "positive"],
    "signature": ["signature"],
    "medical-pills": ["pill"],
}


def fetch(name: str, root: Path) -> Path:
    """The dataset's data.yaml, downloading and unpacking it the first time.

    Each zip is unpacked into ``<root>/<name>/`` and searched there, whatever
    its layout: ``images/train`` + ``images/val``, the same one folder down,
    or ``train/images`` + ``valid/images`` (as Roboflow exports it).
    """
    folder = root / name
    archive = root / f"{name}.zip"
    if not folder.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            print(f"downloading {name}…", flush=True)
            urllib.request.urlretrieve(f"{ASSETS}/{name}.zip", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
    splits = {}
    for split, aliases in (("train", ("train",)), ("val", ("val", "valid", "validation"))):
        found = [p for a in aliases for p in [*folder.glob(f"**/images/{a}"),
                                              *folder.glob(f"**/{a}/images")] if p.is_dir()]
        if not found:
            raise SystemExit(f"{folder}: no {split} images found inside (looked for "
                             f"images/{split} and {split}/images); unpack {archive} there by hand")
        splits[split] = min(found, key=lambda p: len(p.parts))
    # the folder above images/<split> or <split>/images: labels sit beside it
    common = splits["train"].parent.parent
    data = folder / "bench.yaml"
    data.write_text(yaml.safe_dump({"path": str(common.resolve()),
                                    "train": str(splits["train"].relative_to(common)),
                                    "val": str(splits["val"].relative_to(common)),
                                    "names": dict(enumerate(DATASETS[name]))}))
    return data


def subset(data: Path, n: int) -> Path:
    """A copy of a dataset with only its first ``n`` training pictures (links, not copies)."""
    from easydetect.data.labels import label_path

    cfg = yaml.safe_load(data.read_text())
    src = Path(cfg["path"])
    dst = src.parent / f"{src.name}-{n}"
    for split, keep in (("train", n), ("val", None)):
        pics = sorted(p for p in (src / cfg[split]).iterdir() if p.is_file())[:keep]
        for pic in pics:
            label = Path(label_path(pic))
            for a in (pic, label):
                b = dst / a.relative_to(src)
                b.parent.mkdir(parents=True, exist_ok=True)
                if a.exists() and not b.exists():
                    b.symlink_to(a.resolve())
    out = dst.parent / f"{dst.name}.yaml"
    out.write_text(yaml.safe_dump(dict(cfg, path=str(dst.resolve()))))
    return out


def train_dfine(data: Path, out: Path, args) -> tuple[Path, float]:
    from easydetect import Detector

    started = time.time()
    best = Detector(args.dfine, verbose=False).train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.epochs, project=str(out), name=args.dfine, workers=args.workers)
    seconds = time.time() - started
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — a CPU-only machine has nothing to free
        pass
    return Path(best), seconds


def train_yolo(data: Path, out: Path, args) -> tuple[Path, float]:
    name = Path(args.yolo).stem
    started = time.time()
    subprocess.run(["yolo", "detect", "train", f"model={args.yolo}", f"data={data}",
                    f"epochs={args.epochs}", f"imgsz={args.imgsz}", f"batch={args.batch}",
                    f"patience={args.epochs}", f"project={out.resolve()}", f"name={name}",
                    f"workers={args.workers}", "exist_ok=True", "plots=False"], check=True)
    seconds = time.time() - started
    best = out / name / "weights" / "best.pt"
    subprocess.run(["yolo", "export", f"model={best}", "format=onnx", f"imgsz={args.imgsz}"],
                   check=True)
    return best.with_suffix(".onnx"), seconds


def score(data: Path, yolo_onnx: Path, dfine_pt: Path, args) -> dict:
    """compare_yolo.py's runners and evaluator on the val split."""
    import cv2  # noqa: F401 — compare_yolo needs it

    sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np
    from compare_yolo import Dfine, Yolo, evaluate

    from easydetect.data.dataset import DetDataset

    ds = DetDataset(data, "val", 64, augment=False)
    truth = []
    for f in ds.files:
        lab = ds._load_labels(f)
        cx, cy, bw, bh = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
        truth.append((np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1),
                      lab[:, 0].astype(int)))
    out = {"val_pictures": len(ds.files), "val_boxes": int(sum(len(t[1]) for t in truth))}
    for key, runner in (("yolo", Yolo(str(yolo_onnx), args.device, args.imgsz)),
                        ("dfine", Dfine(str(dfine_pt), args.device))):
        m50, m, ms = evaluate(key, runner, ds.files, truth, ds.nc)
        out[key] = {"map50": round(m50, 4), "map": round(m, 4), "ms": round(ms, 1)}
    return out


def _bold_if_higher(a: float, b: float) -> str:
    return f"**{a:.3f}**" if a > b else f"{a:.3f}"


def table(results: dict, args) -> str:
    yolo, dfine = Path(args.yolo).stem, args.dfine
    rows = [f"| dataset | train pictures | {yolo} mAP50-95 | {dfine} mAP50-95 | "
            f"{yolo} mAP50 | {dfine} mAP50 | {yolo} ms | {dfine} ms |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        y, d = r["yolo"], r["dfine"]
        best_y, best_d = _bold_if_higher(y["map"], d["map"]), _bold_if_higher(d["map"], y["map"])
        rows.append(f"| {name} | {r['train_pictures']} | {best_y} | {best_d} | "
                    f"{y['map50']:.3f} | {d['map50']:.3f} | {y['ms']} | {d['ms']} |")
    return "\n".join(rows) + (
        f"\n\n{args.epochs} epochs, {args.imgsz} px, batch {args.batch}, each trainer's own "
        f"defaults; scored by one evaluator, inference times on OpenVINO {args.device}.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="datasets/bench", help="where the datasets go")
    parser.add_argument("--out", default="runs/bench")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--low-data", type=int, default=100,
                        help="also african-wildlife cut to this many training pictures (0: no)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--yolo", default="yolo26s.pt")
    parser.add_argument("--dfine", default="dfine-s")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for scoring")
    args = parser.parse_args()
    if shutil.which("yolo") is None:
        raise SystemExit("the yolo command is missing: pip install ultralytics")

    root, out = Path(args.root).expanduser(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = out / "results.json"
    results = json.loads(record.read_text()) if record.exists() else {}

    plan = [(name, fetch(name, root)) for name in args.datasets]
    if args.low_data:
        plan.append((f"african-wildlife-{args.low_data}",
                     subset(fetch("african-wildlife", root), args.low_data)))
    for name, data in plan:
        if name in results:
            print(f"{name}: done already", flush=True)
            continue
        cfg = yaml.safe_load(data.read_text())
        train_pictures = sum(1 for p in (Path(cfg["path"]) / cfg["train"]).iterdir() if p.is_file())
        print(f"=== {name} ({train_pictures} training pictures)", flush=True)
        dfine_pt, dfine_s = train_dfine(data, out / name, args)
        yolo_onnx, yolo_s = train_yolo(data, out / name, args)
        r = score(data, yolo_onnx, dfine_pt, args)
        r.update(train_pictures=train_pictures,
                 train_minutes={"dfine": round(dfine_s / 60, 1), "yolo": round(yolo_s / 60, 1)})
        results[name] = r
        record.write_text(json.dumps(results, indent=2))
        print(json.dumps(r), flush=True)

    text = table(results, args)
    (out / "results.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
