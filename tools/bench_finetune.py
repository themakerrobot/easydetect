#!/usr/bin/env python3
# Apache-2.0
"""Fine-tune ed-s and a YOLO on the same small datasets; score both the same way.

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

and, with ``--low-data N``, on african-wildlife cut to N training pictures
drawn at random (the same val split).

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
import faulthandler
import json
import os
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


# a log file is block-buffered: without this, what each step prints shows up
# minutes later, out of order with the yolo commands' own output
UNBUFFERED = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")


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
    """A copy of a dataset with ``n`` training pictures drawn at random (seed 0;
    links, not copies). Not the first ``n``: files are often named by class,
    and the first 100 of african-wildlife are all buffalo."""
    import random

    from easydetect.data.labels import label_path

    cfg = yaml.safe_load(data.read_text())
    src = Path(cfg["path"])
    dst = src.parent / f"{src.name}-{n}-random"
    for split, keep in (("train", n), ("val", None)):
        pics = sorted(p for p in (src / cfg[split]).iterdir() if p.is_file())
        if keep is not None:
            pics = sorted(random.Random(0).sample(pics, min(keep, len(pics))))
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


def train_ed(data: Path, out: Path, args) -> tuple[Path, float]:
    for run in sorted(out.glob(f"{args.label}*")):   # finished before an interruption
        summary = run / "summary.json"
        if summary.exists() and (run / "weights" / "best.pt").exists():
            done = json.loads(summary.read_text())
            if done.get("epochs_run") == args.epochs:
                print(f"{run}: trained already, reusing it", flush=True)
                return run / "weights" / "best.pt", float(done.get("wall_seconds") or 0)
    # in a process of its own, as the yolo command is: a crash in what training
    # leaves behind (loader workers, CUDA) then takes only that process down
    started = time.time()
    extra = {k: v for k, v in (("queries", args.queries), ("denoising", args.denoising),
                               ("freeze", args.freeze)) if v is not None}
    if args.no_augment:
        extra["augment"] = False
    if args.seed:
        extra["seed"] = args.seed
    if args.clean_bn:
        extra["clean_bn"] = args.clean_bn
    if args.lr0:
        extra["lr0"] = args.lr0
    code = ("import json, sys; from easydetect import Detector; "
            "print('BEST', Detector(sys.argv[1], verbose=False).train(data=sys.argv[2], "
            "epochs=int(sys.argv[3]), imgsz=int(sys.argv[4]), batch=int(sys.argv[5]), "
            "patience=int(sys.argv[3]), project=sys.argv[6], name=sys.argv[9], "
            "workers=int(sys.argv[7]), **json.loads(sys.argv[8])))")
    best = None
    env = dict(UNBUFFERED, **({"EASYDETECT_NEW_ROWS": args.new_rows} if args.new_rows else {}))
    with subprocess.Popen([sys.executable, "-c", code, args.ed, str(data), str(args.epochs),
                           str(args.imgsz), str(args.batch), str(out), str(args.workers),
                           json.dumps(extra), args.label],
                          env=env, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:                    # passed on as it comes
            print(line, end="", flush=True)
            if line.startswith("BEST "):
                best = Path(line[5:].strip())
    if proc.returncode or best is None:
        raise SystemExit(f"ed training failed (exit {proc.returncode})")
    return best, time.time() - started


EXIT_GRACE = 300  # seconds a finished yolo training may take to exit


def train_yolo(data: Path, out: Path, args) -> tuple[Path, float]:
    name = Path(args.yolo).stem
    best = out / name / "weights" / "best.pt"
    log = out / name / "results.csv"
    if best.exists() and log.exists() and len(log.read_text().splitlines()) > args.epochs:
        print(f"{out / name}: trained already, reusing it", flush=True)
        return _export_yolo(best, args), 0.0   # its time was not recorded
    started = time.time()
    proc = subprocess.Popen(["yolo", "detect", "train", f"model={args.yolo}", f"data={data}",
                             f"epochs={args.epochs}", f"imgsz={args.imgsz}",
                             f"batch={args.batch}", f"patience={args.epochs}",
                             f"project={out.resolve()}", f"name={name}",
                             f"workers={args.workers}", f"seed={args.seed}", "exist_ok=True",
                             "plots=False"],
                            env=UNBUFFERED)
    finished = None
    while proc.poll() is None:
        time.sleep(10)
        complete = log.exists() and len(log.read_text().splitlines()) > args.epochs
        if complete and finished is None:
            finished = time.time()
        # it has hung on its way out, with the training done and saved:
        # give its final validation time, then stop waiting for it
        if finished is not None and time.time() - finished > EXIT_GRACE:
            print(f"yolo train: done, but still running {EXIT_GRACE} s later; stopping it",
                  flush=True)
            proc.kill()
            proc.wait()
            break
    if not (best.exists() and log.exists() and len(log.read_text().splitlines()) > args.epochs):
        raise SystemExit(f"yolo training failed (exit {proc.returncode})")
    seconds = (finished or time.time()) - started
    return _export_yolo(best, args), seconds


def _export_yolo(best: Path, args) -> Path:
    onnx = best.with_suffix(".onnx")
    if not onnx.exists():
        # a timeout: an export that hangs (it has, once) fails loudly instead
        subprocess.run(["yolo", "export", f"model={best}", "format=onnx",
                        f"imgsz={args.imgsz}"], check=True, env=UNBUFFERED, timeout=900)
    return onnx


def score(data: Path, yolo_onnx: Path, ed_pt: Path, args) -> dict:
    """compare_yolo.py's runners and evaluator on the val split."""
    import cv2  # noqa: F401 — compare_yolo needs it

    sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np
    from compare_yolo import EdRunner, Yolo, evaluate

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
                        ("ed", EdRunner(str(ed_pt), args.device))):
        # as each project scores itself (a box may count for several classes),
        # then as predict hands them out (one class a box)
        runner.multi_label = True
        m50, m, ms = evaluate(key, runner, ds.files, truth, ds.nc)
        runner.multi_label = False
        s50, sm, _ = evaluate(key, runner, ds.files, truth, ds.nc)
        out[key] = {"map50": round(m50, 4), "map": round(m, 4), "ms": round(ms, 1),
                    "map50_one_class": round(s50, 4), "map_one_class": round(sm, 4)}
    return out


def _bold_if_higher(a: float, b: float) -> str:
    return f"**{a:.3f}**" if a > b else f"{a:.3f}"


def table(results: dict, args) -> str:
    yolo, ed = Path(args.yolo).stem, args.label
    rows = [f"| dataset | train pictures | {yolo} mAP50-95 | {ed} mAP50-95 | "
            f"{yolo} one class a box | {ed} one class a box | "
            f"{yolo} mAP50 | {ed} mAP50 | {yolo} ms | {ed} ms |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        y, d = r["yolo"], r["ed"]
        one_y, one_d = y.get("map_one_class"), d.get("map_one_class")
        ones = (f"{_bold_if_higher(one_y, one_d)} | {_bold_if_higher(one_d, one_y)}"
                if one_y is not None and one_d is not None else "| ")
        rows.append(f"| {name} | {r['train_pictures']} | {_bold_if_higher(y['map'], d['map'])} | "
                    f"{_bold_if_higher(d['map'], y['map'])} | {ones} | "
                    f"{y['map50']:.3f} | {d['map50']:.3f} | {y['ms']} | {d['ms']} |")
    return "\n".join(rows) + (
        f"\n\n{args.epochs} epochs, {args.imgsz} px, batch {args.batch}, each trainer's own "
        "defaults; scored by one evaluator (COCO's mAP, up to 100 boxes a picture). mAP "
        "counts a box for every class it scores (as each project scores itself); \"one "
        "class a box\" is what predict hands out. Inference times on OpenVINO "
        f"{args.device}.\n")


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
    parser.add_argument("--ed", default="ed-s",
                        help="a model name, or a .pt to start from (a COCO fine-tune at 320)")
    parser.add_argument("--label", help="the ed column's name (default: the model name, "
                                        "or the run folder of a .pt)")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for scoring")
    parser.add_argument("--queries", type=int, help="ed decoder queries (of 300)")
    parser.add_argument("--denoising", type=int, help="ed denoising queries (100)")
    parser.add_argument("--freeze", help="ed parts not trained (backbone)")
    parser.add_argument("--seed", type=int, default=0, help="both trainers' seed")
    parser.add_argument("--lr0", type=float, help="ed learning rate (default: from the batch)")
    parser.add_argument("--new-rows", choices=["auto", "mean", "fresh"],
                        help="classes COCO does not know: from the mean of its rows, fresh, "
                             "or the mean only when it knows none (auto, the default)")
    parser.add_argument("--clean-bn", choices=["auto", "keep", "adapt"],
                        help="the original last plain-picture epochs: keep the batch-norm "
                             "statistics (0.7.5) or re-estimate them (before)")
    parser.add_argument("--no-augment", action="store_true",
                        help="ed without zoom, crop and colour jitter (the lab's classroom "
                             "preset); YOLO keeps its own")
    parser.add_argument("--rescore", action="store_true",
                        help="score finished datasets again (their trained models are reused)")
    args = parser.parse_args()
    if not args.label:
        start = Path(args.ed).expanduser()
        args.label = start.parent.parent.name if start.suffix == ".pt" else args.ed
    sys.stdout.reconfigure(line_buffering=True)
    faulthandler.enable()  # a crash prints where it happened
    if shutil.which("yolo") is None:
        raise SystemExit("the yolo command is missing: pip install ultralytics")

    root, out = Path(args.root).expanduser(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = out / "results.json"
    results = json.loads(record.read_text()) if record.exists() else {}

    plan = [(name, fetch(name, root)) for name in args.datasets]
    for stale in [k for k in results if k == f"african-wildlife-{args.low_data}"]:
        # before 0.7.1 this was the first pictures, all of one class
        print(f"{stale}: dropped (it was one class only); now african-wildlife-"
              f"{args.low_data}-random", flush=True)
        del results[stale]
    if args.low_data:
        plan.append((f"african-wildlife-{args.low_data}-random",
                     subset(fetch("african-wildlife", root), args.low_data)))
    for name, data in plan:
        if name in results and not args.rescore:
            print(f"{name}: done already", flush=True)
            continue
        cfg = yaml.safe_load(data.read_text())
        train_pictures = sum(1 for p in (Path(cfg["path"]) / cfg["train"]).iterdir() if p.is_file())
        print(f"=== {name} ({train_pictures} training pictures)", flush=True)
        ed_pt, ed_s = train_ed(data, out / name, args)
        yolo_onnx, yolo_s = train_yolo(data, out / name, args)
        r = score(data, yolo_onnx, ed_pt, args)
        minutes = {"ed": round(ed_s / 60, 1), "yolo": round(yolo_s / 60, 1)}
        if name in results:            # rescored: the training times are the first run's
            minutes = results[name].get("train_minutes", minutes)
        r.update(train_pictures=train_pictures, train_minutes=minutes)
        results[name] = r
        record.write_text(json.dumps(results, indent=2))
        print(json.dumps(r), flush=True)

    text = table(results, args)
    (out / "results.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
