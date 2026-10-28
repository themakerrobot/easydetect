#!/usr/bin/env python3
# Apache-2.0
"""The lab's 수업용 training on Roboflow 100 datasets, against YOLO26n.

    pip install ultralytics pyarrow       # only for this comparison
    python tools/rf100.py --root ~/datasets/rf100 --out runs/rf100-seed0 --seed 0

Three small datasets (signature, medical-pills, african-wildlife) set the
classroom defaults; this checks them on 26 others from RF100 (Ciaglia et al.,
2022), across its seven domains: aerial, video games, microscopy, underwater,
documents, X-ray and thermal, and the everyday. They come from the public
Hugging Face copies (``Francesco/<name>``, CC BY 4.0 mostly; each dataset's
card has its licence), so no Roboflow key is needed. They are used to score
models here and are not redistributed.

Each dataset becomes what a class brings to the lab: at most ``--train``
training pictures drawn at random (seed 0), the ``validation`` split (at most
``--val``) for picking the best epoch, and the ``test`` split (at most
``--test``) to score on. RF100's parent category (class 0, ``aquarium`` over
``fish``, ``shark``…) has no boxes of its own and is dropped.

ed-n trains as the 수업용 preset does: 320 px, batch 8, 100 queries and 30
denoising queries, no zoom/crop augmentation, ``freeze="auto"``, and as many
epochs as make 1,000 steps (800 when no class is one COCO knows, as the
backbone then trains too), 35 to 100. YOLO26n trains the same epochs at the
same size and batch with its own defaults. Both are scored by
``bench_finetune.py``'s evaluator on the test split. Results land in
``<out>/results.json`` (a dataset done is skipped on a rerun) and
``<out>/results.md``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import shutil
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import yaml

HERE = Path(__file__).resolve().parent
HUB = "https://huggingface.co/api/datasets/Francesco"
FILES = "https://huggingface.co/datasets/Francesco/{name}/resolve/main/{path}"

#: name on the hub → RF100 domain
SETS = {
    "aerial-pool": "aerial", "aerial-cows": "aerial", "solar-panels-taxvb": "aerial",
    "weed-crop-aerial": "aerial",
    "apex-videogame": "videogame", "csgo-videogame": "videogame",
    "bccd-ouzjz": "microscopic", "parasites-1s07h": "microscopic",
    "aquarium-qlnqy": "underwater", "underwater-pipes-4ng4t": "underwater",
    "coral-lwptl": "underwater",
    "document-parts": "document", "signatures-xc8up": "document",
    "thermal-dogs-and-people-x6ejw": "electromagnetic",
    "x-ray-rheumatology": "electromagnetic", "bone-fracture-7fylg": "electromagnetic",
    "pills-sxdht": "real world", "soda-bottles": "real world",
    "chess-pieces-mjzgj": "real world", "road-signs-6ih4y": "real world",
    "mask-wearing-608pr": "real world", "construction-safety-gsnvb": "real world",
    "furniture-ngpea": "real world", "vehicles-q0x2v": "real world",
    "animals-ij5d2": "real world", "apples-fvpl5": "real world",
}

#: the lab's 수업용 epochs: about this many steps of a batch of 8
CLASSROOM_STEPS, CLASSROOM_STEPS_UNFROZEN = 1000, 800


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "easydetect-rf100"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def _category_names(name: str) -> dict[int, str]:
    card = json.loads(_get(f"{HUB}/{name}"))["cardData"]["dataset_info"]
    for feature in card["features"]:
        if feature["name"] == "objects":
            for field in feature["sequence"]:
                if field["name"] == "category":
                    names = field["dtype"]["class_label"]["names"]
                    return {int(k): str(v) for k, v in names.items()}
    raise SystemExit(f"{name}: no category names on its card")


def prepare(name: str, root: Path, caps: dict[str, int]) -> tuple[Path, Path, dict]:
    """``(train.yaml, score.yaml, facts)`` for one dataset, built the first time."""
    folder = root / name
    train_yaml, score_yaml, facts_file = (folder / "train.yaml", folder / "score.yaml",
                                          folder / "facts.json")
    if facts_file.exists():
        return train_yaml, score_yaml, json.loads(facts_file.read_text())
    import pyarrow.parquet as pq

    names = _category_names(name)
    tree = json.loads(_get(f"{HUB}/{name}/tree/main/data"))
    tables = {}
    for split in ("train", "validation", "test"):
        path = next(f["path"] for f in tree if f["path"].split("/")[-1].startswith(split + "-"))
        print(f"{name}: downloading {path}", flush=True)
        tables[split] = pq.read_table(io.BytesIO(_get(FILES.format(name=name, path=path))))

    rows = {split: table.to_pylist() for split, table in tables.items()}
    used = sorted({c for split in rows.values() for r in split for c in r["objects"]["category"]})
    remap = {old: new for new, old in enumerate(used)}          # the parent class goes
    kept = {remap[c]: names[c] for c in used}
    shutil.rmtree(folder, ignore_errors=True)
    counts = {}
    for split, out in (("train", "train"), ("validation", "val"), ("test", "test")):
        pictures = rows[split]
        cap = caps[out]
        if len(pictures) > cap:
            pictures = random.Random(0).sample(pictures, cap)
        (folder / "images" / out).mkdir(parents=True)
        (folder / "labels" / out).mkdir(parents=True)
        for r in pictures:
            stem = f"{r['image_id']}"
            suffix = Path((r["image"] or {}).get("path") or "x.jpg").suffix or ".jpg"
            (folder / "images" / out / f"{stem}{suffix}").write_bytes(r["image"]["bytes"])
            w, h = float(r["width"]), float(r["height"])
            lines = []
            for (x, y, bw, bh), c in zip(r["objects"]["bbox"], r["objects"]["category"],
                                         strict=True):
                if bw <= 0 or bh <= 0:
                    continue
                cx, cy = min(max((x + bw / 2) / w, 0.0), 1.0), min(max((y + bh / 2) / h, 0.0), 1.0)
                lines.append(f"{remap[c]} {cx:.6f} {cy:.6f} {min(bw / w, 1.0):.6f} "
                             f"{min(bh / h, 1.0):.6f}")
            (folder / "labels" / out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[out] = len(pictures)
    base = {"path": str(folder.resolve()), "names": kept}
    train_yaml.write_text(yaml.safe_dump(dict(base, train="images/train", val="images/val"),
                                         allow_unicode=True))
    score_yaml.write_text(yaml.safe_dump(dict(base, train="images/train", val="images/test"),
                                         allow_unicode=True))
    facts = {"domain": SETS.get(name, "?"), "classes": len(kept), "names": kept, **counts}
    facts_file.write_text(json.dumps(facts, ensure_ascii=False, indent=2))
    return train_yaml, score_yaml, facts


def classroom_epochs(train_pictures: int, coco_known: bool, batch: int = 8) -> int:
    steps = max(1, math.ceil(train_pictures / batch))
    target = CLASSROOM_STEPS if coco_known else CLASSROOM_STEPS_UNFROZEN
    return min(100, max(35, round(target / steps)))


def coco_knows(names: dict) -> bool:
    from easydetect.class_names import ALIASES, match_classes

    return bool(match_classes(dict(enumerate(ALIASES)), {int(k): v for k, v in names.items()}))


def table(results: dict) -> str:
    import statistics

    rows = ["| dataset | domain | classes | train | COCO knows one | epochs | YOLO26n | ed-n | "
            "ed − YOLO |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    diffs = []
    for name, r in results.items():
        y, d = r["yolo"]["map"], r["ed"]["map"]
        diffs.append(d - y)
        rows.append(f"| {name} | {r['domain']} | {r['classes']} | {r['train_pictures']} | "
                    f"{'yes' if r['coco_known'] else 'no'} | {r['epochs']} | {y:.3f} | {d:.3f} | "
                    f"{d - y:+.3f} |")
    if diffs:
        wins = sum(x > 0 for x in diffs)
        rows += ["", f"ed-n ahead on {wins} of {len(diffs)}; mean difference "
                 f"{statistics.mean(diffs):+.3f}, median {statistics.median(diffs):+.3f} "
                 "(mAP50-95 on each test split, one evaluator)."]
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="datasets/rf100", help="where the datasets go")
    parser.add_argument("--out", default="runs/rf100")
    parser.add_argument("--datasets", nargs="*", default=list(SETS))
    parser.add_argument("--train", type=int, default=300, help="training pictures at most")
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="CPU", help="OpenVINO device for scoring")
    parser.add_argument("--yolo", default="yolo26n.pt")
    parser.add_argument("--ed", default="ed-n")
    parser.add_argument("--epochs", type=int, help="instead of the classroom's (for a quick try)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, str(HERE))
    from bench_finetune import score, train_ed, train_yolo

    if shutil.which("yolo") is None:
        raise SystemExit("the yolo command is missing: pip install ultralytics")
    root, out = Path(args.root).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    record = out / "results.json"
    results = json.loads(record.read_text()) if record.exists() else {}
    caps = {"train": args.train, "val": args.val, "test": args.test}
    for name in args.datasets:
        if name in results:
            print(f"{name}: done already", flush=True)
            continue
        train_yaml, score_yaml, facts = prepare(name, root, caps)
        known = coco_knows(facts["names"])
        epochs = args.epochs or classroom_epochs(facts["train"], known)
        print(f"=== {name} ({facts['domain']}, {facts['classes']} classes, {facts['train']} "
              f"training pictures, COCO knows one: {known}, {epochs} epochs)", flush=True)
        run = SimpleNamespace(
            ed=args.ed, label=args.ed, yolo=args.yolo, epochs=epochs, imgsz=320, batch=8,
            workers=args.workers, device=args.device, seed=args.seed, queries=100,
            denoising=30, freeze="auto", no_augment=True, clean_bn=None, lr0=None,
            new_rows=None)
        ed_pt, ed_s = train_ed(train_yaml, out / name, run)
        yolo_onnx, yolo_s = train_yolo(train_yaml, out / name, run)
        r = score(score_yaml, yolo_onnx, ed_pt, run)
        r.update(domain=facts["domain"], classes=facts["classes"], train_pictures=facts["train"],
                 coco_known=known, epochs=epochs,
                 train_minutes={"ed": round(ed_s / 60, 1), "yolo": round(yolo_s / 60, 1)})
        results[name] = r
        record.write_text(json.dumps(results, indent=2))
        (out / "results.md").write_text(table(results))
        print(json.dumps(r), flush=True)
        # the weights are not needed again: the disk is not
        for weights in (out / name).glob("**/weights"):
            shutil.rmtree(weights, ignore_errors=True)
    (out / "results.md").write_text(table(results))
    print("\n" + table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
