#!/usr/bin/env python3
# Apache-2.0
"""Open Images V7's boxes as a YOLO dataset easydetect trains on.

    python tools/openimages2yolo.py --out ~/datasets/openimages
    python tools/openimages2yolo.py --out /tmp/oi-try --train-images 2000 --val-images 200

Open Images (Kuznetsova et al., 2020; storage.googleapis.com/openimages):
600 classes boxed on 1.74 million training pictures. The boxes are CC BY 4.0
and the pictures are listed as CC BY 2.0 by their owners (Google does not
check each one), so a model trained on them can be shared with that credit.

Not every object in a picture is boxed. A class is taught as absent wherever
it has no box, as on COCO, except where that is known to be wrong; those
classes are listed in ``<stem>.unknown`` beside the label file, and
data.yaml says ``unknown_classes: true``, so training does not teach them as
absent there and scoring counts their boxes neither right nor wrong:

* a class boxed as a group ("a crowd of people", IsGroupOf) loses all its
  boxes in that picture: the people in the crowd are not boxed one by one;
* the classes above a boxed one ("Vehicle" over "Car"), where only the
  narrower one is boxed;
* a class checked as present but not boxed (it is there, without a box).

Only 2-3 classes a picture are checked as absent, against 600, so teaching
only those as absent (as LVIS does) would leave too little to learn from.

Pictures are fetched from the public open-images-dataset bucket (no key) and
stored with their longer side at most ``--max-side`` px; labels are
normalised, so the size does not change them. A rerun skips what is there.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ANNOTATIONS = "https://storage.googleapis.com/openimages"
CSVS = {
    "classes": f"{ANNOTATIONS}/v7/oidv7-class-descriptions-boxable.csv",
    "hierarchy": f"{ANNOTATIONS}/2018_04/bbox_labels_600_hierarchy.json",
    "train-boxes": f"{ANNOTATIONS}/v6/oidv6-train-annotations-bbox.csv",
    "train-labels": f"{ANNOTATIONS}/v7/oidv7-train-annotations-human-imagelabels.csv",
    "validation-boxes": f"{ANNOTATIONS}/v5/validation-annotations-bbox.csv",
    "validation-labels": f"{ANNOTATIONS}/v7/oidv7-val-annotations-human-imagelabels.csv",
}
PICTURES = "https://open-images-dataset.s3.amazonaws.com/{split}/{image}.jpg"
SPLITS = {"train": "train", "validation": "val"}


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    part.replace(dest)
    return dest


def ancestors(hierarchy: dict) -> dict[str, set[str]]:
    """Each class's broader classes ("/m/0k4j" Car → Land vehicle, Vehicle)."""
    up: dict[str, set[str]] = defaultdict(set)

    def walk(node, above):
        name = node["LabelName"]
        up[name] |= above
        for child in node.get("Subcategory", []):
            walk(child, above | {name})

    walk(hierarchy, set())
    return up


def rows_by_image(path: Path, keep_images: set[str] | None = None):
    """``(image, [rows])`` for a CSV sorted by ImageID, one picture at a time."""
    current, rows, seen = None, [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = row["ImageID"]
            if image != current:
                if current is not None and (keep_images is None or current in keep_images):
                    yield current, rows
                if image in seen:
                    raise SystemExit(f"{path.name} is not sorted by ImageID ({image} again)")
                seen.add(image)
                current, rows = image, []
            rows.append(row)
    if current is not None and (keep_images is None or current in keep_images):
        yield current, rows


def checked_present(path: Path, index: dict[str, int], images: set[str]) -> dict[str, set[int]]:
    """Per picture, the boxable classes a person saw there (Confidence 1)."""
    present: dict[str, set[int]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["Confidence"]) > 0 and row["ImageID"] in images:
                cls = index.get(row["LabelName"])
                if cls is not None:
                    present[row["ImageID"]].add(cls)
    return present


def picture_ids(path: Path) -> list[str]:
    ids, last = [], None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] != last:
                last = row["ImageID"]
                ids.append(last)
    return ids


def labels_for(rows, index, up) -> tuple[list[str], list[int], set[int]]:
    """``(YOLO lines, classes boxed, classes unknown)`` for one picture's box rows."""
    grouped = {index[r["LabelName"]] for r in rows
               if r["IsGroupOf"] == "1" and r["LabelName"] in index}
    lines, boxed = [], set()
    for r in rows:
        cls = index.get(r["LabelName"])
        if cls is None or cls in grouped:
            continue
        x1, x2, y1, y2 = (float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cls} {x1 + w / 2:.6f} {y1 + h / 2:.6f} {w:.6f} {h:.6f}")
        boxed.add(cls)
    broader = {index[a] for r in rows if index.get(r["LabelName"]) in boxed
               for a in up.get(r["LabelName"], ()) if a in index}
    return lines, sorted(boxed), grouped | (broader - boxed)


def store_picture(url: str, dest: Path, max_side: int) -> bool:
    import cv2
    import numpy as np

    if dest.exists() and dest.stat().st_size:
        return True
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = np.frombuffer(r.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return False
            scale = max_side / max(img.shape[:2])
            if scale < 1:
                img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)),
                                 interpolation=cv2.INTER_AREA)
            part = dest.with_suffix(".part.jpg")
            cv2.imwrite(str(part), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            part.replace(dest)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return False
        except (OSError, TimeoutError):
            pass
        time.sleep(2 ** attempt)
    return False


def convert(split: str, args, index, up, names) -> int:
    csv_dir = Path(args.out) / "annotations"
    boxes = fetch(CSVS[f"{split}-boxes"], csv_dir / Path(CSVS[f"{split}-boxes"]).name)
    labels = fetch(CSVS[f"{split}-labels"], csv_dir / Path(CSVS[f"{split}-labels"]).name)
    ids = picture_ids(boxes)
    cap = args.train_images if split == "train" else args.val_images
    if cap and cap < len(ids):
        ids = sorted(random.Random(0).sample(ids, cap))
    wanted = set(ids)
    print(f"{split}: {len(ids)} pictures; reading which classes were seen", flush=True)
    present = checked_present(labels, index, wanted)
    folder = SPLITS[split]
    img_dir, lab_dir = Path(args.out) / "images" / folder, Path(args.out) / "labels" / folder
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for image, rows in rows_by_image(boxes, wanted):
        lines, boxed, unknown = labels_for(rows, index, up)
        if not lines:
            continue
        unknown = (unknown | present.get(image, set())) - set(boxed)
        (lab_dir / f"{image}.txt").write_text("\n".join(lines) + "\n")
        sidecar = lab_dir / f"{image}.unknown"
        if unknown:
            sidecar.write_text(" ".join(map(str, sorted(unknown))) + "\n")
        else:
            sidecar.unlink(missing_ok=True)
        todo.append(image)
    print(f"{split}: {len(todo)} pictures with boxes; fetching them", flush=True)
    done, failed, lock, started = [0], [], threading.Lock(), time.time()

    def one(image):
        ok = store_picture(PICTURES.format(split=split, image=image), img_dir / f"{image}.jpg",
                           args.max_side)
        with lock:
            done[0] += 1
            if not ok:
                failed.append(image)
            if done[0] % 5000 == 0:
                rate = done[0] / max(time.time() - started, 1e-6)
                print(f"{split}: {done[0]}/{len(todo)} pictures ({rate:.0f}/s)", flush=True)

    with ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(one, todo))
    for image in failed:            # no picture: no labels either
        for suffix in (".txt", ".unknown"):
            (lab_dir / f"{image}{suffix}").unlink(missing_ok=True)
    print(f"{split}: {len(todo) - len(failed)} pictures, {len(failed)} could not be fetched",
          flush=True)
    return len(todo) - len(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="~/datasets/openimages")
    parser.add_argument("--train-images", type=int, default=0, help="at most this many (0: all)")
    parser.add_argument("--val-images", type=int, default=5000,
                        help="validation pictures to keep, drawn at random (0: all 41,620)")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--splits", default="validation,train")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    args.out = str(Path(args.out).expanduser())
    csv_dir = Path(args.out) / "annotations"
    with open(fetch(CSVS["classes"], csv_dir / "oidv7-class-descriptions-boxable.csv"),
              newline="", encoding="utf-8") as f:
        classes = [(r["LabelName"], r["DisplayName"]) for r in csv.DictReader(f)]
    index = {label: i for i, (label, _) in enumerate(classes)}
    names = {i: name for i, (_, name) in enumerate(classes)}
    up = ancestors(json.loads(fetch(CSVS["hierarchy"],
                                    csv_dir / "bbox_labels_600_hierarchy.json").read_text()))
    counts = {split: convert(split, args, index, up, names) for split in args.splits.split(",")}
    data = Path(args.out) / "data.yaml"
    import yaml

    data.write_text(yaml.safe_dump({"path": args.out, "train": "images/train",
                                    "val": "images/val", "names": names,
                                    "unknown_classes": True}, allow_unicode=True,
                                   sort_keys=False))
    print(f"{data}: {counts} pictures, {len(names)} classes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
