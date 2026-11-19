#!/usr/bin/env python3
# Apache-2.0
"""easydetect-hub's datasets, downloaded and laid out the way easydetect trains.

    python tools/hub_data.py taco                      # one
    python tools/hub_data.py --all                     # every one hub_datasets.py lists

Each lands in ``<root>/<key>/`` (``~/datasets/hub`` by default) with
``images/{train,val}`` (links to the pictures where they were unpacked),
``labels/{train,val}`` and ``data.yaml``. Everything resumes: a download
continues where it stopped (``wget -c``), an archive is unpacked into a
temporary folder and only then moved into place, and a dataset already laid
out (``.ready``) is not touched again. A power cut at any point costs at most
the file being written.

LVIS reuses the COCO pictures of ``--coco``; Open Images and construction-ppe
are made by long_run.py's own steps. Roboflow 100 sets keep their test split
for scoring (``score.yaml``), as ``rf100.py`` lays them out.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

#: Roboflow 100 sets on Hugging Face (Francesco/<name>, tagged rf100, CC BY 4.0),
#: less bccd (BCCD is here on its own, MIT) and flir-camera-objects (FLIR's own
#: terms could not be read)
RF100_SETS = """
4-fold-defect abdomen-mri acl-x-ray activity-diagrams-qdobr aerial-cows aerial-pool aerial-spheres
animals-ij5d2 apex-videogame apples-fvpl5 aquarium-qlnqy asbestos avatar-recognition-nuexe
axial-mri bacteria-ptywi bees-jt5in bone-fracture-7fylg brain-tumor-m2pbp cable-damage
cables-nl42k cavity-rs0uf cell-towers cells-uyemf chess-pieces-mjzgj circuit-elements
circuit-voltages cloud-types coins-1apki construction-safety-gsnvb coral-lwptl corrosion-bi3q3
cotton-20xz5 cotton-plant-disease csgo-videogame currency-v4f8j digits-t2eg6 document-parts
excavators-czvg9 farcry6-videogame fish-market-ggjso furniture-ngpea gauge-u2lwv grass-weeds
gynecology-mri halo-infinite-angel-videogame hand-gestures-jps7z insects-mytwu leaf-disease-nsdsr
lettuce-pallets liver-disease marbles mask-wearing-608pr mitosis-gjs3g number-ops paper-parts
parasites-1s07h peanuts-sd4kf peixos-fish people-in-paintings pests-2xlvx phages pills-sxdht
poker-cards-cxcvz printed-circuit-board radio-signal road-signs-6ih4y road-traffic
robomasters-285km secondary-chains sedimentary-features-9eosf shark-teeth-5atku sign-language-sokdr
signatures-xc8up smoke-uvylj soccer-players-5fuqs soda-bottles solar-panels-taxvb stomata-cells
street-work team-fight-tactics thermal-cheetah-my4dp thermal-dogs-and-people-x6ejw trail-camera
truck-movement tweeter-posts tweeter-profile underwater-objects-5v7p8 underwater-pipes-4ng4t
uno-deck valentines-chocolate vehicles-q0x2v wall-damage washroom-rf1fa weed-crop-aerial
wine-labels x-ray-rheumatology
""".split()

URLS = {
    "bccd": "https://github.com/Shenggan/BCCD_Dataset.git",
    "plantdoc": "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git",
    "gwhd": "https://zenodo.org/api/records/5092309/files/gwhd_2021.zip/content",
    "taco": "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json",
    "doclaynet": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
                 "dax-doclaynet/1.0.0/DocLayNet_core.zip",
    "cct-images": "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/"
                  "eccv_18_all_images_sm.tar.gz",
    "cct-labels": "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/"
                  "eccv_18_annotations.tar.gz",
    "lvis-train": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
    "lvis-val": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
}


# ---------------------------------------------------------------- fetching


def _run(cmd: list, cwd: Path | None = None) -> None:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)


def _fetch(url: str, dest: Path) -> Path:
    """``dest``, downloaded once; an interrupted download continues."""
    ok = dest.with_name(dest.name + ".ok")
    if ok.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["wget", "-c", "--progress=dot:giga", "--tries=50", "--waitretry=30",
          "--retry-connrefused", "-O", dest, url])
    ok.touch()
    return dest


def _unpack(url: str, archive: Path, dest: Path, keep: bool = False) -> Path:
    """``dest`` holding the archive's contents; unpacked into a temporary folder
    first, so a half-unpacked one is never taken for done. The archive goes
    afterwards unless ``keep``."""
    marker = dest / ".unpacked"
    if marker.exists():
        return dest
    _fetch(url, archive)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        _run(["unzip", "-q", "-o", archive, "-d", tmp])
    else:
        _run(["tar", "-xf", archive, "-C", tmp])
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    marker.touch()
    if not keep:
        archive.unlink(missing_ok=True)
        archive.with_name(archive.name + ".ok").unlink(missing_ok=True)
    return dest


def _clone(url: str, dest: Path) -> Path:
    if (dest / ".cloned").exists():
        return dest
    shutil.rmtree(dest, ignore_errors=True)
    _run(["git", "clone", "--depth", "1", url, dest])
    (dest / ".cloned").touch()
    return dest


# ---------------------------------------------------------------- laying out


class Layout:
    """``images/<split>`` links and ``labels/<split>`` files under ``folder``,
    made afresh (they are cheap: the pictures stay where they were unpacked)."""

    def __init__(self, folder: Path):
        self.folder = folder
        for part in ("images", "labels"):
            shutil.rmtree(folder / part, ignore_errors=True)
        for split in ("train", "val"):
            (folder / "images" / split).mkdir(parents=True)
            (folder / "labels" / split).mkdir(parents=True)
        self.count = defaultdict(int)

    def add(self, split: str, image: Path, rows: list, unknown: list | None = None,
            stem: str | None = None) -> None:
        """``rows``: (class, cx, cy, w, h) normalised."""
        stem = stem or image.stem
        link = self.folder / "images" / split / f"{stem}{image.suffix.lower()}"
        if link.exists() or link.is_symlink():
            return                                   # the same picture twice
        os.symlink(image.resolve(), link)
        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows]
        label = self.folder / "labels" / split / f"{stem}.txt"
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
        if unknown:
            label.with_suffix(".unknown").write_text(" ".join(map(str, sorted(unknown))) + "\n")
        self.count[split] += 1

    def finish(self, names: list[str], unknown: bool = False) -> Path:
        data = self.folder / "data.yaml"
        cfg = {"path": str(self.folder.resolve()), "train": "images/train", "val": "images/val",
               "names": dict(enumerate(names))}
        if unknown:
            cfg["unknown_classes"] = True
        data.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        (self.folder / ".ready").touch()
        print(f"{self.folder.name}: {self.count['train']} train, {self.count['val']} val, "
              f"{len(names)} classes -> {data}", flush=True)
        return data


def box(c: int, x0: float, y0: float, x1: float, y1: float, w: float, h: float):
    """Corners in pixels of a ``w`` x ``h`` picture -> (c, cx, cy, bw, bh) normalised,
    clipped to the picture; None when nothing is left."""
    x0, x1 = max(0.0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(h, max(y0, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (c, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)


def _coco_like(lay: Layout, split: str, ann: dict, image_dir: Path, cat_index: dict,
               path_of=None, unknown_of=None) -> None:
    """A COCO-format split: boxes [x, y, w, h] in pixels of each image's width x height."""
    per_image = defaultdict(list)
    for a in ann["annotations"]:
        if a.get("bbox") and a["category_id"] in cat_index and not a.get("iscrowd"):
            per_image[a["image_id"]].append(a)
    for im in ann["images"]:
        path = path_of(im) if path_of else image_dir / im["file_name"]
        if not path.exists():
            continue
        w, h = float(im["width"]), float(im["height"])
        rows = [r for a in per_image[im["id"]]
                for r in [box(cat_index[a["category_id"]], a["bbox"][0], a["bbox"][1],
                              a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3], w, h)]
                if r]
        lay.add(split, path, rows, unknown_of(im) if unknown_of else None,
                stem=str(im["id"]) if path_of else None)


# ---------------------------------------------------------------- the datasets


def bccd(folder: Path) -> Path:
    base = _clone(URLS["bccd"], folder / "src") / "BCCD"
    names = ["Platelets", "RBC", "WBC"]
    lay = Layout(folder)
    for split, lists in (("train", ["train"]), ("val", ["val", "test"])):
        for listed in lists:
            for stem in (base / "ImageSets" / "Main" / f"{listed}.txt").read_text().split():
                root = ET.parse(base / "Annotations" / f"{stem}.xml").getroot()
                w = float(root.findtext("size/width"))
                h = float(root.findtext("size/height"))
                rows = []
                for o in root.iter("object"):
                    name = (o.findtext("name") or "").strip()
                    if name in names:
                        b = o.find("bndbox")
                        r = box(names.index(name), *(float(b.findtext(k)) for k in
                                                     ("xmin", "ymin", "xmax", "ymax")), w, h)
                        if r:
                            rows.append(r)
                lay.add(split, base / "JPEGImages" / f"{stem}.jpg", rows)
    return lay.finish(names)


def plantdoc(folder: Path) -> Path:
    import csv

    src = _clone(URLS["plantdoc"], folder / "src")
    tables = {"train": ("train_labels.csv", "TRAIN"), "val": ("test_labels.csv", "TEST")}
    rows_of = {}
    for split, (table, _) in tables.items():
        with open(src / table, newline="", encoding="utf-8") as f:
            rows_of[split] = list(csv.DictReader(f))
    names = sorted({r["class"].strip() for rows in rows_of.values() for r in rows})
    lay = Layout(folder)
    for split, (_, images) in tables.items():
        per_file = defaultdict(list)
        for r in rows_of[split]:
            per_file[r["filename"]].append(r)
        for k, (name, rows) in enumerate(per_file.items()):
            path = next((p for p in (src / images / name,
                                     src / images / urllib.parse.unquote(name)) if p.exists()),
                        None)
            if path is None:
                continue
            boxes = []
            for r in rows:
                w, h = float(r["width"] or 0), float(r["height"] or 0)
                if w > 0 and h > 0:
                    b = box(names.index(r["class"].strip()), float(r["xmin"]), float(r["ymin"]),
                            float(r["xmax"]), float(r["ymax"]), w, h)
                    if b:
                        boxes.append(b)
            lay.add(split, path, boxes, stem=f"{split}{k:05d}")
    return lay.finish(names)


def gwhd(folder: Path) -> Path:
    import csv

    import cv2

    src = _unpack(URLS["gwhd"], folder / "gwhd_2021.zip", folder / "src")
    lay = Layout(folder)
    for split, table in (("train", "competition_train.csv"), ("val", "competition_val.csv")):
        found = next(src.rglob(table))
        images = found.parent / "images"
        with open(found, newline="") as f:
            for r in csv.DictReader(f):
                path = images / r["image_name"]
                img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
                if img is None:
                    continue
                h, w = img.shape[0] * 8, img.shape[1] * 8          # 1024 x 1024
                rows = []
                if r["BoxesString"].strip() != "no_box":
                    for part in r["BoxesString"].split(";"):
                        b = box(0, *map(float, part.split()), w, h)
                        if b:
                            rows.append(b)
                lay.add(split, path, rows)
    return lay.finish(["wheat head"])


def taco(folder: Path, workers: int = 8) -> Path:
    import concurrent.futures
    import urllib.request

    import cv2
    import numpy as np

    ann = json.loads(_fetch(URLS["taco"], folder / "annotations.json").read_text())
    src = folder / "src"

    def get(im) -> None:
        out = src / (Path(im["file_name"]).with_suffix(".jpg"))
        if out.exists():
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                request = urllib.request.Request(im["flickr_url"],
                                                 headers={"User-Agent": "easydetect"})
                with urllib.request.urlopen(request, timeout=60) as r:
                    raw = np.frombuffer(r.read(), np.uint8)
                break
            except Exception:  # noqa: BLE001  (a picture gone from Flickr is skipped)
                raw = None
        if raw is None:
            print(f"taco: {im['file_name']} could not be fetched; left out", flush=True)
            return
        # the boxes are on the picture as its EXIF says to show it
        w, h = im["width"], im["height"]
        for flags in (cv2.IMREAD_COLOR, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION):
            img = cv2.imdecode(raw, flags)
            if img is not None and img.shape[:2] == (h, w):
                break
        else:
            print(f"taco: {im['file_name']} is not {w}x{h}; left out", flush=True)
            return
        scale = 1600 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tmp = out.with_name(out.stem + ".part.jpg")
        cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tmp.rename(out)

    total = len(ann["images"])
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(get, ann["images"]), 1):
            if i % 50 == 0 or i == total:
                have = sum(1 for _ in src.rglob("*.jpg")) if src.exists() else 0
                print(f"taco: {i}/{total} pictures tried, {have} here", flush=True)
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}
    ids = sorted(im["id"] for im in ann["images"])
    random.Random(0).shuffle(ids)
    val = set(ids[: len(ids) // 10])
    lay = Layout(folder)
    for split in ("train", "val"):
        part = {"images": [im for im in ann["images"] if (im["id"] in val) == (split == "val")],
                "annotations": ann["annotations"]}
        _coco_like(lay, split, part, src, index,
                   path_of=lambda im: src / Path(im["file_name"]).with_suffix(".jpg"))
    return lay.finish([c["name"] for c in cats])


def doclaynet(folder: Path) -> Path:
    src = _unpack(URLS["doclaynet"], folder / "DocLayNet_core.zip", folder / "src")
    coco = next(src.rglob("COCO"))
    png = next(p for p in src.rglob("PNG") if p.is_dir())
    lay = Layout(folder)
    names = None
    for split in ("train", "val"):
        ann = json.loads((coco / f"{split}.json").read_text())
        cats = sorted(ann["categories"], key=lambda c: c["id"])
        names = [c["name"] for c in cats]
        _coco_like(lay, split, ann, png, {c["id"]: i for i, c in enumerate(cats)})
    return lay.finish(names)


def caltech_camera_traps(folder: Path) -> Path:
    images = _unpack(URLS["cct-images"], folder / "cct_images.tar.gz", folder / "src-images")
    labels = _unpack(URLS["cct-labels"], folder / "cct_labels.tar.gz", folder / "src-labels")
    files = {p.name: p for p in images.rglob("*.jpg")}

    def find(im) -> Path:
        return files.get(Path(im["file_name"]).name, Path("/nonexistent"))

    # the benchmark's test splits are boxed too: here they train
    splits = {"train": ["train_annotations.json", "cis_test_annotations.json",
                        "trans_test_annotations.json"],
              "val": ["cis_val_annotations.json", "trans_val_annotations.json"]}
    anns = {s: [json.loads(next(labels.rglob(n)).read_text()) for n in names]
            for s, names in splits.items()}
    cats = {}
    for parts in anns.values():
        for a in parts:
            cats.update({c["id"]: c["name"] for c in a["categories"]})
    kept = sorted((i for i, n in cats.items() if n.lower() != "empty"), key=lambda i: cats[i])
    index = {i: k for k, i in enumerate(kept)}
    lay = Layout(folder)
    for split, parts in anns.items():
        for a in parts:
            # a picture with an animal named but not boxed would teach it as absent
            unboxed = {x["image_id"] for x in a["annotations"]
                       if x["category_id"] in index and not x.get("bbox")}
            a = {"images": [im for im in a["images"] if im["id"] not in unboxed],
                 "annotations": a["annotations"]}
            _coco_like(lay, split, a, images, index, path_of=find)
    return lay.finish([cats[i] for i in kept])


def lvis(folder: Path, coco: Path) -> Path:
    anns = {}
    for split in ("train", "val"):
        unpacked = _unpack(URLS[f"lvis-{split}"], folder / f"lvis_v1_{split}.json.zip",
                           folder / f"src-{split}", keep=False)
        anns[split] = json.loads(next(unpacked.rglob("*.json")).read_text())
    cats = sorted(anns["train"]["categories"], key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}

    def path_of(im):
        part = "train2017" if "train2017" in im["coco_url"] else "val2017"
        return coco / "images" / part / f"{im['id']:012d}.jpg"

    def unknown_of(im):
        # LVIS boxes only some of the instances of these classes in this picture
        return [index[c] for c in im.get("not_exhaustive_category_ids", []) if c in index]

    lay = Layout(folder)
    for split, ann in anns.items():
        _coco_like(lay, split, ann, coco, index, path_of=path_of, unknown_of=unknown_of)
    return lay.finish([c["name"] for c in cats], unknown=True)


def rf100(name: str, root: Path) -> tuple[Path, Path]:
    """(train.yaml, score.yaml) of the whole set, laid out by rf100.py."""
    sys.path.insert(0, str(HERE))
    from rf100 import prepare

    big = 10 ** 9
    train, score, _ = prepare(name, root / "rf100", {"train": big, "val": big, "test": big})
    return train, score


MAKERS = {"bccd": bccd, "plantdoc": plantdoc, "gwhd": gwhd, "taco": taco,
          "doclaynet": doclaynet, "caltech-camera-traps": caltech_camera_traps}


#: what a prefetch fetches, in two lanes at once: the quick ones (training
#: never waits long for the next) and the big downloads, which take hours
QUICK = ["bccd", "taco", "plantdoc", *(f"rf100-{n}" for n in RF100_SETS)]
BIG = ["gwhd", "doclaynet", "caltech-camera-traps", "lvis"]
ORDER = QUICK + BIG


def _lock_path(key: str, root: Path) -> Path:
    return root / ".locks" / f"{key}.lock"


def is_ready(key: str, root: Path) -> bool:
    """Laid out already: training can start on it at once."""
    if key.startswith("rf100-"):
        return (root / "rf100" / key[len("rf100-"):] / "facts.json").exists()
    return (root / key / ".ready").exists()


def prefetch_running(root: Path) -> bool:
    """A prefetch is under way: what it has not laid out yet, it will."""
    return is_busy("prefetch", root)


def is_busy(key: str, root: Path) -> bool:
    """Another process (a prefetch) is downloading or laying it out now."""
    import fcntl

    path = _lock_path(key, root)
    if not path.exists():
        return False
    with open(path) as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


@contextlib.contextmanager
def _locked(path: Path, what: str):
    """One process at a time on a dataset (a prefetch and a training step may
    both want it); the other waits, saying so every ten minutes."""
    import fcntl
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        waited = 0
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if waited % 600 == 0:
                    print(f"{what}: another process is preparing it; waiting", flush=True)
                time.sleep(30)
                waited += 30
        yield


def prepare(key: str, root: Path, coco: Path) -> tuple[Path, Path | None]:
    """(data.yaml to train on, data.yaml to score on when not the same)."""
    with _locked(_lock_path(key, root), key):
        if key.startswith("rf100-"):
            return rf100(key[len("rf100-"):], root)
        folder = root / key
        if (folder / ".ready").exists():
            return folder / "data.yaml", None
        folder.mkdir(parents=True, exist_ok=True)
        if key == "lvis":
            return lvis(folder, coco), None
        return MAKERS[key](folder), None


def prefetch(keys: list[str], root: Path, coco: Path) -> int:
    """Every dataset in ``keys`` made ready, one after another, while training
    runs on the GPU; one prefetch at a time, a failure left for the training
    step to try again."""
    import fcntl

    (root / ".locks").mkdir(parents=True, exist_ok=True)
    handle = open(_lock_path("prefetch", root), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("a prefetch is running already", flush=True)
        return 0
    import concurrent.futures

    def lane(names: list[str]) -> None:
        for key in names:
            try:
                print(f"{key}: {prepare(key, root, coco)[0]}", flush=True)
            except Exception as exc:  # noqa: BLE001  (its training step tries again)
                print(f"{key}: not ready ({exc}); its training step will try again",
                      flush=True)

    lanes = [[k for k in keys if k not in BIG], [k for k in keys if k in BIG]]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        list(pool.map(lane, lanes))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("keys", nargs="*")
    parser.add_argument("--all", action="store_true", help="every dataset, in training order")
    parser.add_argument("--skip", nargs="*", default=[], help="leave these out")
    parser.add_argument("--prefetch", action="store_true",
                        help="one at a time behind a lock, going on past a failure")
    parser.add_argument("--root", default="~/datasets/hub")
    parser.add_argument("--coco", default="~/datasets/coco")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    keys = [k for k in (ORDER if args.all else args.keys) if k not in args.skip]
    root, coco = Path(args.root).expanduser(), Path(args.coco).expanduser()
    if args.prefetch:
        return prefetch(keys, root, coco)
    for key in keys:
        print(prepare(key, root, coco), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
