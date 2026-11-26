#!/usr/bin/env python3
# Apache-2.0
"""easydetect-hub's datasets, downloaded and laid out the way easydetect trains.

    python tools/hub_data.py taco                      # one
    python tools/hub_data.py --all                     # every one hub_datasets.py lists
    python tools/hub_data.py --new --prefetch          # the ones added later, beside training

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
import re
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
    "fashionpedia-train": "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip",
    "fashionpedia-val": "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip",
    "fashionpedia-ann-train": "https://s3.amazonaws.com/ifashionist-dataset/annotations/"
                              "instances_attributes_train2020.json",
    "fashionpedia-ann-val": "https://s3.amazonaws.com/ifashionist-dataset/annotations/"
                            "instances_attributes_val2020.json",
    # AP-10K's split1 as YOLO-pose files, unchanged from the upstream archive
    # (github.com/AlexTheBad/AP-10K, CC BY 4.0), which is on Google Drive only
    "ap10k": "https://huggingface.co/datasets/LibreYOLO/ap10k-pose/resolve/main/ap10k-pose.zip",
    "izembek-meta": "https://storage.googleapis.com/public-datasets-lila/izembek-lagoon-birds/"
                    "izembek-lagoon-birds-metadata.zip",
    "izembek-images": "https://storage.googleapis.com/public-datasets-lila/izembek-lagoon-birds/",
    "fathomnet": "https://database.fathomnet.org/api/images/query",
    "coco-text": "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip",
    "ena24-images": "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip",
    "ena24-labels": "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json",
    "boxes-on-bees": "https://storage.googleapis.com/public-datasets-lila/boxes-on-bees/"
                     "boxes-on-bees-2021.zip",
    "bbbc038": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
    "nuinsseg": "https://zenodo.org/api/records/10518968/files/NuInsSeg.zip/content",
    "pped": "https://zenodo.org/api/records/6551758/files/PPED_data.zip/content",
    "loco-images": "https://go.mytum.de/239870",
    "loco-labels": "https://raw.githubusercontent.com/tum-fml/loco/main/rgb/loco-all-v1.json",
    "msu-apples": "https://zenodo.org/api/records/7925397/files/MSUAppleData.zip/content",
    "swinepose": "https://zenodo.org/api/records/24221533/files/SwinePose_v2.0.zip/content",
    "bbbc039-images": "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
    "bbbc039-masks": "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
    "bbbc039-metadata": "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
    "noaa-psnf-images": "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/"
                        "noaa_estuary_fish-images.zip",
    "noaa-psnf-labels": "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/"
                        "noaa_estuary_fish-annotations-2023.08.19.zip",
    "armbench": "https://huggingface.co/datasets/amazon/armbench-segmentation/resolve/main/"
                "archive/armbench-segmentation-0.1.tar.gz",
    "artf-clothes": "https://zenodo.org/api/records/14501499/files/"
                    "artf-all-categories-resized_512x256.zip/content",
    "everwatch": "https://zenodo.org/api/records/17317117/files/everwatch-dataset.zip/content",
    "visa": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
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
    import time

    for n in range(6):              # a server that refuses for a while (Zenodo)
        try:
            _run(["wget", "-c", "--progress=dot:giga", "--tries=50", "--waitretry=30",
                  "--retry-connrefused", "-O", dest, url])
            break
        except subprocess.CalledProcessError:
            if n == 5:
                raise
            print(f"{dest.name}: the server refused; again in {5 * (n + 1)} minutes",
                  flush=True)
            time.sleep(300 * (n + 1))
    ok.touch()
    return dest


def _unzip(archive: Path, dest: Path) -> None:
    """unzip; where it stops, Python's zipfile a file at a time, leaving out
    the few whose contents are damaged (bad CRC). Many damaged, or no archive
    at all: deleted, so the next try downloads it again."""
    import zipfile

    try:
        _run(["unzip", "-q", "-o", archive, "-d", dest])
        return
    except subprocess.CalledProcessError as exc:
        print(f"{archive.name}: unzip stopped ({exc.returncode}); going through it with "
              "Python's zipfile", flush=True)
    bad, total = [], 0
    try:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                total += 1
                try:
                    z.extract(info, dest)
                except (zipfile.BadZipFile, zlib_error()) as exc:
                    bad.append(info.filename)
                    (dest / info.filename).unlink(missing_ok=True)
                    print(f"{archive.name}: left out {info.filename} ({exc})", flush=True)
    except (zipfile.BadZipFile, EOFError, OSError):
        bad.append("(the archive)")
    if bad and (bad[-1] == "(the archive)" or len(bad) > max(10, total // 100)):
        archive.unlink(missing_ok=True)
        archive.with_name(archive.name + ".ok").unlink(missing_ok=True)
        raise RuntimeError(f"{archive.name}: damaged ({len(bad)} of {total} files); "
                           "deleted, the next try downloads it again")
    if bad:
        print(f"{archive.name}: {len(bad)} of {total} files damaged and left out", flush=True)


def zlib_error() -> type:
    import zlib

    return zlib.error


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
        _unzip(archive, tmp)
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


#: Roboflow100-VL (github.com/roboflow/rf100-vl, 2025): 100 more Roboflow
#: Universe datasets, released by its authors under Apache-2.0. tools/rf100vl.json
#: keeps each one's category and those left out (in Roboflow 100 too, already
#: here on their own, or under terms of their own)
RF100VL = json.loads((HERE / "rf100vl.json").read_text())
RF100VL_URL = "https://huggingface.co/datasets/probicheaux/rf100-vl/resolve/main"
RF100VL_TREE = "https://huggingface.co/api/datasets/probicheaux/rf100-vl/tree/main/data"


def rf100vl_key(name: str) -> str:
    return "rf100vl-" + re.sub(r"-+", "-", name).strip("-")


#: hub key -> RF100-VL dataset name, for the ones trained here
RF100VL_SETS = {rf100vl_key(n): n for n in sorted(RF100VL["category"]) if n not in RF100VL["skip"]}


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


#: Fashionpedia's pictures each keep their own licence; only these are used:
#: CC BY 2.0, public domain, and the free stock sites' (Unsplash, Pexels,
#: freestocks, Burst). Non-commercial, no-derivatives, share-alike and
#: unknown ones are left out.
FASHIONPEDIA_PICTURES = {"Attribution License", "Public Domain Dedication", "unsplash", "pexels",
                         "freestocks", "burst"}


def fashionpedia(folder: Path) -> Path:
    """Clothes and their parts (46 classes), from the pictures whose licence
    allows any use (``FASHIONPEDIA_PICTURES``). The masks stay in
    ``ann-{train,val}.json`` for the segmentation model."""
    train_dir = _unpack(URLS["fashionpedia-train"], folder / "train2020.zip", folder / "src-train")
    val_dir = _unpack(URLS["fashionpedia-val"], folder / "val_test2020.zip", folder / "src-val")
    files = {p.name: p for d in (train_dir, val_dir) for p in d.rglob("*.jpg")}
    lay = Layout(folder)
    names = None
    for split in ("train", "val"):
        ann = json.loads(_fetch(URLS[f"fashionpedia-ann-{split}"], folder / f"ann-{split}.json")
                         .read_text())
        free = {x["id"] for x in ann["licenses"] if x["name"] in FASHIONPEDIA_PICTURES}
        ann = {"images": [im for im in ann["images"] if im.get("license") in free],
               "annotations": ann["annotations"], "categories": ann["categories"]}
        cats = sorted(ann["categories"], key=lambda c: c["id"])
        names = [c["name"] for c in cats]
        _coco_like(lay, split, ann, folder, {c["id"]: i for i, c in enumerate(cats)},
                   path_of=lambda im: files.get(Path(im["file_name"]).name, Path("/nonexistent")))
    return lay.finish(names)


def ap10k(folder: Path) -> Path:
    """54 animal species, boxed; the 17 keypoints of each stay in the source
    labels (``src/**/labels``) for a keypoint model."""
    src = _unpack(URLS["ap10k"], folder / "ap10k-pose.zip", folder / "src")
    cfg = yaml.safe_load(next(src.rglob("*.yaml")).read_text())
    base = next(p.parent for p in src.rglob("images") if (p / "train").is_dir())
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    lay = Layout(folder)
    for split in ("train", "val"):
        for image in sorted((base / "images" / split).iterdir()):
            label = base / "labels" / split / f"{image.stem}.txt"
            rows = []
            if label.exists():
                for line in label.read_text().splitlines():
                    v = line.split()
                    if len(v) >= 5:
                        rows.append((int(float(v[0])), *map(float, v[1:5])))
            lay.add(split, image, rows)
    return lay.finish(names)


def ap10k_pose_yaml(folder: Path) -> Path:
    """``<folder>/pose.yaml``: AP-10K's keypoint labels as easydetect reads a
    keypoint dataset (``kpt_names``, a skeleton counted from 0, its OKS sigmas)."""
    src = folder / "src"
    cfg = yaml.safe_load(next(src.rglob("*.yaml")).read_text())
    base = next(p.parent for p in src.rglob("images") if (p / "train").is_dir())
    k = int(cfg["kpt_shape"][0])
    skeleton = [list(map(int, e)) for e in cfg.get("skeleton") or []]
    if skeleton and max(max(e) for e in skeleton) == k:          # counted from 1
        skeleton = [[a - 1, b - 1] for a, b in skeleton]
    out = folder / "pose.yaml"
    out.write_text(yaml.safe_dump({
        "path": str(base.resolve()), "train": "images/train", "val": "images/val",
        "names": cfg["names"], "kpt_shape": cfg["kpt_shape"], "flip_idx": cfg["flip_idx"],
        "kpt_names": cfg.get("keypoints"), "skeleton": skeleton,
        "oks_sigmas": cfg.get("oks_sigmas")}, allow_unicode=True, sort_keys=False))
    return out


def _get(url: str, dest: Path, tries: int = 4) -> bool:
    """One file over http, into place only when whole."""
    import time
    import urllib.request

    tmp = dest.with_name(dest.name + ".part")
    for n in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "easydetect"})
            with urllib.request.urlopen(request, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.rename(dest)
            return True
        except Exception:  # noqa: BLE001  (tried again, then left out)
            time.sleep(5 * (n + 1))
    tmp.unlink(missing_ok=True)
    return False


def izembek(folder: Path, workers: int = 8, tile: int = 640, empty: float = 0.05,
            limit: int | None = None) -> Path:
    """Waterfowl from the air (CC0): 8688 x 5792 pictures where a bird is some
    36 px wide, so each picture is cut into ``tile`` px squares (birds keep
    their size) as it arrives, and the picture is dropped; the squares with
    birds are kept, and ``empty`` of the others. A box goes with the square
    holding at least half of it. Val takes whole pictures."""
    import concurrent.futures

    import cv2

    meta = _unpack(URLS["izembek-meta"], folder / "meta.zip", folder / "src-meta")
    ann = json.loads(next(meta.rglob("*.json")).read_text())
    cats = sorted((c for c in ann["categories"] if c["name"] != "Empty"), key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}
    per_image = defaultdict(list)
    for a in ann["annotations"]:
        if a.get("bbox") and a["category_id"] in index:
            per_image[a["image_id"]].append(a)
    images = sorted((im for im in ann["images"] if per_image[im["id"]]), key=lambda im: im["id"])
    images = images[:limit]                     # a few, to try it
    tiles, tmp = folder / "tiles", folder / "tmp"
    tiles.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)

    def starts(n: int) -> list[int]:
        return sorted(set(list(range(0, max(n - tile, 0), tile)) + [max(n - tile, 0)]))

    def cut(im) -> None:
        stem = Path(im["file_name"]).stem
        done = tiles / f"{stem}.json"
        if done.exists():
            return
        raw = tmp / Path(im["file_name"]).name
        if not raw.exists() and not _get(URLS["izembek-images"] + im["file_name"], raw):
            print(f"izembek: {im['file_name']} could not be fetched; left out", flush=True)
            return
        img = cv2.imread(str(raw))
        if img is None:
            raw.unlink(missing_ok=True)
            return
        h, w = img.shape[:2]
        rng = random.Random(stem)
        made = []
        for y0 in starts(h):
            for x0 in starts(w):
                rows = []
                for a in per_image[im["id"]]:
                    bx, by, bw, bh = a["bbox"]
                    ix = min(bx + bw, x0 + tile) - max(bx, x0)
                    iy = min(by + bh, y0 + tile) - max(by, y0)
                    if ix > 0 and iy > 0 and ix * iy >= 0.5 * bw * bh:
                        r = box(index[a["category_id"]], bx - x0, by - y0, bx + bw - x0,
                                by + bh - y0, tile, tile)
                        if r:
                            rows.append(r)
                if not rows and rng.random() >= empty:
                    continue
                name = f"{stem}_{x0}_{y0}.jpg"
                cv2.imwrite(str(tiles / name), img[y0:y0 + tile, x0:x0 + tile],
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                made.append([name, rows])
        done.write_text(json.dumps(made))
        raw.unlink(missing_ok=True)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(cut, images), 1):
            if i % 100 == 0 or i == len(images):
                print(f"izembek: {i}/{len(images)} pictures cut", flush=True)
    stems = [Path(im["file_name"]).stem for im in images]
    random.Random(0).shuffle(stems)
    val = set(stems[: len(stems) // 10])
    lay = Layout(folder)
    for stem in sorted(stems):
        done = tiles / f"{stem}.json"
        if not done.exists():
            continue
        for name, rows in json.loads(done.read_text()):
            lay.add("val" if stem in val else "train", tiles / name, [tuple(r) for r in rows])
    return lay.finish([c["name"] for c in cats])


#: FathomNet pictures and boxes whose licences allow any use; the rest
#: (much of it CC BY-NC-ND) is left out
FATHOMNET_LICENCES = {"CC0-1.0", "CC-BY-4.0"}


def fathomnet(folder: Path, workers: int = 16, classes: int = 60, min_boxes: int = 300,
              most: int = 60000, page: int = 40, limit: int | None = None) -> Path:
    """Sea life (FathomNet), only pictures that are CC0 or CC BY 4.0 with every
    box CC0 or CC BY 4.0: the whole catalogue is read once (``catalog.jsonl``,
    resumable), the ``classes`` concepts with the most boxes (at least
    ``min_boxes``) are kept with the pictures holding only those, at most
    ``most`` of them. ``credits.json`` lists each picture's licence and
    address, for the credit CC BY asks."""
    import concurrent.futures
    import time
    import urllib.request

    catalog, state = folder / "catalog.jsonl", folder / "catalog.offset"

    def read(offset: int, size: int) -> list | None:
        """A page of the catalogue; None when the answer came cut short (the
        server cuts long ones) or not at all."""
        body = json.dumps({"limit": size, "offset": offset}).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "easydetect"}
        try:
            request = urllib.request.Request(URLS["fathomnet"], data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=300) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001  (a smaller page, or later)
            return None

    if not (folder / "catalog.done").exists():
        offset = int(state.read_text()) if state.exists() else 0
        size, misses = page, 0
        with open(catalog, "a") as out:
            while True:
                got = read(offset, size)
                if got is None:
                    if size > 1:                       # cut short: ask for less
                        size = max(1, size // 2)
                        continue
                    misses += 1
                    if misses >= 8:                    # one picture it never sends whole
                        print(f"fathomnet: picture {offset} would not come; skipped",
                              flush=True)
                        offset, misses = offset + 1, 0
                        continue
                    time.sleep(30 * misses)
                    continue
                misses = 0
                if not got or (limit and offset >= limit):
                    break
                for im in got:
                    boxes = im.get("boundingBoxes") or []
                    if (im.get("valid", True) and im.get("imageLicense") in FATHOMNET_LICENCES
                            and boxes and all(b.get("annotationLicense") in FATHOMNET_LICENCES
                                              and not b.get("groupOf") for b in boxes)):
                        out.write(json.dumps({
                            "uuid": im["uuid"], "url": im["url"], "w": im["width"],
                            "h": im["height"], "licence": im["imageLicense"],
                            "boxes": [[b["concept"], b["x"], b["y"], b["width"], b["height"]]
                                      for b in boxes]}) + "\n")
                out.flush()
                before, offset = offset, offset + len(got)
                state.write_text(str(offset))
                if offset // 10000 != before // 10000:
                    print(f"fathomnet: {offset} pictures read", flush=True)
                size = min(page, size * 2)             # back up after a short one
        (folder / "catalog.done").touch()
    entries = {}
    for line in catalog.read_text().splitlines():
        e = json.loads(line)
        entries[e["uuid"]] = e
    count = defaultdict(int)
    for e in entries.values():
        for b in e["boxes"]:
            count[b[0]] += 1
    names = sorted(c for c in sorted(count, key=count.get, reverse=True)[:classes]
                   if count[c] >= min_boxes)
    index = {c: i for i, c in enumerate(names)}
    # a picture with a box of a concept left out would teach that animal as absent
    keep = sorted(u for u, e in entries.items() if all(b[0] in index for b in e["boxes"]))
    random.Random(0).shuffle(keep)
    keep = keep[:most]
    src = folder / "src"
    src.mkdir(exist_ok=True)

    def get(u: str) -> None:
        e = entries[u]
        dest = src / f"{u}{Path(urllib.parse.urlparse(e['url']).path).suffix.lower() or '.jpg'}"
        if not dest.exists():
            _get(e["url"], dest)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(get, keep), 1):
            if i % 2000 == 0 or i == len(keep):
                print(f"fathomnet: {i}/{len(keep)} pictures tried", flush=True)
    files = {p.stem: p for p in src.iterdir() if not p.name.endswith(".part")}
    val = set(keep[: len(keep) // 10])
    lay = Layout(folder)
    credits = []
    for u in sorted(keep):
        e = entries[u]
        if u not in files:
            continue
        w, h = float(e["w"]), float(e["h"])
        rows = [r for c, x, y, bw, bh in e["boxes"]
                for r in [box(index[c], x, y, x + bw, y + bh, w, h)] if r]
        lay.add("val" if u in val else "train", files[u], rows, stem=u)
        credits.append({"uuid": u, "licence": e["licence"], "url": e["url"]})
    (folder / "credits.json").write_text(json.dumps(credits))
    return lay.finish(names)


def _split(keys: list, share: float = 0.1) -> set:
    """The val part of ``keys``: a tenth, the same every time (seed 0)."""
    keys = sorted(keys)
    random.Random(0).shuffle(keys)
    return set(keys[: max(1, round(len(keys) * share))])


def coco_text(folder: Path, coco: Path) -> Path:
    """Text in COCO's pictures (COCO-Text v2, CC BY 4.0): every word boxed,
    legible or not, as one class; its own train and val sets."""
    src = _unpack(URLS["coco-text"], folder / "cocotext.v2.zip", folder / "src")
    ann = json.loads(next(src.rglob("cocotext.v2.json")).read_text())
    per_image = defaultdict(list)
    for a in ann["anns"].values():
        per_image[a["image_id"]].append(a)
    lay = Layout(folder)
    for im in ann["imgs"].values():
        path = next((p for p in (coco / "images" / part / f"{im['id']:012d}.jpg"
                                 for part in ("train2017", "val2017")) if p.exists()), None)
        if path is None:
            continue
        w, h = float(im["width"]), float(im["height"])
        rows = [r for a in per_image[im["id"]]
                for r in [box(0, a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                              a["bbox"][1] + a["bbox"][3], w, h)] if r]
        lay.add("val" if im["set"] == "val" else "train", path, rows)
    return lay.finish(["text"])


def ena24(folder: Path) -> Path:
    """Animals in camera-trap pictures of eastern North America (LILA BC,
    CDLA-Permissive): 23 classes, every picture boxed."""
    images = _unpack(URLS["ena24-images"], folder / "ena24.zip", folder / "src")
    ann = json.loads(_fetch(URLS["ena24-labels"], folder / "ena24.json").read_text())
    files = {p.name: p for p in images.rglob("*.jpg")}
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}
    val = _split([im["id"] for im in ann["images"]])
    lay = Layout(folder)
    for split in ("train", "val"):
        part = {"images": [im for im in ann["images"] if (im["id"] in val) == (split == "val")],
                "annotations": ann["annotations"]}
        _coco_like(lay, split, part, images, index,
                   path_of=lambda im: files.get(Path(im["file_name"]).name, Path("/nonexistent")))
    return lay.finish([c["name"] for c in cats])


def boxes_on_bees(folder: Path) -> Path:
    """Bees, with and without pollen, at hive entrances (LILA BC,
    CDLA-Permissive), from Pascal VOC files."""
    src = _unpack(URLS["boxes-on-bees"], folder / "boxes-on-bees-2021.zip", folder / "src")
    xmls = sorted(src.rglob("*.xml"))
    parsed = {}
    for x in xmls:
        root = ET.parse(x).getroot()
        objects = [((o.findtext("name") or "").strip(),
                    [float(o.find("bndbox").findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
                   for o in root.iter("object") if o.find("bndbox") is not None]
        w, h = float(root.findtext("size/width") or 0), float(root.findtext("size/height") or 0)
        parsed[x] = (w, h, objects)
    names = sorted({n for _, _, objs in parsed.values() for n, _ in objs if n})
    val = _split([x.stem for x in xmls])
    lay = Layout(folder)
    for x, (w, h, objs) in parsed.items():
        picture = next((p for p in (x.with_suffix(e) for e in (".jpg", ".JPG", ".png", ".jpeg"))
                        if p.exists()), None)
        if picture is None or not w or not h:
            continue
        rows = [r for n, b in objs if n in names for r in [box(names.index(n), *b, w, h)] if r]
        lay.add("val" if x.stem in val else "train", picture, rows)
    return lay.finish(names)


def _outlines(labels, image_id: int, start: int) -> list[dict]:
    """COCO annotations (polygon and box) for each object of an instance label
    picture (0 background, one number an object)."""
    import cv2
    import numpy as np

    anns = []
    for k, value in enumerate(v for v in np.unique(labels) if v):
        mask = (labels == value).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]
        if not polys:
            continue
        ys, xs = np.nonzero(mask)
        x0, y0 = float(xs.min()), float(ys.min())
        anns.append({"id": start + k, "image_id": image_id, "category_id": 0, "iscrowd": 0,
                     "segmentation": polys, "area": float(mask.sum()),
                     "bbox": [x0, y0, float(xs.max()) + 1 - x0, float(ys.max()) + 1 - y0]})
    return anns


def _instances(folder: Path, pictures: list[tuple[Path, object]], name: str,
               val: set | None = None) -> Path:
    """A one-class layout from pictures and their instance labels, and the same
    objects as COCO polygons in ``seg-{train,val}.json`` for the mask model;
    ``val``: the stems of the val pictures (a tenth, by default)."""
    val = val if val is not None else _split([p.stem for p, _ in pictures])
    seg = {s: {"images": [], "annotations": [], "categories": [{"id": 0, "name": name}]}
           for s in ("train", "val")}
    lay = Layout(folder)
    next_id = 1
    for i, (picture, labels_of) in enumerate(sorted(pictures, key=lambda t: str(t[0]))):
        labels = labels_of() if callable(labels_of) else labels_of
        if labels is None:
            continue
        h, w = labels.shape[:2]
        split = "val" if picture.stem in val else "train"
        anns = _outlines(labels, i, next_id)
        next_id += len(anns) + 1
        rows = [r for a in anns for r in [box(0, a["bbox"][0], a["bbox"][1],
                                              a["bbox"][0] + a["bbox"][2],
                                              a["bbox"][1] + a["bbox"][3], w, h)] if r]
        lay.add(split, picture, rows, stem=f"{i:05d}")
        seg[split]["images"].append({"id": i, "width": w, "height": h,
                                     "file_name": picture.name, "path": str(picture.resolve())})
        seg[split]["annotations"].extend(anns)
    for split, d in seg.items():
        (folder / f"seg-{split}.json").write_text(json.dumps(d))
    return lay.finish([name])


def bbbc038(folder: Path) -> Path:
    """Nuclei in microscope pictures of many kinds (BBBC038, the 2018 Data
    Science Bowl, CC0): one mask a nucleus, as boxes and polygons."""
    import cv2
    import numpy as np

    src = _unpack(URLS["bbbc038"], folder / "stage1_train.zip", folder / "src")
    pictures = []
    for image_dir in sorted(p for p in src.rglob("images") if p.is_dir()):
        picture = next(image_dir.glob("*.png"), None)
        masks = sorted((image_dir.parent / "masks").glob("*.png"))
        if picture is None or not masks:
            continue

        def labels_of(masks=masks):
            first = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
            if first is None:
                return None
            out = np.zeros(first.shape, np.int32)
            for k, m in enumerate(masks, 1):
                one = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
                if one is not None:
                    out[one > 0] = k
            return out

        pictures.append((picture, labels_of))
    return _instances(folder, pictures, "nucleus")


def nuinsseg(folder: Path) -> Path:
    """Nuclei in H&E tissue of 31 human and mouse organs (NuInsSeg, CC BY
    4.0): its instance label masks as boxes and polygons."""
    import cv2

    src = _unpack(URLS["nuinsseg"], folder / "NuInsSeg.zip", folder / "src")
    pictures = []
    for picture in sorted(src.rglob("tissue images/*.png")):
        found = [p for p in (picture.parent.parent / "label masks").glob(picture.stem + ".*")]
        if not found:
            continue
        pictures.append((picture, lambda f=found[0]: cv2.imread(str(f), cv2.IMREAD_UNCHANGED)))
    return _instances(folder, pictures, "nucleus")


def _voc(folder: Path, xmls: list[Path], image_of, val: set) -> Path:
    """A layout from Pascal VOC files: classes from the files themselves."""
    parsed = {}
    for x in xmls:
        root = ET.parse(x).getroot()
        w, h = float(root.findtext("size/width") or 0), float(root.findtext("size/height") or 0)
        parsed[x] = (w, h, [((o.findtext("name") or "").strip(),
                             [float(o.find("bndbox").findtext(k)) for k in
                              ("xmin", "ymin", "xmax", "ymax")])
                            for o in root.iter("object") if o.find("bndbox") is not None])
    names = sorted({n for _, _, objs in parsed.values() for n, _ in objs if n})
    lay = Layout(folder)
    for x, (w, h, objs) in parsed.items():
        picture = image_of(x)
        if picture is None or not picture.exists() or not w or not h:
            continue
        rows = [r for n, b in objs if n in names for r in [box(names.index(n), *b, w, h)] if r]
        lay.add("val" if x.stem in val else "train", picture, rows)
    return lay.finish(names)


def _seg_json(folder: Path, splits: dict) -> None:
    """``seg-{split}.json`` beside a layout, for the mask model: ``splits``
    {split: (images, annotations, path_of)} in COCO terms, polygons only."""
    for split, (images, anns, path_of) in splits.items():
        keep = [dict(im, path=str(path_of(im).resolve())) for im in images if path_of(im).exists()]
        ids = {im["id"] for im in keep}
        out = [a for a in anns if a["image_id"] in ids and isinstance(a.get("segmentation"), list)
               and any(len(p) >= 6 for p in a["segmentation"])]
        cats = sorted({a["category_id"] for a in anns})
        (folder / f"seg-{split}.json").write_text(json.dumps(
            {"images": keep, "annotations": out,
             "categories": [{"id": c, "name": str(c)} for c in cats]}))


def _seg_named(folder: Path, cats: list[dict]) -> None:
    """The mask files' categories, named and in the layout's class order."""
    for split in ("train", "val"):
        f = folder / f"seg-{split}.json"
        if f.exists():
            d = json.loads(f.read_text())
            index = {c["id"]: i for i, c in enumerate(cats)}
            d["annotations"] = [dict(a, category_id=index[a["category_id"]])
                                for a in d["annotations"] if a["category_id"] in index]
            d["categories"] = [{"id": i, "name": c["name"]} for i, c in enumerate(cats)]
            f.write_text(json.dumps(d))


def pped(folder: Path) -> Path:
    """Protective equipment in a chemical plant, photographed by its authors
    (PPED, CC BY 4.0): Pascal VOC files and their own split."""
    src = _unpack(URLS["pped"], folder / "PPED_data.zip", folder / "src")
    base = next(p.parent for p in src.rglob("Annotations") if p.is_dir())
    lists = base / "ImageSets" / "Main"
    xmls = sorted((base / "Annotations").glob("*.xml"))
    held = set()
    for name in ("val.txt", "test.txt"):
        if (lists / name).exists():
            held |= set((lists / name).read_text().split())
    val = held or _split([x.stem for x in xmls])
    return _voc(folder, xmls, lambda x: base / "JPEGImages" / f"{x.stem}.jpg", val)


def loco(folder: Path) -> Path:
    """Logistics objects in warehouses (LOCO, TU Munich, CC0): forklifts,
    pallets, pallet trucks, small load carriers, stillages. Val takes whole
    recordings, as their frames are near-copies of each other."""
    images = _unpack(URLS["loco-images"], folder / "dataset.zip", folder / "src")
    ann = json.loads(_fetch(URLS["loco-labels"], folder / "loco-all-v1.json").read_text())
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}

    def path_of(im):
        return images / im["path"].lstrip("/")

    val_runs = _split(sorted({str(Path(im["path"]).parent) for im in ann["images"]}))
    lay = Layout(folder)
    for split in ("train", "val"):
        part = {"images": [im for im in ann["images"]
                           if (str(Path(im["path"]).parent) in val_runs) == (split == "val")],
                "annotations": ann["annotations"]}
        _coco_like(lay, split, part, images, index, path_of=path_of)
    return lay.finish([c["name"] for c in cats])


def msu_apples(folder: Path) -> Path:
    """Apples in an orchard (MSUAppleData, CC BY 4.0): one class, its own
    train and test sets, from VGG Image Annotator files."""
    from PIL import Image

    src = _unpack(URLS["msu-apples"], folder / "MSUAppleData.zip", folder / "src")
    lay = Layout(folder)
    for part, split in (("train", "train"), ("test", "val")):
        notes = next(p for p in src.rglob(f"{part}/annotations.json") if "__MACOSX" not in str(p))
        meta = json.loads(notes.read_text()).get("_via_img_metadata", {})
        for entry in meta.values():
            picture = notes.parent / entry["filename"]
            if not picture.exists():
                continue
            with Image.open(picture) as im:
                w, h = im.size
            rows = []
            for region in entry.get("regions", []):
                a = region.get("shape_attributes", {})
                if a.get("name") == "rect":
                    r = box(0, a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"], w, h)
                    if r:
                        rows.append(r)
            lay.add(split, picture, rows)
    return lay.finish(["apple"])


#: SwinePose names no lines between its keypoints: these, to draw a pig
SWINE_SKELETON = [("nose", "front"), ("front", "hip"),
                  ("front", "right_elbow"), ("right_elbow", "right_carpal"),
                  ("right_carpal", "front_right_toe"), ("front", "left_elbow"),
                  ("left_elbow", "left_carpal"), ("left_carpal", "front_left_toe"),
                  ("hip", "right_stifle"), ("right_stifle", "right_tarsal"),
                  ("right_tarsal", "rear_right_toe"), ("hip", "left_stifle"),
                  ("left_stifle", "left_tarsal"), ("left_tarsal", "rear_left_toe")]


def _flip_pairs(names: list[str]) -> list[int]:
    """Each keypoint's mirror image: left and right swapped in its name."""
    def mirror(n):
        return (n.replace("left", "\0").replace("right", "left").replace("\0", "right"))
    return [names.index(mirror(n)) if mirror(n) in names else i for i, n in enumerate(names)]


def swinepose(folder: Path) -> Path:
    """Pigs walking, seen from the side (SwinePose v2, CC BY 4.0): boxes, and
    15 keypoints each in ``pose/`` with ``pose.yaml`` for the keypoint model.
    Its pig-wise split: no pig is in both."""
    src = _unpack(URLS["swinepose"], folder / "SwinePose_v2.0.zip", folder / "src")
    notes = {split: json.loads(next(src.rglob(f"{name}_pigsplit.json")).read_text())
             for split, name in (("train", "train"), ("val", "test"))}
    images_dir = next(p for p in src.rglob("images") if p.is_dir())
    cat = notes["train"]["categories"][0]
    k = len(cat["keypoints"])
    lay = Layout(folder)
    pose = folder / "pose"
    for split, ann in notes.items():
        _coco_like(lay, split, ann, images_dir, {cat["id"]: 0})
        for d in ("images", "labels"):
            shutil.rmtree(pose / d / split, ignore_errors=True)
            (pose / d / split).mkdir(parents=True)
        per_image = defaultdict(list)
        for a in ann["annotations"]:
            per_image[a["image_id"]].append(a)
        for im in ann["images"]:
            picture = images_dir / im["file_name"]
            if not picture.exists():
                continue
            w, h = float(im["width"]), float(im["height"])
            lines = []
            for a in per_image[im["id"]]:
                r = box(0, a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                        a["bbox"][1] + a["bbox"][3], w, h)
                if not r:
                    continue
                kp = a["keypoints"]
                pts = " ".join(f"{kp[3 * i] / w:.6f} {kp[3 * i + 1] / h:.6f} {int(kp[3 * i + 2])}"
                               for i in range(k))
                lines.append(f"0 {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {pts}")
            os.symlink(picture.resolve(), pose / "images" / split / picture.name)
            (pose / "labels" / split / f"{picture.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
    names = list(cat["keypoints"])
    (folder / "pose.yaml").write_text(yaml.safe_dump({
        "path": str(pose.resolve()), "train": "images/train", "val": "images/val",
        "names": {0: cat["name"]}, "kpt_shape": [k, 3], "flip_idx": _flip_pairs(names),
        "kpt_names": names,
        "skeleton": [[names.index(a), names.index(b)] for a, b in SWINE_SKELETON],
        "oks_sigmas": None}, allow_unicode=True, sort_keys=False))
    return lay.finish([cat["name"]])


def pose_yaml(key: str, root: Path) -> Path:
    """The keypoint data.yaml of a hub dataset that trains a keypoint model."""
    if key == "ap10k":
        return ap10k_pose_yaml(root / key)
    path = root / key / "pose.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{key}: no keypoints laid out ({path})")
    return path


def bbbc039(folder: Path) -> Path:
    """Nuclei in fluorescence pictures (BBBC039, CC0): its 16-bit pictures
    stretched to 8 bits, each nucleus from its mask (drawn in four colours so
    that neighbours differ), and its own split (validation and test hold out)."""
    import cv2
    import numpy as np

    images = _unpack(URLS["bbbc039-images"], folder / "images.zip", folder / "src-images")
    masks = _unpack(URLS["bbbc039-masks"], folder / "masks.zip", folder / "src-masks")
    meta = _unpack(URLS["bbbc039-metadata"], folder / "metadata.zip", folder / "src-meta")
    held = set()
    for name in ("validation.txt", "test.txt"):
        held |= {Path(n).stem for n in next(p for p in meta.rglob(name)
                                             if "__MACOSX" not in p.parts).read_text().split()}
    png = folder / "png"
    png.mkdir(exist_ok=True)
    pictures = []
    def real(paths):                    # not macOS's ._ copies in __MACOSX
        return (p for p in paths if "__MACOSX" not in p.parts and not p.name.startswith("._"))

    for mask in sorted(real(masks.rglob("*.png"))):
        tif = next(real(images.rglob(f"{mask.stem}.tif")), None)
        if tif is None:
            continue
        out = png / f"{mask.stem}.png"
        if not out.exists():
            from PIL import Image

            raw = cv2.imread(str(tif), cv2.IMREAD_UNCHANGED)
            if raw is None:                    # a TIFF OpenCV does not read
                with Image.open(tif) as im:
                    raw = np.asarray(im)
            raw = raw.astype(np.float32)
            lo, hi = np.percentile(raw, (0.5, 99.8))
            stretched = np.clip((raw - lo) / max(hi - lo, 1) * 255, 0, 255)
            cv2.imwrite(str(out), stretched.astype(np.uint8))

        def labels_of(mask=mask):
            from PIL import Image

            colour = np.asarray(Image.open(mask))[..., 0]
            labels = np.zeros(colour.shape, np.int32)
            for v in (c for c in np.unique(colour) if c):
                n, parts = cv2.connectedComponents((colour == v).astype(np.uint8))
                labels[parts > 0] = parts[parts > 0] + labels.max()
            return labels

        pictures.append((out, labels_of))
    return _instances(folder, pictures, "nucleus", val=held)


def noaa_psnf(folder: Path) -> Path:
    """Fish and crabs in Puget Sound eelgrass and shores (NOAA, LILA BC,
    CDLA-Permissive). Pictures with a box of fish-or-crab or unknown are left
    out (they would teach the animal as absent); a tenth of the empty ones stay."""
    images = _unpack(URLS["noaa-psnf-images"], folder / "images.zip", folder / "src")
    labels = _unpack(URLS["noaa-psnf-labels"], folder / "labels.zip", folder / "src-labels")
    ann = json.loads(next(labels.rglob("*.json")).read_text())
    named = {c["id"]: c["name"] for c in ann["categories"]}
    keep_cats = [i for i, n in sorted(named.items()) if n in ("fish", "crab")]
    index = {c: i for i, c in enumerate(keep_cats)}
    per_image = defaultdict(list)
    for a in ann["annotations"]:
        per_image[a["image_id"]].append(a)
    files = {p.name: p for p in images.rglob("*.jpg")}
    chosen = []
    for im in ann["images"]:
        anns = per_image[im["id"]]
        if any(named.get(a["category_id"]) in ("fish_or_crab", "unknown") for a in anns):
            continue
        boxed = [a for a in anns if a.get("bbox") and a["category_id"] in index]
        if not boxed and random.Random(im["id"]).random() >= 0.1:
            continue
        chosen.append(im)
    val = _split([im["id"] for im in chosen])
    lay = Layout(folder)
    for split in ("train", "val"):
        part = {"images": [im for im in chosen if (im["id"] in val) == (split == "val")],
                "annotations": ann["annotations"]}
        _coco_like(lay, split, part, images, index,
                   path_of=lambda im: files.get(Path(im["file_name"]).name, Path("/nonexistent")))
    return lay.finish([named[c] for c in keep_cats])


def _coco_pair(folder: Path, splits: dict, names_from: dict, path_of) -> Path:
    """A layout (and the mask files) from COCO files {split: annotations}."""
    cats = sorted(names_from["categories"], key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}
    lay = Layout(folder)
    for split, ann in splits.items():
        _coco_like(lay, split, ann, folder, index, path_of=path_of)
    _seg_json(folder, {split: (ann["images"], ann["annotations"], path_of)
                       for split, ann in splits.items()})
    _seg_named(folder, cats)
    return lay.finish([c["name"] for c in cats])


def armbench(folder: Path) -> Path:
    """Products in warehouse totes (Amazon ARMBench, CC BY 4.0): its
    Mix-Object-Tote set, boxes and masks, its own train and validation."""
    src = _unpack(URLS["armbench"], folder / "armbench-segmentation-0.1.tar.gz", folder / "src")
    tote = next(p for p in src.rglob("mix-object-tote") if p.is_dir())
    jsons = sorted(tote.rglob("*.json"))

    def find(word):
        named = [p for p in jsons if word in p.name.lower()]
        for p in sorted(named, key=lambda p: "coco" not in str(p).lower()):
            ann = json.loads(p.read_text())
            if isinstance(ann, dict) and "images" in ann and "annotations" in ann:
                return ann
        raise FileNotFoundError(f"armbench: no {word} COCO file in {tote} "
                                f"(looked at {[str(p.relative_to(tote)) for p in named][:8]})")

    splits = {"train": find("train"), "val": find("val")}
    files = {}
    for p in tote.rglob("*.jpg"):
        files.setdefault(p.name, p)
        files.setdefault(p.stem, p)

    def path_of(im):
        name = Path(im["file_name"])
        return files.get(name.name) or files.get(name.stem) or Path("/nonexistent")

    return _coco_pair(folder, splits, splits["train"], path_of)


def artf_clothes(folder: Path) -> Path:
    """Clothes on tables and floors (aRTF Clothes, Ghent University, CC BY
    4.0): T-shirts, towels, shorts and boxer shorts, boxes and masks, in its
    own resized 512 x 256 release and train / val split."""
    src = _unpack(URLS["artf-clothes"], folder / "artf.zip", folder / "src")
    for inner in sorted(src.rglob("*.zip")):                # one zip a split inside
        inner.with_name(inner.name + ".ok").touch()         # here already: nothing to fetch
        _unpack(str(inner), inner, inner.with_suffix(""), keep=True)
    notes = {p.parent.name.split("_")[0].replace("artf-", ""): p
             for p in src.rglob("annotations.json")}
    splits = {"train": json.loads(notes["train"].read_text()),
              "val": json.loads(notes["val"].read_text())}
    roots = {split: notes[split].parent for split in ("train", "val")}
    path_of = {}
    for split, ann in splits.items():
        for im in ann["images"]:
            path_of[id(im)] = roots[split] / im["file_name"]
    return _coco_pair(folder, splits, splits["train"], lambda im: path_of.get(id(im), Path("/x")))


def everwatch(folder: Path) -> Path:
    """Wading birds from the air over the Everglades (EverWatch, CC0): its
    1500 px crops, its own train and validation."""
    src = _unpack(URLS["everwatch"], folder / "everwatch-dataset.zip", folder / "src")
    base = next(p.parent for p in src.rglob("train_annotations.json"))
    splits = {"train": json.loads((base / "train_annotations.json").read_text()),
              "val": json.loads((base / "validation_annotations.json").read_text())}
    cats = sorted(splits["train"]["categories"], key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}
    lay = Layout(folder)
    for split, ann in splits.items():
        _coco_like(lay, split, ann, base, index, path_of=lambda im: base / im["file_name"])
    return lay.finish([c["name"] for c in cats])


def visa(folder: Path) -> Path:
    """Defects on twelve kinds of product (Amazon VisA, CC BY 4.0): each
    defect region of its masks a box and an outline, one class. Its
    supervised split (2cls_highshot: defects in train and test); its 1-class
    split has none in train, so without that file a tenth of the pictures, at
    random, are val. A fifth of the flawless training pictures are kept."""
    import csv

    import cv2
    import numpy as np

    src = _unpack(URLS["visa"], folder / "VisA_20220922.tar", folder / "src")
    table = next(iter(sorted(src.rglob("2cls_highshot.csv"))), None)
    own = table is None
    table = table or next(src.rglob("1cls.csv"))
    root = table.parent.parent
    pictures, val = [], set()
    with open(table, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            picture = root / row["image"]
            if not picture.exists():
                continue
            flawed = row.get("label", "").strip().lower() == "anomaly" and row.get("mask")
            training = own or row["split"].strip() == "train"
            if training and not flawed and random.Random(i).random() >= 0.2:
                continue
            mask = root / row["mask"] if flawed else None
            stem = f"{row['object']}_{picture.stem}"

            def labels_of(mask=mask, picture=picture):
                if mask is None:
                    h, w = cv2.imread(str(picture), cv2.IMREAD_GRAYSCALE).shape
                    return np.zeros((h, w), np.int32)
                m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                return None if m is None else cv2.connectedComponents((m > 0).astype("uint8"))[1]

            link = folder / "named" / f"{stem}{picture.suffix.lower()}"
            link.parent.mkdir(exist_ok=True)
            if not link.exists():
                os.symlink(picture.resolve(), link)
            pictures.append((link, labels_of))
            if not own and row["split"].strip() != "train":
                val.add(link.stem)
    if own:
        val = _split([link.stem for link, _ in pictures])
    return _instances(folder, pictures, "defect", val=val)


def rf100vl_all(root: Path) -> Path:
    """Every RF100-VL dataset of ``RF100VL_SETS`` laid out under
    ``<root>/rf100vl/<name>/``, from its authors' parquet files (42 GB, the
    pictures inside): a shard at a time, downloaded, written out per dataset
    and split, then deleted. Train trains, valid picks the best epoch, test
    scores (``score.yaml``), as Roboflow 100's do."""
    import urllib.request

    import pyarrow.parquet as pq

    base = root / "rf100vl"
    if (base / ".ready").exists():
        return base
    meta = json.loads(_fetch(f"{RF100VL_URL}/dataset_metadata.json",
                             base / "dataset_metadata.json").read_text())
    request = urllib.request.Request(RF100VL_TREE, headers={"User-Agent": "easydetect"})
    with urllib.request.urlopen(request, timeout=120) as r:
        shards = sorted(e["path"] for e in json.loads(r.read()) if e["path"].endswith(".parquet"))
    import time

    wanted = set(RF100VL_SETS.values())
    work = base / ".shards"
    missed = []
    for n, shard in enumerate(shards, 1):
        name = Path(shard).name
        done = work / f"{name}.done"
        if done.exists():
            continue
        split = name.split("-")[0]                      # train, valid or test
        local = None
        for attempt in range(6):        # Hugging Face turns a long run of downloads away a while
            try:
                local = _fetch(f"{RF100VL_URL}/{shard}", work / name)
                break
            except subprocess.CalledProcessError:
                wait = 300 * (attempt + 1)
                print(f"rf100vl: {name} refused; again in {wait // 60} minutes", flush=True)
                time.sleep(wait)
        if local is None:
            missed.append(name)
            continue
        pf = pq.ParquetFile(local)
        kept = 0
        outlined = defaultdict(lambda: [0, 0])          # dataset: [boxes, with a polygon]
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g).to_pylist():
                if row["dataset_name"] not in wanted:
                    continue
                folder = base / row["dataset_name"]
                images, labels = folder / "images" / split, folder / "labels" / split
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                stem = row["image_id"]
                suffix = Path(row["file_name"]).suffix.lower() or ".jpg"
                picture = images / f"{stem}{suffix}"
                if not picture.exists():
                    tmp = picture.with_name(picture.name + ".part")
                    tmp.write_bytes(row["image"]["bytes"])
                    tmp.rename(picture)
                w, h, a = float(row["width"]), float(row["height"]), row["annotations"]
                rows = [r for c, (x, y, bw, bh) in zip(a["category_id"], a["bbox"], strict=True)
                        for r in [box(int(c), x, y, x + bw, y + bh, w, h)] if r]
                count = outlined[row["dataset_name"]]
                count[0] += len(a["bbox"])
                count[1] += sum(1 for p in (a.get("segmentation") or [])
                                if p and any(len(q) >= 6 for q in p))
                lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for c, cx, cy, bw, bh in rows]
                (labels / f"{stem}.txt").write_text("".join(lines))
                kept += 1
        # which datasets outline their objects too: those could get a mask model
        with open(work / "outlined.jsonl", "a") as f:
            f.write(json.dumps({"shard": name, "outlined": outlined}) + "\n")
        done.touch()
        local.unlink(missing_ok=True)
        local.with_name(local.name + ".ok").unlink(missing_ok=True)
        print(f"rf100vl: shard {n}/{len(shards)} ({name}): {kept} pictures", flush=True)
    if missed:                          # laid out only whole: the next try fetches these
        raise RuntimeError(f"rf100vl: {len(missed)} files would not come ({', '.join(missed)})")
    by_name = {d["name"]: i for i, d in meta["datasets"].items()}
    for name in sorted(wanted):
        folder = base / name
        if name not in by_name or not folder.exists():
            print(f"rf100vl: {name} not in the files; left out", flush=True)
            continue
        cats = meta["categories"][by_name[name]]
        cats = (cats.get("train") or next(iter(cats.values())))["categories"]
        names = {int(c["id"]): str(c["name"]) for c in cats}
        names = {i: names.get(i, f"class{i}") for i in range(max(names) + 1)}
        have = {sp for sp in ("train", "valid", "test") if (folder / "images" / sp).is_dir()}
        val = "valid" if "valid" in have else "test"
        cfg = {"path": str(folder.resolve()), "train": "images/train", "val": f"images/{val}",
               "names": names}
        (folder / "data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        cfg["val"] = "images/test" if "test" in have else f"images/{val}"
        (folder / "score.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        (folder / ".ready").touch()
    (base / ".ready").touch()
    return base


def rf100(name: str, root: Path) -> tuple[Path, Path]:
    """(train.yaml, score.yaml) of the whole set, laid out by rf100.py."""
    sys.path.insert(0, str(HERE))
    from rf100 import prepare

    big = 10 ** 9
    train, score, _ = prepare(name, root / "rf100", {"train": big, "val": big, "test": big})
    return train, score


MAKERS = {"bccd": bccd, "plantdoc": plantdoc, "gwhd": gwhd, "taco": taco,
          "doclaynet": doclaynet, "caltech-camera-traps": caltech_camera_traps,
          "fashionpedia": fashionpedia, "ap10k": ap10k, "izembek": izembek,
          "fathomnet": fathomnet, "ena24": ena24, "boxes-on-bees": boxes_on_bees,
          "bbbc038": bbbc038, "nuinsseg": nuinsseg, "pped": pped, "loco": loco,
          "msu-apples": msu_apples, "swinepose": swinepose, "bbbc039": bbbc039,
          "noaa-psnf": noaa_psnf, "armbench": armbench, "artf-clothes": artf_clothes,
          "everwatch": everwatch, "visa": visa}


#: what a prefetch fetches, in two lanes at once: the quick ones (training
#: never waits long for the next) and the big downloads, which take hours
QUICK = ["bccd", "taco", "plantdoc", *(f"rf100-{n}" for n in RF100_SETS)]
BIG = ["gwhd", "doclaynet", "caltech-camera-traps", "lvis"]
ORDER = QUICK + BIG
#: added after the first run: fetched on their own (``--new``) until they train;
#: the big downloads take the second lane
#: the second batch: boxes on more subjects, and nuclei with masks
MORE = ["coco-text", "bbbc038", "nuinsseg", "ena24", "boxes-on-bees",
        # the third: protective equipment, logistics, teaching sets, masks and keypoints
        "pped", "loco", "msu-apples", "swinepose", "bbbc039", "artf-clothes", "visa",
        "noaa-psnf", "everwatch", "armbench"]
NEW = ["ap10k", "fashionpedia", "izembek", "fathomnet", *RF100VL_SETS, *MORE]
QUICK += ["ap10k", "coco-text", "bbbc038", "nuinsseg", "msu-apples", "swinepose", "bbbc039",
          "artf-clothes"]
BIG += ["fashionpedia", "izembek", "fathomnet", *RF100VL_SETS, "ena24", "boxes-on-bees", "pped",
        "loco", "visa", "noaa-psnf", "everwatch", "armbench"]
ORDER = QUICK + BIG


def _lock_path(key: str, root: Path) -> Path:
    return root / ".locks" / f"{key}.lock"


def _folder(key: str, root: Path) -> Path:
    if key.startswith("rf100vl-"):
        return root / "rf100vl" / RF100VL_SETS[key]
    if key.startswith("rf100-"):
        return root / "rf100" / key[len("rf100-"):]
    return root / key


def is_ready(key: str, root: Path) -> bool:
    """Laid out already: training can start on it at once."""
    if key.startswith("rf100vl-"):
        return (_folder(key, root) / ".ready").exists()
    if key.startswith("rf100-"):
        return (root / "rf100" / key[len("rf100-"):] / "facts.json").exists()
    return (root / key / ".ready").exists()


def prefetch_running(root: Path) -> bool:
    """A prefetch is under way: what it has not laid out yet, it will."""
    return is_busy("prefetch", root)


def boxes_per_picture(key: str, root: Path) -> float:
    """Mean boxes a training picture (0 before it is laid out). Pictures with
    hundreds of boxes (cells, circuit parts) need far more GPU memory."""
    labels = list((_folder(key, root) / "labels" / "train").glob("*.txt"))
    if not labels:
        return 0.0
    return sum(sum(1 for line in f.read_text().splitlines() if line.strip())
               for f in labels) / len(labels)


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
    if key.startswith("rf100vl-"):                  # one download lays out all of them
        with _locked(_lock_path("rf100vl", root), "rf100vl"):
            rf100vl_all(root)
        folder = _folder(key, root)
        if not (folder / ".ready").exists():
            raise FileNotFoundError(f"{key}: not in RF100-VL's files")
        return folder / "data.yaml", folder / "score.yaml"
    with _locked(_lock_path(key, root), key):
        if key.startswith("rf100-"):
            return rf100(key[len("rf100-"):], root)
        folder = root / key
        if (folder / ".ready").exists():
            return folder / "data.yaml", None
        folder.mkdir(parents=True, exist_ok=True)
        if key == "lvis":
            return lvis(folder, coco), None
        if key == "coco-text":
            return coco_text(folder, coco), None
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


def drop_unreadable(key: str, root: Path) -> int:
    """Take out of a laid-out dataset the pictures OpenCV cannot read (a
    damaged file in the source): their links and labels. How many went."""
    import cv2

    folder = _folder(key, root)
    gone = 0
    for split in ("train", "val"):
        for link in sorted((folder / "images" / split).iterdir()):
            if cv2.imread(str(link)) is not None:
                continue
            link.unlink()
            for ext in (".txt", ".unknown"):
                (folder / "labels" / split / f"{link.stem}{ext}").unlink(missing_ok=True)
            gone += 1
    print(f"{key}: {gone} unreadable pictures taken out", flush=True)
    return gone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("keys", nargs="*")
    parser.add_argument("--all", action="store_true", help="every dataset, in training order")
    parser.add_argument("--new", action="store_true", help="the datasets added after the first run")
    parser.add_argument("--skip", nargs="*", default=[], help="leave these out")
    parser.add_argument("--prefetch", action="store_true",
                        help="one at a time behind a lock, going on past a failure")
    parser.add_argument("--drop-unreadable", action="store_true",
                        help="take the pictures OpenCV cannot read out of these laid-out datasets")
    parser.add_argument("--root", default="~/datasets/hub")
    parser.add_argument("--coco", default="~/datasets/coco")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    chosen = ORDER if args.all else NEW if args.new else args.keys
    keys = [k for k in chosen if k not in args.skip]
    root, coco = Path(args.root).expanduser(), Path(args.coco).expanduser()
    if args.drop_unreadable:
        for key in keys:
            drop_unreadable(key, root)
        return 0
    if args.prefetch:
        return prefetch(keys, root, coco)
    failed = []
    for key in keys:                            # one failing does not stop the rest
        try:
            print(prepare(key, root, coco), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{key}: not ready ({exc})", flush=True)
            failed.append(key)
    if failed:
        print(f"not ready: {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
