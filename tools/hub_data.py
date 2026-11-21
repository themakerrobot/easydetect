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
              most: int = 60000, page: int = 200, limit: int | None = None) -> Path:
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
    if not (folder / "catalog.done").exists():
        offset = int(state.read_text()) if state.exists() else 0
        with open(catalog, "a") as out:
            while True:
                body = json.dumps({"limit": page, "offset": offset}).encode()
                for n in range(8):
                    try:
                        headers = {"Content-Type": "application/json", "User-Agent": "easydetect"}
                        request = urllib.request.Request(URLS["fathomnet"], data=body,
                                                         headers=headers)
                        with urllib.request.urlopen(request, timeout=300) as r:
                            got = json.loads(r.read())
                        break
                    except Exception:  # noqa: BLE001
                        time.sleep(30 * (n + 1))
                else:
                    raise RuntimeError(f"fathomnet: the catalogue would not answer at {offset}")
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
                offset += len(got)
                state.write_text(str(offset))
                if offset % 10000 < page:
                    print(f"fathomnet: {offset} pictures read", flush=True)
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
          "fathomnet": fathomnet}


#: what a prefetch fetches, in two lanes at once: the quick ones (training
#: never waits long for the next) and the big downloads, which take hours
QUICK = ["bccd", "taco", "plantdoc", *(f"rf100-{n}" for n in RF100_SETS)]
BIG = ["gwhd", "doclaynet", "caltech-camera-traps", "lvis"]
ORDER = QUICK + BIG
#: added after the first run: fetched on their own (``--new``) until they train;
#: the big downloads take the second lane
NEW = ["ap10k", "fashionpedia", "izembek", "fathomnet"]
QUICK += ["ap10k"]
BIG += ["fashionpedia", "izembek", "fathomnet"]
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


def boxes_per_picture(key: str, root: Path) -> float:
    """Mean boxes a training picture (0 before it is laid out). Pictures with
    hundreds of boxes (cells, circuit parts) need far more GPU memory."""
    folder = root / "rf100" / key[len("rf100-"):] if key.startswith("rf100-") else root / key
    labels = list((folder / "labels" / "train").glob("*.txt"))
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
    parser.add_argument("--new", action="store_true", help="the datasets added after the first run")
    parser.add_argument("--skip", nargs="*", default=[], help="leave these out")
    parser.add_argument("--prefetch", action="store_true",
                        help="one at a time behind a lock, going on past a failure")
    parser.add_argument("--root", default="~/datasets/hub")
    parser.add_argument("--coco", default="~/datasets/coco")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    chosen = ORDER if args.all else NEW if args.new else args.keys
    keys = [k for k in chosen if k not in args.skip]
    root, coco = Path(args.root).expanduser(), Path(args.coco).expanduser()
    if args.prefetch:
        return prefetch(keys, root, coco)
    for key in keys:
        print(prepare(key, root, coco), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
