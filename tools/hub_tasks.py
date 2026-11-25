#!/usr/bin/env python3
# Apache-2.0
"""easydetect-hub's segmentation and keypoint models: their data, training and scores.

    python tools/hub_tasks.py seg-data coco          # ~/datasets/hub/coco-seg/
    python tools/hub_tasks.py seg-train coco         # .../coco-seg/run/decoder.onnx
    python tools/hub_tasks.py score-seg --model ed-s-640.onnx \\
        --decoder mask_decoder.onnx --truth ~/datasets/hub/coco-seg/val.json
    python tools/hub_tasks.py score-pose --model ed-s-640.onnx --pose pose-s.onnx

A hub model of another task is the dataset's detector with a second network
beside it, in a folder of its own (``<dataset>/ed-s-640-seg/``,
``<dataset>/ed-s-640-pose/``): MobileSAM's mask decoder fine-tuned on the
dataset's masks, or a keypoint network. Only datasets that label masks or
keypoints get one: the detector's boxes alone teach neither.

``seg-data`` writes ``<root>/<key>-seg/`` from the dataset's own annotations:
polygon labels (one line an object, its parts joined) for a share of the train
split, a few hundred val pictures to pick the best epoch by, and ``val.json``,
the whole val split as COCO-format truth (crowd regions included, which COCO's
evaluation ignores) for ``score-seg``. The splits are the detector's: no
picture it trained on is scored.

Scores are COCO's, through pycocotools: mask mAP50-95 (``segm``) and OKS AP
(``keypoints``), 0-1.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

#: datasets the hub has whose annotations outline objects / place keypoints
SEG = ("coco", "taco", "lvis", "fashionpedia", "bbbc038", "nuinsseg", "bbbc039", "artf-clothes",
       "visa", "armbench")
#: COCO's people take easydetect's keypoint network as it is; the others get
#: one trained on their own keypoints (long_run's hub-pose-train)
POSE = ("coco", "ap10k", "swinepose")
#: train pictures the decoder learns from: it starts from MobileSAM and only
#: its small decoder trains, on embeddings cached at 2 MB a picture
TRAIN_PICTURES = 10000
PICK_PICTURES = 500


# ---------------------------------------------------------------- sources


def _coco(coco: Path):
    cats, splits = None, {}
    for split, part in (("train", "train2017"), ("val", "val2017")):
        ann = json.loads((coco / "annotations" / f"instances_{part}.json").read_text())
        cats = cats or sorted(ann["categories"], key=lambda c: c["id"])
        splits[split] = (ann["images"], ann["annotations"],
                         lambda im, part=part: coco / "images" / part / im["file_name"], None)
    return cats, splits


def _taco(folder: Path):
    """TACO's split as hub_data.py's taco() makes it (a tenth of the ids, shuffled
    with seed 0) and its pictures as fetched there (the longest side 1600)."""
    ann = json.loads((folder / "annotations.json").read_text())
    ids = sorted(im["id"] for im in ann["images"])
    random.Random(0).shuffle(ids)
    val = set(ids[: len(ids) // 10])
    src = folder / "src"

    def path_of(im):
        return src / Path(im["file_name"]).with_suffix(".jpg")

    splits = {split: ([im for im in ann["images"] if (im["id"] in val) == (split == "val")],
                      ann["annotations"], path_of, None) for split in ("train", "val")}
    return sorted(ann["categories"], key=lambda c: c["id"]), splits


def _lvis(folder: Path, coco: Path):
    cats, splits = None, {}
    for split in ("train", "val"):
        ann = json.loads(next((folder / f"src-{split}").rglob("*.json")).read_text())
        cats = cats or sorted(ann["categories"], key=lambda c: c["id"])

        def path_of(im):
            part = "train2017" if "train2017" in im["coco_url"] else "val2017"
            return coco / "images" / part / f"{im['id']:012d}.jpg"

        def unknown_of(im):
            # LVIS outlines only some of the instances of these classes here
            return im.get("not_exhaustive_category_ids", [])

        splits[split] = (ann["images"], ann["annotations"], path_of, unknown_of)
    return cats, splits


def _fashionpedia(folder: Path):
    """Fashionpedia's masks on the pictures hub_data.py's fashionpedia() trains
    on: those whose own licence allows any use."""
    sys.path.insert(0, str(HERE))
    from hub_data import FASHIONPEDIA_PICTURES

    files = {p.name: p for d in ("src-train", "src-val") for p in (folder / d).rglob("*.jpg")}
    cats, splits = None, {}
    for split in ("train", "val"):
        ann = json.loads((folder / f"ann-{split}.json").read_text())
        free = {x["id"] for x in ann["licenses"] if x["name"] in FASHIONPEDIA_PICTURES}
        cats = cats or sorted(ann["categories"], key=lambda c: c["id"])
        splits[split] = ([im for im in ann["images"] if im.get("license") in free],
                         ann["annotations"],
                         lambda im: files.get(Path(im["file_name"]).name, Path("/nonexistent")),
                         None)
    return cats, splits


def _saved(folder: Path):
    """The masks hub_data.py wrote beside a layout (``seg-{train,val}.json``,
    COCO polygons, each picture with its ``path``)."""
    cats, splits = None, {}
    for split in ("train", "val"):
        ann = json.loads((folder / f"seg-{split}.json").read_text())
        cats = cats or sorted(ann["categories"], key=lambda c: c["id"])
        splits[split] = (ann["images"], ann["annotations"], lambda im: Path(im["path"]), None)
    return cats, splits


def sources(key: str, root: Path, coco: Path):
    """``(categories in class order, {split: (images, annotations, path_of, unknown_of)})``."""
    if key == "coco":
        return _coco(coco)
    if key == "taco":
        return _taco(root / "taco")
    if key == "lvis":
        return _lvis(root / "lvis", coco)
    if key == "fashionpedia":
        return _fashionpedia(root / "fashionpedia")
    if (root / key / "seg-train.json").exists():
        return _saved(root / key)
    raise KeyError(f"{key}: no masks on the hub's list (those with: {', '.join(SEG)})")


# ---------------------------------------------------------------- polygons


def joined(parts: list[list[float]]) -> np.ndarray | None:
    """An object's polygon parts as one outline (P, 2) in pixels: each part
    reached from the one before by a zero-width bridge between their nearest
    points and left by the same way back, so filling it fills every part."""
    polys = [np.asarray(p, np.float64).reshape(-1, 2) for p in parts if len(p) >= 6]
    if not polys:
        return None
    out = polys[0]
    for nxt in polys[1:]:
        d = ((out[:, None, :] - nxt[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(d.argmin()), d.shape)
        nxt = np.roll(nxt, -j, axis=0)
        out = np.concatenate([out[: i + 1], nxt, nxt[:1], out[i:]])
    return out


def _size(path: Path) -> tuple[int, int]:
    """(width, height) of a picture, from its header."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except ImportError:
        import cv2

        h, w = cv2.imread(str(path)).shape[:2]
        return w, h


def _scaled(a: dict, sx: float, sy: float, size: tuple[int, int]) -> dict:
    """An annotation for the picture as it is here (TACO's were made smaller)."""
    if sx == 1 and sy == 1:
        return a
    a = dict(a)
    x, y, w, h = a["bbox"]
    a["bbox"] = [x * sx, y * sy, w * sx, h * sy]
    a["area"] = a.get("area", w * h) * sx * sy
    seg = a.get("segmentation")
    if isinstance(seg, list):
        a["segmentation"] = [[v * (sx if k % 2 == 0 else sy) for k, v in enumerate(p)]
                             for p in seg]
    elif isinstance(seg, dict):
        import cv2
        from pycocotools import mask as mask_utils

        rle = seg if isinstance(seg["counts"], str) else \
            mask_utils.frPyObjects(seg, *seg["size"])
        m = cv2.resize(mask_utils.decode(rle), size, interpolation=cv2.INTER_NEAREST)
        rle = mask_utils.encode(np.asfortranarray(m))
        a["segmentation"] = {"size": rle["size"], "counts": rle["counts"].decode()}
    return a


def seg_data(key: str, root: Path, coco: Path, train_pictures: int = TRAIN_PICTURES,
             pick_pictures: int = PICK_PICTURES) -> Path:
    """``<root>/<key>-seg/data.yaml`` (polygons to train the decoder on) and
    ``val.json`` (the val split's truth, to score with)."""
    cats, splits = sources(key, root, coco)
    index = {c["id"]: i for i, c in enumerate(cats)}
    folder = root / f"{key}-seg"
    for part in ("images", "labels"):
        shutil.rmtree(folder / part, ignore_errors=True)
    rng = random.Random(0)
    truth = {"images": [], "annotations": [],
             "categories": [{"id": i, "name": c["name"]} for i, c in enumerate(cats)]}
    for split, (images, anns, path_of, unknown_of) in splits.items():
        per_image = defaultdict(list)
        for a in anns:
            if a["category_id"] in index:
                per_image[a["image_id"]].append(a)
        here = [im for im in images if path_of(im).exists()]
        images = [im for im in here if per_image[im["id"]]]
        if split == "val":
            for im in here:                  # pictures with nothing in them count too
                path = path_of(im)
                w, h = _size(path)
                sx, sy = w / im["width"], h / im["height"]
                truth["images"].append({"id": im["id"], "width": w, "height": h,
                                        "path": str(path.resolve()),
                                        "unknown": sorted(index[c] for c in
                                                          (unknown_of(im) if unknown_of else [])
                                                          if c in index)})
                for a in per_image[im["id"]]:
                    a = _scaled(a, sx, sy, (w, h))
                    truth["annotations"].append({"iscrowd": 0, **a,
                                                 "category_id": index[a["category_id"]]})
        keep = train_pictures if split == "train" else pick_pictures
        if keep and len(images) > keep:
            images = rng.sample(images, keep)
        (folder / "images" / split).mkdir(parents=True)
        (folder / "labels" / split).mkdir(parents=True)
        for im in images:
            w, h = float(im["width"]), float(im["height"])
            lines = []
            for a in per_image[im["id"]]:
                seg = a.get("segmentation")
                if a.get("iscrowd") or not isinstance(seg, list):
                    continue
                poly = joined(seg)
                if poly is None:
                    continue
                poly = np.clip(poly / [w, h], 0, 1)
                lines.append(f"{index[a['category_id']]} "
                             + " ".join(f"{v:.5f}" for v in poly.ravel()))
            if not lines:
                continue
            path = path_of(im)
            stem = str(im["id"])
            os.symlink(path.resolve(), folder / "images" / split / f"{stem}{path.suffix.lower()}")
            (folder / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    (folder / "val.json").write_text(json.dumps(truth))
    data = folder / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(folder.resolve()), "train": "images/train",
                                    "val": "images/val",
                                    "names": {i: c["name"] for i, c in enumerate(cats)}},
                                   allow_unicode=True, sort_keys=False))
    (folder / ".ready").touch()
    print(f"{folder.name}: {sum(1 for _ in (folder / 'labels' / 'train').iterdir())} train, "
          f"{len(truth['images'])} val pictures to score", flush=True)
    return data


def seg_train(key: str, root: Path, epochs: int = 20, batch: int = 16,
              device: str | None = None) -> Path:
    """MobileSAM's mask decoder fine-tuned on ``<root>/<key>-seg``'s polygons:
    ``run/decoder.onnx`` (never worse than MobileSAM on the val pictures: it
    scores first, as epoch 0)."""
    from easydetect.seg_trainer import train_masks

    folder = root / f"{key}-seg"
    out = folder / "run"
    onnx = train_masks(folder / "data.yaml", out, epochs=epochs, batch=batch, device=device,
                       log=lambda m: print(m, flush=True))
    if onnx is None:
        raise SystemExit(f"{key}: the mask decoder did not train")
    shutil.rmtree(out / "embeddings", ignore_errors=True)  # 2 MB a picture
    return Path(onnx)


# ---------------------------------------------------------------- scores


def score_seg(model: str, decoder: str, truth: str, limit: int | None = None,
              device: str = "CPU", backend: str | None = None,
              seg_device: str | None = None) -> dict:
    """Mask mAP50-95 and mAP50 on ``truth`` (a val.json of ``seg-data``),
    COCO's way: each picture's 100 most confident objects above 0.05, a mask
    each from the decoder in the box. Objects of a class the picture's
    ``unknown`` lists (LVIS: not outlined everywhere) are neither right nor wrong."""
    import contextlib
    import io

    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from easydetect import Detector
    from easydetect.utils.imageio import imread

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(truth)
    det = Detector(model, task="segment", device=device, verbose=False,
                   **({"backend": backend} if backend else {}))
    det.seg_model = decoder
    if seg_device:                  # MobileSAM's encoder on a device of its own (a GPU)
        det._ensure_segmenter(seg_device)
    ids = sorted(gt.imgs)[:limit] if limit else sorted(gt.imgs)
    found = []
    for k, image_id in enumerate(ids):
        im = gt.imgs[image_id]
        r = det(imread(im["path"]), conf=0.05, max_det=100, verbose=False)[0]
        unknown = set(im.get("unknown", []))
        if r.masks is None or not len(r.boxes):
            continue
        for m, c, s in zip(r.masks.data, r.boxes.cls, r.boxes.conf, strict=True):
            if int(c) in unknown:
                continue
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            found.append({"image_id": image_id, "category_id": int(c), "score": float(s),
                          "segmentation": {"size": rle["size"], "counts": rle["counts"].decode()}})
        if (k + 1) % 500 == 0:
            print(f"  masks: {k + 1}/{len(ids)} pictures", flush=True)
    if not found:
        return {"mask_map50_95": 0.0, "mask_map50": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(found), "segm")
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"mask_map50_95": round(float(ev.stats[0]), 4),
            "mask_map50": round(float(ev.stats[1]), 4)}


def score_pose(model: str, pose: str, coco: Path, limit: int | None = None,
               device: str = "CPU") -> dict:
    """OKS AP on COCO val2017 the way it is used: the detector's people, a
    keypoint set each (tools/train_pose.py's pipeline), scored by pycocotools."""
    sys.path.insert(0, str(HERE))
    from train_pose import evaluate_pipeline

    r = evaluate_pipeline(str(coco), Path(pose), model, device=device, limit=limit,
                          pycocotools=True)
    return {"oks_ap": round(float(r["coco_ap"]), 4), "oks_ap50": round(float(r["coco_ap50"]), 4)}


def score_pose_data(model: str, pose: str, data: str, limit: int | None = None,
                    device: str = "CPU", pose_device: str | None = None,
                    backend: str | None = "onnxruntime") -> dict:
    """OKS AP on a keypoint dataset's val split (``data``, Ultralytics' pose
    format) the way it is used: the detector's boxes of the classes the
    keypoint model is for, a keypoint set each, every picture's misses and
    false boxes counted, with the model's own OKS sigmas."""
    import cv2

    from easydetect import Detector
    from easydetect.data.keypoints import load_yolo
    from easydetect.pose import KeypointAP, KeypointEstimator, keypoint_rows

    estimator = KeypointEstimator(pose, device=pose_device or device)
    spec = estimator.spec
    images, people = load_yolo(data, "val", spec)
    # onnxruntime: "CUDA" is one of its devices, not OpenVINO's
    detector = Detector(model, device=device, verbose=False,
                        **({"backend": backend} if backend else {}))
    ap = KeypointAP(sigmas=spec.sigmas)
    ids = sorted(images)[:limit] if limit else sorted(images)
    k = len(spec)
    for image_id in ids:
        img = cv2.imread(str(images[image_id]))
        r = detector(img, conf=0.05, iou=0.7, max_det=100, verbose=False)[0]
        rows = keypoint_rows(spec, r.names, r.boxes.cls)
        boxes, box_conf = r.boxes.xyxy[rows], r.boxes.conf[rows]
        xy, conf = estimator(img, boxes)
        score = np.array([b * (c[c > 0.2].mean() if (c > 0.2).any() else 0.0)
                          for b, c in zip(box_conf, conf, strict=True)])
        anns = people.get(image_id, [])
        ap.add(xy, score, np.array([a["keypoints"] for a in anns], np.float64).reshape(-1, k, 3),
               np.array([a["area"] for a in anns]), np.zeros(len(anns), bool),
               np.array([a["bbox"] for a in anns], np.float64).reshape(-1, 4))
    r = ap.compute()
    return {"oks_ap": round(float(r["ap"]), 4), "oks_ap50": round(float(r.get("ap50", 0.0)), 4)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=["seg-data", "seg-train", "score-seg", "score-pose"])
    p.add_argument("dataset", nargs="?", help=f"seg-data / seg-train: one of {', '.join(SEG)}")
    p.add_argument("--root", default="~/datasets/hub")
    p.add_argument("--coco", default="~/datasets/coco")
    p.add_argument("--model", help="score-*: the detector (.onnx)")
    p.add_argument("--decoder", help="score-seg: mask_decoder.onnx")
    p.add_argument("--truth", help="score-seg: val.json of seg-data")
    p.add_argument("--pose", help="score-pose: pose.onnx")
    p.add_argument("--limit", type=int, help="val pictures to score (a quick try)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    root, coco = Path(args.root).expanduser(), Path(args.coco).expanduser()
    if args.what in ("seg-data", "seg-train") and args.dataset not in SEG:
        p.error(f"which dataset: one of {', '.join(SEG)}")
    if args.what == "seg-data":
        seg_data(args.dataset, root, coco)
    elif args.what == "seg-train":
        print(seg_train(args.dataset, root, epochs=args.epochs, device=args.device))
    elif args.what == "score-seg":
        print(json.dumps(score_seg(args.model, args.decoder, args.truth, args.limit,
                                   device=args.device or "CPU")))
    else:
        print(json.dumps(score_pose(args.model, args.pose, coco, args.limit,
                                    device=args.device or "CPU")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
