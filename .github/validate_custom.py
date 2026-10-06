# ruff: noqa: E501
"""Real-data check of custom training (diag branch only).

    python .github/validate_custom.py <coco folder> <work folder>

1. Masks: COCO val2017's polygons as a segmentation dataset (pictures
   0-1199 train, 1200-1499 val); MobileSAM's decoder fine-tuned with the
   real encoder and weights. Prints val mean IoU at epoch 0 (MobileSAM as it
   was) and after.
2. Keypoints: COCO's five face keypoints (nose, eyes, ears) as a pose-format
   dataset of its own (K=5, flip_idx), the keypoint stage trained from
   dfine-s's backbone; prints OKS AP with the labelled boxes.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve())


# DataLoader workers start by spawning: they re-import this file, which
# must not train again
def main():
    coco, work = Path(sys.argv[1]), Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)


    inst = json.loads((coco / "annotations" / "instances_val2017.json").read_text())
    images = sorted(inst["images"], key=lambda im: im["id"])
    by_img = defaultdict(list)
    for a in inst["annotations"]:
        if not a.get("iscrowd") and isinstance(a["segmentation"], list) and a["area"] > 400:
            by_img[a["image_id"]].append(a)
    images = [im for im in images if by_img[im["id"]]][:1500]
    seg = work / "seg"
    for n, im in enumerate(images):
        split = "train" if n < int(len(images) * 0.8) else "val"
        link(coco / "images" / "val2017" / im["file_name"], seg / "images" / split / im["file_name"])
        lines = []
        for a in by_img[im["id"]]:
            poly = max(a["segmentation"], key=len)  # the largest part of a split object
            if len(poly) < 6:
                continue
            xy = [f"{v / (im['width'] if i % 2 == 0 else im['height']):.5f}" for i, v in enumerate(poly)]
            lines.append("0 " + " ".join(xy))
        (seg / "labels" / split).mkdir(parents=True, exist_ok=True)
        (seg / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines) + "\n")
    (seg / "data.yaml").write_text(yaml.safe_dump({"path": str(seg), "train": "images/train",
                                                   "val": "images/val", "names": {0: "object"}}))

    import os  # noqa: E402

    from easydetect.seg_trainer import read_polygons, train_masks  # noqa: E402

    print("polygon pictures:", len(read_polygons(seg / "data.yaml", "train")), "train,",
          len(read_polygons(seg / "data.yaml", "val")), "val", flush=True)
    smoke = bool(os.environ.get("SMOKE"))
    print("=== masks ===", flush=True)
    if not smoke and os.environ.get("MASKS", "1") == "1":
        train_masks(seg / "data.yaml", work / "seg_run", epochs=6, batch=16, device="cpu",
                    log=lambda m: print(m, flush=True))

    kp = json.loads((coco / "annotations" / "person_keypoints_val2017.json").read_text())
    kimgs = {im["id"]: im for im in kp["images"]}
    people = defaultdict(list)
    for a in kp["annotations"]:
        k = a["keypoints"][:15]  # nose, left/right eye, left/right ear
        if not a.get("iscrowd") and sum(1 for v in k[2::3] if v > 0) >= 3 and a["bbox"][3] > 40:
            people[a["image_id"]].append((a["bbox"], k))
    ids = sorted(people)[:2000]
    pose = work / "pose"
    for n, iid in enumerate(ids):
        im = kimgs[iid]
        split = "train" if n < int(len(ids) * 0.8) else "val"
        link(coco / "images" / "val2017" / im["file_name"], pose / "images" / split / im["file_name"])
        w, h = im["width"], im["height"]
        lines = []
        for (x, y, bw, bh), k in people[iid]:
            pts = " ".join(f"{k[i] / w:.5f} {k[i + 1] / h:.5f} {k[i + 2]}" for i in range(0, 15, 3))
            lines.append(f"0 {(x + bw / 2) / w:.5f} {(y + bh / 2) / h:.5f} {bw / w:.5f} {bh / h:.5f} {pts}")
        (pose / "labels" / split).mkdir(parents=True, exist_ok=True)
        (pose / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines) + "\n")
    (pose / "data.yaml").write_text(yaml.safe_dump({
        "path": str(pose), "train": "images/train", "val": "images/val", "names": {0: "head"},
        "kpt_shape": [5, 3], "flip_idx": [0, 2, 1, 4, 3],
        "kpt_names": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "skeleton": [[0, 1], [0, 2], [1, 3], [2, 4]]}))

    from easydetect.data.keypoints import KeypointDataset  # noqa: E402
    from easydetect.pose_trainer import train_keypoints  # noqa: E402

    print("=== keypoints ===", flush=True)
    tr = KeypointDataset.from_yolo(pose / "data.yaml", "train", augment=True)
    va = KeypointDataset.from_yolo(pose / "data.yaml", "val", augment=False)
    print(tr.spec, len(tr), "train,", len(va), "val", flush=True)
    if not smoke:
        train_keypoints(tr, va, work / "pose_run", size="s", init="dfine-s", epochs=10, batch=64, lr=1e-3,
                        warmup=200, workers=3, val_every=2, device="cpu", log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
