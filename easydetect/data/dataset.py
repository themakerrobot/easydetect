# Apache-2.0
"""Detection dataset: a data.yaml plus images/ and labels/*.txt.

data.yaml:
    path: dataset root (optional)
    train: images dir or txt list
    val: images dir or txt list
    names: {0: person, ...} or [person, ...]
Labels: <images-dir with 'images' replaced by 'labels'>/<stem>.txt
        each line: cls cx cy w h  (normalized)
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from . import augment as aug
from .labels import label_path, label_row_to_box

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    yaml_dir = Path(path).resolve().parent
    root = Path(cfg.get("path") or yaml_dir)
    if not root.is_absolute():
        root = yaml_dir / root
    if not root.exists():
        # a dataset downloaded from elsewhere often keeps its author's path
        # (/content/datasets/..., a Windows home folder); the yaml's own folder
        # is the only root that can be right on this machine
        root = yaml_dir
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}
    kpt_shape = cfg.get("kpt_shape")
    if kpt_shape is not None:
        kpt_shape = [int(kpt_shape[0]), int(kpt_shape[1]) if len(kpt_shape) > 1 else 3]
        if kpt_shape[1] not in (2, 3) or kpt_shape[0] < 1:
            raise ValueError(f"{path}: kpt_shape must be [K, 2] or [K, 3], not {cfg['kpt_shape']}")
    return {
        "root": root,
        "yaml_dir": yaml_dir,
        "train": cfg.get("train"),
        "val": cfg.get("val"),
        "names": names,
        "nc": len(names),
        # keypoint datasets (Ultralytics' pose format): [K, 2 or 3], the
        # left/right swap for flips, and optionally names and a skeleton
        "kpt_shape": kpt_shape,
        "flip_idx": cfg.get("flip_idx"),
        "kpt_names": cfg.get("kpt_names"),
        "skeleton": cfg.get("skeleton"),
    }


def _locate(root: Path, yaml_dir: Path, spec: str) -> Path:
    """Where a train/val entry points, trying the ways data.yaml files are written.

    Relative to ``path:`` first; then to the yaml itself; then with leading
    ``../`` dropped — exports that sit beside their splits still write
    ``train: ../train/images``.
    """
    spec_path = Path(spec)
    if spec_path.is_absolute():
        return spec_path
    tried = [root / spec_path, yaml_dir / spec_path]
    stripped = Path(*[part for part in spec_path.parts if part != ".."] or ["."])
    tried.append(yaml_dir / stripped)
    for candidate in tried:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"train/val entry {spec!r} not found; tried "
        + ", ".join(str(c) for c in dict.fromkeys(tried))
    )


def list_images(root: Path, spec, yaml_dir: Path | None = None):
    """Images for one split: a folder, a .txt list, or a list of either."""
    if isinstance(spec, (list, tuple)):
        files = [f for s in spec for f in list_images(root, s, yaml_dir)]
        return list(dict.fromkeys(files))
    p = _locate(root, yaml_dir or root, str(spec))
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
    if p.suffix == ".txt":
        base = p.parent
        out = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                q = Path(line)
                out.append(q if q.is_absolute() else base / q)
        return out
    raise FileNotFoundError(f"train/val entry not found: {p}")


_label_path = label_path  # older name, kept for callers


def worker_context():
    """How DataLoader workers start: forked, unless this process has run OpenVINO.

    A fork copies only the calling thread, so a process whose OpenVINO thread
    pool is alive (it lives on after a prediction) can hand its workers locks
    held by threads that no longer exist — the lab, which serves predictions
    and trains in one process, crashed that way. Such a process starts its
    workers from a ``forkserver``, a small process that never ran any of it.

    Not every process, though: a forkserver worker first runs the main script
    again, so a script without ``if __name__ == "__main__":`` would train once
    per worker, and one piped in (``python - < train.py``) has no file to run
    — their workers die on start. A plain training script never loads
    OpenVINO, and is forked as before. Windows has neither: spawn, always.
    """
    import multiprocessing
    import sys

    methods = multiprocessing.get_all_start_methods()
    if "forkserver" not in methods or "openvino" not in sys.modules:
        return None  # the platform's default: fork on Linux, spawn on Windows and macOS
    main = sys.modules.get("__main__")
    named = getattr(getattr(main, "__spec__", None), "name", None)  # python -m: re-imported
    if not named and str(getattr(main, "__file__", "")).startswith("<"):
        return None  # piped in: nothing a worker could run again
    return "forkserver"


class DetDataset(Dataset):
    def __init__(self, data_yaml, split="train", imgsz=640, augment=True,
                 mosaic=0.0, mixup=0.0):
        cfg = load_data_yaml(data_yaml)
        self.names, self.nc = cfg["names"], cfg["nc"]
        self.imgsz = imgsz
        self.augment = augment and split == "train"
        # zoom-out, crop and colour jitter (and mosaic, mixup); the trainer
        # turns them off for the last epochs (the flip stays)
        self.strong = True
        self.mosaic, self.mixup = float(mosaic), float(mixup)
        if cfg[split] is None:
            raise ValueError(
                f"{data_yaml} has no '{split}:' entry. Add one — it may point at the "
                f"training images, but then mAP only measures memorisation."
            )
        self.files = list_images(cfg["root"], cfg[split], cfg["yaml_dir"])
        if not self.files:
            raise FileNotFoundError(f"no images for split '{split}'")
        self.kpt_shape = cfg["kpt_shape"]

    def __len__(self):
        return len(self.files)

    def _load_labels(self, img_file):
        lp = _label_path(img_file)
        if not lp.exists():
            return np.zeros((0, 5), np.float32)
        rows = []
        for line in lp.read_text().splitlines():
            row = label_row_to_box(line.split(), self.kpt_shape)
            if row is not None:
                rows.append(row)
        return np.asarray(rows, np.float32) if rows else np.zeros((0, 5), np.float32)

    def _raw(self, i):
        f = self.files[i]
        img = cv2.imread(str(f))
        if img is None:
            raise FileNotFoundError(f)
        return img, self._load_labels(f)  # cls, cx, cy, w, h (normalized)

    def _augmented(self, i):
        """One training picture at imgsz x imgsz: a mosaic of four now and then."""
        if self.strong and self.mosaic and random.random() < self.mosaic:
            others = random.choices(range(len(self.files)), k=3)
            return aug.mosaic([self._raw(j) for j in (i, *others)], self.imgsz)
        img, labels = self._raw(i)
        return aug.apply(img, labels, strong=self.strong, size=self.imgsz)

    def __getitem__(self, i):
        if self.augment:   # rendered straight at imgsz x imgsz
            img, labels = self._augmented(i)
            if self.strong and self.mixup and random.random() < self.mixup:
                img, labels = aug.mixup((img, labels),
                                        self._augmented(random.randrange(len(self.files))))
        else:
            img, labels = self._raw(i)
            img = cv2.resize(img, (self.imgsz, self.imgsz))  # plain resize, as D-FINE trains
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

        target = {
            "labels": torch.as_tensor(labels[:, 0], dtype=torch.long),
            "boxes": torch.as_tensor(labels[:, 1:5], dtype=torch.float32),  # cxcywh 0..1
        }
        return tensor, target

    @staticmethod
    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = [b[1] for b in batch]
        return imgs, targets
