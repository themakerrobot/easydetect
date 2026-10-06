# Apache-2.0
"""Fine-tuning task="segment"'s mask decoder on a segmentation dataset of
your own (Ultralytics' format: a polygon on each label line).

MobileSAM outlines what is inside a box. Its image encoder (the heavy, 5 M
parameter part) stays as it is; its prompt encoder and mask decoder (4 M)
learn your objects' outlines from your polygons, prompted with their boxes —
jittered, as a detector's boxes are never exact. Each picture is encoded once
and the embedding cached, so an epoch costs only the small decoder. The
result is a ``decoder.onnx`` that drops in for MobileSAM's own.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .nn.sam_decoder import GRID, IMG_SIZE, LOW_RES

BOX_JITTER = 0.1  # of the box's side, each edge, in training


def read_polygons(data_yaml: str | Path,
                  split: str) -> list[tuple[Path, list[tuple[int, np.ndarray]]]]:
    """``[(image, [(class, polygon (P, 2) 0-1), ...]), ...]`` for the pictures
    with at least one polygon label; box-only lines are left out."""
    from .data.dataset import list_images, load_data_yaml
    from .data.labels import label_path

    cfg = load_data_yaml(data_yaml)
    if cfg[split] is None:
        raise ValueError(f"{data_yaml} has no '{split}:' entry")
    out = []
    for path in list_images(cfg["root"], cfg[split], cfg["yaml_dir"]):
        lp = Path(label_path(path))
        if not lp.exists():
            continue
        polys = []
        for line in lp.read_text().splitlines():
            v = line.split()
            if len(v) >= 7 and len(v) % 2 == 1:
                polys.append((int(float(v[0])), np.asarray(v[1:], np.float32).reshape(-1, 2)))
        if polys:
            out.append((Path(path), polys))
    return out


def has_polygons(data_yaml: str | Path, split: str = "train", look: int = 200) -> bool:
    """Whether a dataset's labels are polygons (a segmentation dataset), from
    the first ``look`` label files; a keypoint dataset never is."""
    from .data.dataset import list_images, load_data_yaml
    from .data.labels import label_path

    cfg = load_data_yaml(data_yaml)
    if cfg["kpt_shape"] is not None or cfg[split] is None:
        return False
    for path in list_images(cfg["root"], cfg[split], cfg["yaml_dir"])[:look]:
        lp = Path(label_path(path))
        if lp.exists() and any(len(v := line.split()) >= 7 and len(v) % 2 == 1
                               for line in lp.read_text().splitlines()):
            return True
    return False


class EmbeddingCache:
    """Image embeddings from the segmenter's encoder, computed once a picture
    and kept as float16 files (2 MB each) under ``folder``."""

    def __init__(self, segmenter, folder: str | Path) -> None:
        self.segmenter = segmenter
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def _key(self, path: Path) -> Path:
        stat = path.stat()
        tag = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        return self.folder / (hashlib.sha1(tag.encode()).hexdigest()[:20] + ".npy")

    def __call__(self, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        """``(embedding (256, 64, 64) float16, (height, width))``."""
        import cv2

        file = self._key(path)
        meta = file.with_suffix(".json")
        if file.exists() and meta.exists():
            return np.load(file), tuple(json.loads(meta.read_text()))
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        tensor, _ = self.segmenter.preprocess(img)
        emb = self.segmenter._run("encoder", {"image": tensor})[0][0].astype(np.float16)
        np.save(file, emb)
        meta.write_text(json.dumps(img.shape[:2]))
        return emb, img.shape[:2]


class MaskDataset:
    """One item an object: ``(embedding (256, 64, 64), box (4,) in the 1024
    frame, target (256, 256) 0/1 at the low-res masks' scale)``."""

    def __init__(self, data_yaml, split: str, cache: EmbeddingCache, augment: bool = False,
                 log=print, on_progress=None) -> None:
        pictures = read_polygons(data_yaml, split)
        self.augment = augment
        self.objects = []  # (picture index, polygon in the 1024 frame)
        self.embeddings, self.sizes = [], []
        started = reported = time.time()
        for n, (path, polys) in enumerate(pictures):
            emb, (h, w) = cache(path)
            self.embeddings.append(emb)
            scale = IMG_SIZE / max(h, w)
            for _, poly in polys:
                pts = poly * np.array([w, h], np.float32) * scale
                if np.ptp(pts[:, 0]) >= 1 and np.ptp(pts[:, 1]) >= 1:
                    self.objects.append((n, pts))
            if on_progress is not None and (time.time() - reported >= 1.0
                                            or n + 1 == len(pictures)):
                reported = time.time()
                on_progress({"phase": "masks", "stage": "encode", "split": split,
                             "step": n + 1, "steps": len(pictures),
                             "seconds": reported - started})
            if (n + 1) % 100 == 0:
                log(f"  {split}: encoded {n + 1}/{len(pictures)} pictures "
                    f"({time.time() - started:.0f}s)")

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, i: int):
        import cv2

        n, pts = self.objects[i]
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        if self.augment:
            w, h = x1 - x0, y1 - y0
            jitter = np.random.uniform(-BOX_JITTER, BOX_JITTER, 4) * np.array([w, h, w, h])
            x0, y0, x1, y1 = np.array([x0, y0, x1, y1]) + jitter
            x0, x1 = min(x0, x1 - 1), max(x1, x0 + 1)
            y0, y1 = min(y0, y1 - 1), max(y1, y0 + 1)
        box = np.clip(np.array([x0, y0, x1, y1], np.float32), 0, IMG_SIZE - 1)
        target = np.zeros((LOW_RES, LOW_RES), np.uint8)
        f = LOW_RES / IMG_SIZE
        cv2.fillPoly(target, [np.round(pts * f * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
        return self.embeddings[n].astype(np.float32), box, target.astype(np.float32)


def _losses(masks, ious, target):
    """SAM's: focal (x20) + dice on each of the three box-prompt masks, the
    best of them counted; the IoU head learns the IoU each one has."""
    import torch
    import torch.nn.functional as F

    logits = masks[:, 1:]  # (B, 3, 256, 256): the outputs a box prompt chooses among
    t = target[:, None].expand_as(logits)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t)
    focal = (0.25 * t + 0.75 * (1 - t)) * (1 - pt) ** 2 * ce
    focal = focal.flatten(2).mean(-1)
    inter = (p * t).flatten(2).sum(-1)
    dice = 1 - (2 * inter + 1) / (p.flatten(2).sum(-1) + t.flatten(2).sum(-1) + 1)
    per_mask = 20 * focal + dice
    best = per_mask.min(1).values.mean()
    with torch.no_grad():
        hard = (logits > 0).float()
        actual = (hard * t).flatten(2).sum(-1) / ((hard + t).clamp(max=1).flatten(2).sum(-1) + 1e-6)
    return best + F.mse_loss(ious[:, 1:], actual)


def evaluate(decoder, ds, device, batch: int = 32) -> float:
    """Mean IoU of the chosen mask (the box prompt's best-scored of three) with
    the target, over ``ds``'s objects, prompted with their exact boxes."""
    import torch

    decoder.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(ds), batch):
            items = [ds[i] for i in range(s, min(s + batch, len(ds)))]
            emb = torch.from_numpy(np.stack([it[0] for it in items])).to(device)
            box = torch.from_numpy(np.stack([it[1] for it in items])).to(device)
            tgt = torch.from_numpy(np.stack([it[2] for it in items])).to(device)
            masks, ious = decoder(emb, box)
            pick = ious[:, 1:].argmax(1) + 1
            chosen = masks[torch.arange(len(items)), pick] > 0
            inter = (chosen & (tgt > 0)).flatten(1).sum(1).float()
            union = (chosen | (tgt > 0)).flatten(1).sum(1).float().clamp(min=1)
            total += float((inter / union).sum())
            count += len(items)
    return total / max(count, 1)


def export(decoder, out: str | Path, log=print) -> Path:
    """The decoder as ``decoder.onnx`` in MobileSAM's interface (what
    task="segment" runs), checked against PyTorch on ONNX Runtime."""
    import warnings

    import torch

    from .nn.sam_decoder import SamDecoderOnnx

    out = Path(out)
    wrapper = SamDecoderOnnx(decoder.cpu().eval()).eval()
    inputs = {
        "image_embeddings": torch.randn(1, 256, GRID, GRID),
        "point_coords": torch.tensor([[[100.0, 100.0], [400.0, 300.0]]]),
        "point_labels": torch.tensor([[2.0, 3.0]]),
        "mask_input": torch.zeros(1, 1, LOW_RES, LOW_RES),
        "has_mask_input": torch.zeros(1),
        "orig_im_size": torch.tensor([480.0, 640.0]),
    }
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(wrapper, tuple(inputs.values()), str(out), input_names=list(inputs),
                          output_names=["masks", "iou_predictions", "low_res_masks"],
                          opset_version=17, dynamo=False,
                          dynamic_axes={"point_coords": {0: "boxes", 1: "points"},
                                        "point_labels": {0: "boxes", 1: "points"}})
        ref = wrapper(*inputs.values())[2].numpy()
    import onnxruntime as ort

    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
        None, {k: v.numpy() for k, v in inputs.items()})[2]
    off = float(np.abs(ref - got).max())
    log(f"{out}: {out.stat().st_size / 1e6:.1f} MB, mask logits within {off:.4f} of PyTorch")
    if off > 1e-2:
        raise RuntimeError("the ONNX decoder does not match PyTorch")
    return out


def train_masks(data_yaml, out: str | Path, *, segmenter=None, init: str = "mobile_sam",
                epochs: int = 20, batch: int = 16, lr: float = 1e-4, weight_decay: float = 0.01,
                device=None, seed: int = 0, log=print, on_progress=None) -> Path | None:
    """Fine-tune the mask decoder on ``data_yaml``'s polygons; write the run
    to ``out`` (results.csv, run.json, best.pt) and the best decoder, by val
    mean IoU, to ``out/decoder.onnx``. MobileSAM as it was scores first
    (epoch 0), so the result is never worse than it on the val split.
    ``on_progress`` hears ``{"phase": "masks", "stage": "encode", ...}`` while
    the pictures are encoded, then about once a second ``{"phase": "masks",
    "epoch", "epochs", "step", "steps", "seconds"}`` and, at each epoch's end,
    the same with ``"miou"``."""
    import csv

    import torch

    from . import downloads
    from .nn.sam_decoder import SamDecoder
    from .segment import default_segmenter
    from .trainer import seed_everything

    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    segmenter = segmenter or default_segmenter()
    cache = EmbeddingCache(segmenter, out / "embeddings")
    log("encoding the pictures once (MobileSAM's image encoder)…")
    train_ds = MaskDataset(data_yaml, "train", cache, augment=True, log=log,
                           on_progress=on_progress)
    val_ds = MaskDataset(data_yaml, "val", cache, augment=False, log=log,
                         on_progress=on_progress)
    if not len(train_ds):
        raise ValueError(f"{data_yaml}: no polygon labels in the train split")

    decoder = SamDecoder()
    if init != "none":
        path = Path(init) if init.endswith(".pt") else downloads.download_sam_decoder()
        taken = decoder.load_mobile_sam(torch.load(path, map_location="cpu", weights_only=True))
        log(f"decoder from {path.name}: {taken} tensors")
    decoder.to(device)
    for p in decoder.prompt_encoder.parameters():  # the box embedding stays MobileSAM's
        p.requires_grad_(False)
    params = [p for p in decoder.mask_decoder.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    batch = max(1, min(batch, len(train_ds)))
    steps = math.ceil(len(train_ds) / batch)
    total = steps * epochs
    (out / "run.json").write_text(json.dumps({
        "task": "segment", "init": init, "epochs": epochs, "batch": batch, "lr": lr,
        "weight_decay": weight_decay, "box_jitter": BOX_JITTER, "train_objects": len(train_ds),
        "val_objects": len(val_ds), "device": str(device)}, indent=2))

    def save(epoch, miou):
        torch.save({"kind": "sam_decoder", "model": decoder.state_dict(), "epoch": epoch,
                    "miou": miou}, out / "best.pt")

    scorer = val_ds if len(val_ds) else train_ds
    best = evaluate(decoder, scorer, device)
    save(0, best)
    log(f"{len(train_ds)} objects to train on, {len(val_ds)} to score; "
        f"MobileSAM as it is: mean IoU {best:.4f}")
    rows = [{"epoch": 0, "loss": "", "miou": round(best, 4), "seconds": 0}]
    for epoch in range(1, epochs + 1):
        decoder.train()
        started, running, seen = time.time(), 0.0, 0
        reported = 0.0
        order = np.random.permutation(len(train_ds))
        for k in range(steps):
            items = [train_ds[i] for i in order[k * batch:(k + 1) * batch]]
            emb = torch.from_numpy(np.stack([it[0] for it in items])).to(device)
            box = torch.from_numpy(np.stack([it[1] for it in items])).to(device)
            tgt = torch.from_numpy(np.stack([it[2] for it in items])).to(device)
            step = (epoch - 1) * steps + k
            for g in opt.param_groups:
                g["lr"] = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / total)))
            loss = _losses(*decoder(emb, box), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            running += loss.item() * len(items)
            seen += len(items)
            if on_progress is not None and time.time() - reported >= 1.0:
                reported = time.time()
                on_progress({"phase": "masks", "epoch": epoch, "epochs": epochs,
                             "step": k + 1, "steps": steps, "seconds": reported - started})
        miou = evaluate(decoder, scorer, device)
        if on_progress is not None:
            on_progress({"phase": "masks", "epoch": epoch, "epochs": epochs, "step": steps,
                         "steps": steps, "seconds": time.time() - started,
                         "miou": round(miou, 4)})
        rows.append({"epoch": epoch, "loss": round(running / max(seen, 1), 5),
                     "miou": round(miou, 4), "seconds": round(time.time() - started, 1)})
        log(f"epoch {epoch}/{epochs}: loss {rows[-1]['loss']}, mean IoU {miou:.4f}")
        if miou > best:
            best = miou
            save(epoch, miou)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    decoder.load_state_dict(state["model"])
    log(f"best mean IoU {best:.4f} (epoch {state['epoch']}; 0 is MobileSAM as it was)")
    return export(decoder, out / "decoder.onnx", log=log)
