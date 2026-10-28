# Apache-2.0
"""Fine-tuning task="segment"'s mask decoder on a segmentation dataset of
your own (Ultralytics' format: a polygon on each label line).

MobileSAM outlines what is inside a box. Its image encoder (the heavy, 5 M
parameter part) stays as it is; its prompt encoder and mask decoder (4 M)
learn your objects' outlines from your polygons, prompted with their boxes —
jittered, as a detector's boxes are never exact. Each picture is encoded once
and the embedding cached, so an epoch costs only the small decoder. The
result is a ``decoder.onnx`` that drops in for MobileSAM's own.

All four of the decoder's masks learn: the single-mask output (slot 0) and
the three a box prompt chooses among by predicted IoU. Which way of picking
(``"best_iou"`` or ``"single"``) scores better on the val split, with boxes
jittered as a detector's are, is what the exported decoder does.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from easydetect.utils.imageio import imread  # any path, Hangul included

from .nn.sam_decoder import GRID, IMG_SIZE, LOW_RES, MASK_CHOICES

BOX_JITTER = 0.05  # Gaussian sigma of each edge's shift, as a share of the box's side
EXACT_BOXES = 0.3  # the share of training boxes left exact
CACHE_VERSION = "area"  # bump when the encoder's preprocessing changes


def jitter_box(box: np.ndarray, rng, sigma: float = BOX_JITTER) -> np.ndarray:
    """``box`` (x0, y0, x1, y1) with each edge moved by a Gaussian share of
    the box's width or height, as a detector's boxes are off; at least 1 px."""
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = x1 - x0, y1 - y0
    x0, y0, x1, y1 = np.array([x0, y0, x1, y1]) + rng.normal(0, sigma, 4) * [w, h, w, h]
    x0, x1 = min(x0, x1 - 1), max(x1, x0 + 1)
    y0, y1 = min(y0, y1 - 1), max(y1, y0 + 1)
    return np.clip(np.array([x0, y0, x1, y1], np.float32), 0, IMG_SIZE - 1)


def read_polygons(data_yaml: str | Path,
                  split: str) -> list[tuple[Path, list[tuple[int, np.ndarray]]]]:
    """``[(image, [(class, polygon (P, 2) 0-1), ...]), ...]`` for the pictures
    with at least one polygon label; box-only lines are left out."""
    from .data.dataset import list_images, load_data_yaml
    from .data.labels import label_path, label_row_to_polygon, read_label_rows

    cfg = load_data_yaml(data_yaml)
    if cfg[split] is None:
        raise ValueError(f"{data_yaml} has no '{split}:' entry")
    out = []
    for path in list_images(cfg["root"], cfg[split], cfg["yaml_dir"]):
        lp = Path(label_path(path))
        if not lp.exists():
            continue
        polys = read_label_rows(lp, label_row_to_polygon)
        if polys:
            out.append((Path(path), polys))
    return out


def has_polygons(data_yaml: str | Path, split: str = "train", look: int = 200) -> bool:
    """Whether a dataset's labels are polygons (a segmentation dataset), from
    the first ``look`` label files; a keypoint dataset never is."""
    from .data.dataset import list_images, load_data_yaml
    from .data.labels import label_path, label_row_to_polygon, read_label_rows

    cfg = load_data_yaml(data_yaml)
    if cfg["kpt_shape"] is not None or cfg[split] is None:
        return False
    for path in list_images(cfg["root"], cfg[split], cfg["yaml_dir"])[:look]:
        lp = Path(label_path(path))
        if lp.exists() and read_label_rows(lp, label_row_to_polygon):
            return True
    return False


class EmbeddingCache:
    """Image embeddings from the segmenter's encoder, computed once a picture
    and kept as float16 files (2 MB each) under ``folder``. They are read
    back memory-mapped, so a big dataset is not held in RAM."""

    def __init__(self, segmenter, folder: str | Path) -> None:
        self.segmenter = segmenter
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def _key(self, path: Path) -> Path:
        stat = path.stat()
        tag = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{CACHE_VERSION}"
        return self.folder / (hashlib.sha1(tag.encode()).hexdigest()[:20] + ".npy")

    def file(self, path: Path) -> tuple[Path, tuple[int, int]]:
        """``(the .npy file of the picture's embedding, (height, width))``,
        encoding the picture first if it has not been."""

        file = self._key(path)
        meta = file.with_suffix(".json")
        if file.exists() and meta.exists():
            return file, tuple(json.loads(meta.read_text()))
        img = imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        tensor, _ = self.segmenter.preprocess(img)
        emb = self.segmenter._run("encoder", {"image": tensor})[0][0].astype(np.float16)
        np.save(file, emb)
        meta.write_text(json.dumps(img.shape[:2]))
        return file, tuple(img.shape[:2])

    def __call__(self, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        """``(embedding (256, 64, 64) float16, memory-mapped, (height, width))``."""
        file, size = self.file(path)
        return np.load(file, mmap_mode="r"), size


class MaskDataset:
    """One item an object: ``(embedding (256, 64, 64), box (4,) in the 1024
    frame, target (256, 256) 0/1 at the low-res masks' scale)``. Only the
    embeddings' file names are kept; each is read, memory-mapped, when an
    item needs it (from the page cache after the first epoch)."""

    def __init__(self, data_yaml, split: str, cache: EmbeddingCache, augment: bool = False,
                 log=print, on_progress=None) -> None:
        pictures = read_polygons(data_yaml, split)
        self.augment = augment
        self.objects = []  # (picture index, polygon in the 1024 frame)
        self.files = []  # each picture's embedding (.npy), opened as needed
        started = reported = time.time()
        for n, (path, polys) in enumerate(pictures):
            file, (h, w) = cache.file(path)
            self.files.append(file)
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
        box = np.clip(np.concatenate([pts.min(0), pts.max(0)]).astype(np.float32),
                      0, IMG_SIZE - 1)
        # a detector's boxes are mostly a little off, now and then spot on
        if self.augment and np.random.random() >= EXACT_BOXES:
            box = jitter_box(box, np.random)
        target = np.zeros((LOW_RES, LOW_RES), np.uint8)
        f = LOW_RES / IMG_SIZE
        cv2.fillPoly(target, [np.round(pts * f * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
        emb = np.load(self.files[n], mmap_mode="r").astype(np.float32)
        return emb, box, target.astype(np.float32)


def _losses(masks, ious, target):
    """SAM's: focal (x20) + dice on each mask; of the three a box prompt
    chooses among only the best counts (SAM's multimask loss), and the
    single-mask output (slot 0) counts too, so either way of picking a mask
    can be the one exported. The IoU head learns the IoU each of the four has."""
    import torch
    import torch.nn.functional as F

    logits = masks  # (B, 4, 256, 256): slot 0 single-mask, 1-3 the box prompt's choices
    t = target[:, None].expand_as(logits)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t)
    focal = (0.25 * t + 0.75 * (1 - t)) * (1 - pt) ** 2 * ce
    focal = focal.flatten(2).mean(-1)
    inter = (p * t).flatten(2).sum(-1)
    dice = 1 - (2 * inter + 1) / (p.flatten(2).sum(-1) + t.flatten(2).sum(-1) + 1)
    per_mask = 20 * focal + dice
    best = per_mask[:, 1:].min(1).values.mean()
    single = per_mask[:, 0].mean()
    with torch.no_grad():
        hard = (logits > 0).float()
        actual = (hard * t).flatten(2).sum(-1) / ((hard + t).clamp(max=1).flatten(2).sum(-1) + 1e-6)
    return best + single + F.mse_loss(ious, actual)


def evaluate(decoder, ds, device, batch: int = 32, jitter: float = BOX_JITTER,
             seed: int = 0) -> dict[str, dict[str, float]]:
    """Mean IoU with the target over ``ds``'s objects, for each way of picking
    a box's mask (``"best_iou"``: the best-scored of the three; ``"single"``:
    slot 0), prompted with their exact boxes and with boxes jittered by a
    Gaussian ``jitter`` of their sides (the same jitter each call, from
    ``seed``) — the boxes a detector gives. ``{"exact": {"best_iou": ...,
    "single": ...}, "jittered": {...}}``."""
    import torch

    decoder.eval()
    rng = np.random.default_rng(seed)
    totals = {kind: dict.fromkeys(MASK_CHOICES, 0.0) for kind in ("exact", "jittered")}
    count = 0
    with torch.no_grad():
        for s in range(0, len(ds), batch):
            items = [ds[i] for i in range(s, min(s + batch, len(ds)))]
            emb = torch.from_numpy(np.stack([it[0] for it in items])).to(device)
            tgt = torch.from_numpy(np.stack([it[2] for it in items])).to(device) > 0
            exact = np.stack([it[1] for it in items])
            jittered = np.stack([jitter_box(b, rng, jitter) for b in exact])
            for kind, boxes in (("exact", exact), ("jittered", jittered)):
                masks, ious = decoder(emb, torch.from_numpy(boxes).to(device))
                rows = torch.arange(len(items))
                picks = {"best_iou": ious[:, 1:].argmax(1) + 1, "single": torch.zeros_like(rows)}
                for rule, pick in picks.items():
                    chosen = masks[rows, pick.to(masks.device)] > 0
                    inter = (chosen & tgt).flatten(1).sum(1).float()
                    union = (chosen | tgt).flatten(1).sum(1).float().clamp(min=1)
                    totals[kind][rule] += float((inter / union).sum())
            count += len(items)
    return {kind: {rule: v / max(count, 1) for rule, v in scores.items()}
            for kind, scores in totals.items()}


def choose(scores: dict[str, dict[str, float]]) -> tuple[str, float]:
    """The way of picking a mask that scores best on jittered boxes (a
    detector's boxes are never exact), and that score; ties go to SAM's own
    ``"best_iou"``."""
    rule = max(MASK_CHOICES, key=lambda r: (scores["jittered"][r], r == "best_iou"))
    return rule, scores["jittered"][rule]


def export(decoder, out: str | Path, log=print, mask_choice: str = "best_iou") -> Path:
    """The decoder as ``decoder.onnx`` in MobileSAM's interface (what
    task="segment" runs), checked against PyTorch on ONNX Runtime.
    ``mask_choice`` is how it picks each box's mask (see ``MASK_CHOICES``);
    it is part of the graph, and noted in the file's metadata as
    ``mask_choice``."""
    import warnings

    import torch

    from .nn.sam_decoder import SamDecoderOnnx

    out = Path(out)
    wrapper = SamDecoderOnnx(decoder.cpu().eval(), mask_choice=mask_choice).eval()
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
    _note(out, {"mask_choice": mask_choice})
    import onnxruntime as ort

    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
        None, {k: v.numpy() for k, v in inputs.items()})[2]
    off = float(np.abs(ref - got).max())
    log(f"{out}: {out.stat().st_size / 1e6:.1f} MB, mask logits within {off:.4f} of PyTorch")
    if off > 1e-2:
        raise RuntimeError("the ONNX decoder does not match PyTorch")
    return out


def _note(onnx_file: Path, values: dict[str, str]) -> None:
    """Write ``values`` into an ONNX file's metadata (left as it is without
    the onnx package: the graph alone decides what the decoder does)."""
    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(str(onnx_file))
    for key, value in values.items():
        entry = next((p for p in model.metadata_props if p.key == key), None) \
            or model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.save(model, str(onnx_file))


def train_masks(data_yaml, out: str | Path, *, segmenter=None, init: str = "mobile_sam",
                epochs: int = 20, batch: int = 16, lr: float = 1e-4, weight_decay: float = 0.01,
                device=None, seed: int = 0, log=print, on_progress=None) -> Path | None:
    """Fine-tune the mask decoder on ``data_yaml``'s polygons; write the run
    to ``out`` (results.csv, run.json, best.pt) and the best decoder, by val
    mean IoU on jittered boxes, to ``out/decoder.onnx``, picking each box's
    mask the way (``"best_iou"`` or ``"single"``) that scored better; run.json
    records which, and both ways' scores. MobileSAM as it was scores first
    (epoch 0), so the result is never worse than it on the val split.
    ``on_progress`` hears ``{"phase": "masks", "stage": "encode", ...}`` while
    the pictures are encoded, then about once a second ``{"phase": "masks",
    "epoch", "epochs", "step", "steps", "seconds"}`` and, at each epoch's end,
    the same with ``"miou"``."""
    import copy
    import csv

    import torch

    from . import downloads
    from .nn.sam_decoder import SamDecoder
    from .segment import default_segmenter
    from .trainer import seed_everything
    from .utils.checkpoint import load_checkpoint

    seed_everything(seed)
    from .devices import default_device

    device = torch.device(device or default_device())
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
        "weight_decay": weight_decay, "box_jitter": BOX_JITTER, "exact_boxes": EXACT_BOXES,
        "train_objects": len(train_ds), "val_objects": len(val_ds), "device": str(device)},
        indent=2))

    def save(epoch, miou, rule, scores):
        torch.save({"kind": "sam_decoder", "model": decoder.state_dict(), "epoch": epoch,
                    "miou": miou, "mask_choice": rule, "scores": scores}, out / "best.pt")

    def row(epoch, loss, scores, seconds):
        rule, miou = choose(scores)
        return {"epoch": epoch, "loss": loss, "miou": round(miou, 4),
                "miou_exact": round(scores["exact"][rule], 4), "mask_choice": rule,
                "seconds": seconds}

    if len(val_ds):
        scorer = val_ds
    else:  # the train split scores, with its boxes as they are
        scorer = copy.copy(train_ds)
        scorer.augment = False
    scores = evaluate(decoder, scorer, device)
    rule, best = choose(scores)
    save(0, best, rule, scores)
    log(f"{len(train_ds)} objects to train on, {len(val_ds)} to score; "
        f"MobileSAM as it is: mean IoU {best:.4f} ({rule}, jittered boxes)")
    rows = [row(0, "", scores, 0)]
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
        scores = evaluate(decoder, scorer, device)
        rule, miou = choose(scores)
        if on_progress is not None:
            on_progress({"phase": "masks", "epoch": epoch, "epochs": epochs, "step": steps,
                         "steps": steps, "seconds": time.time() - started,
                         "miou": round(miou, 4)})
        rows.append(row(epoch, round(running / max(seen, 1), 5), scores,
                        round(time.time() - started, 1)))
        log(f"epoch {epoch}/{epochs}: loss {rows[-1]['loss']}, mean IoU {miou:.4f} ({rule}, "
            f"jittered boxes; {scores['exact'][rule]:.4f} exact)")
        if miou > best:
            best = miou
            save(epoch, miou, rule, scores)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    state = load_checkpoint(out / "best.pt")
    decoder.load_state_dict(state["model"])
    rule = state["mask_choice"]
    run = json.loads((out / "run.json").read_text())
    run.update({"best_epoch": state["epoch"], "mask_choice": rule, "miou": round(best, 4),
                "val_miou": {kind: {r: round(v, 4) for r, v in by_rule.items()}
                             for kind, by_rule in state["scores"].items()}})
    (out / "run.json").write_text(json.dumps(run, indent=2))
    log(f"best mean IoU {best:.4f} (epoch {state['epoch']}; 0 is MobileSAM as it was), "
        f"each box's mask picked {rule!r}")
    return export(decoder, out / "decoder.onnx", log=log, mask_choice=rule)
