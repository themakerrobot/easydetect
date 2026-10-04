#!/usr/bin/env python3
# Apache-2.0
"""Train easydetect's keypoint model on COCO 2017 person keypoints.

    pip install "easydetect[train]"
    python tools/train_pose.py --coco ~/datasets/coco            # roughly 4-5 h, one RTX 5090
    python tools/train_pose.py --coco ~/datasets/coco --resume runs/pose/s
    python tools/train_pose.py --export runs/pose/s/best.pt      # -> pose-s.onnx beside it
    python tools/train_pose.py --coco ~/datasets/coco --eval runs/pose/s/pose-s.onnx \
        --detector dfine-m                                       # the whole pipeline

``--coco`` holds ``images/{train,val}2017`` and
``annotations/person_keypoints_{train,val}2017.json`` (COCO's
annotations_trainval2017.zip). The network (easydetect/nn/posenet.py) starts
from the D-FINE detector's COCO backbone of the same size, so it already knows
what people look like. Every ``--val-every`` epochs the EMA weights are scored
on val2017 with the labelled boxes (OKS AP, easydetect.pose.KeypointAP); the
best is kept as ``best.pt`` and, at the end, exported to ``pose-<size>.onnx``
and checked against PyTorch on ONNX Runtime.

Only COCO's keypoint labels (CC BY 4.0) and its pictures go into this model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build(size: str, init: str):
    from easydetect import downloads
    from easydetect.nn.posenet import PoseNet

    net = PoseNet(size, pretrained_backbone=init == "imagenet")
    if init not in ("imagenet", "none"):
        import torch

        path = Path(init) if init.endswith(".pt") else downloads.download_checkpoint(init)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("kind") == "pose":  # a keypoint model: carry on from all of it
            net.load_state_dict(state["model"])
            print(f"every weight from {path} (epoch {state.get('epoch')}, AP {state.get('ap')})")
        else:
            taken = net.load_detector_backbone(state["model"])
            print(f"backbone from {path.name}: {taken} tensors")
    return net


def param_groups(net, weight_decay: float):
    decay, no_decay = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith("keypoint_embed") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


CSV_FIELDS = ["epoch", "loss", "lr", "seconds", "ap", "ap50", "ap75", "ap_ema", "ap_net",
              "weights"]


def append_row(path: Path, row: dict) -> None:
    """Add an epoch to results.csv; a file from before ap_ema/ap_net (a run
    resumed across the change) is rewritten with the wider header first."""
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CSV_FIELDS:
                rows = list(reader)
                with path.open("w", newline="") as out:
                    w = csv.DictWriter(out, fieldnames=CSV_FIELDS)
                    w.writeheader()
                    w.writerows(rows)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def lr_at(step: int, total: int, warmup: int, lr: float, final: float = 0.05) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return lr * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * t)))


def evaluate(net, ds, device, batch: int = 256, workers: int = 4) -> dict:
    """OKS AP of ``net`` on ``ds`` (a val KeypointDataset), using its labelled boxes."""
    import torch
    from torch.utils.data import DataLoader

    from easydetect.data.dataset import worker_context
    from easydetect.pose import KeypointAP, apply, decode, invert

    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        multiprocessing_context=worker_context() if workers else None)
    per_image = defaultdict(list)
    net.eval()
    with torch.no_grad():
        for crops, _, _, index in loader:
            x, y = net(crops.to(device))
            xy, conf = decode(x.float().cpu().numpy(), y.float().cpu().numpy())
            for i, p, c in zip(index.tolist(), xy, conf, strict=True):
                kp = apply(invert(ds.matrix(i)), p)
                good = c > 0.2
                score = float(c[good].mean()) if good.any() else 0.0
                per_image[ds.items[i][0]].append((kp, score))
    ap = KeypointAP()
    for image_id in sorted({image_id for image_id, _ in ds.items}):
        anns = ds.people[image_id]
        found = per_image.get(image_id, [])
        gt = np.array([a["keypoints"] for a in anns], np.float64).reshape(len(anns), 17, 3)
        ap.add(np.array([f[0] for f in found]).reshape(len(found), -1, 2),
               np.array([f[1] for f in found]), gt,
               np.array([a["area"] for a in anns]), np.array([a.get("iscrowd", 0) for a in anns]))
    return ap.compute()


def evaluate_pipeline(coco: str, onnx: Path, detector: str, device: str = "AUTO",
                      limit: int | None = None, flip: bool = True,
                      subpixel: bool = True) -> dict:
    """OKS AP on val2017 the way it is used: the detector's person boxes, then
    keypoints in each — every picture, people missed and false boxes included.
    A person's score is its box confidence times its mean keypoint confidence.

    ``detector="gt"`` takes the labelled boxes instead (confidence 1): the
    keypoint model alone, as training scores it, but through the exported
    model and the runtime's decoding (``flip``, ``subpixel``)."""
    import cv2

    from easydetect import Detector
    from easydetect.data.keypoints import load_coco
    from easydetect.pose import KeypointAP, KeypointEstimator, person_rows

    images, people = load_coco(Path(coco), "val")
    gt_boxes = detector == "gt"
    model = None if gt_boxes else Detector(detector, device=device, verbose=False)
    estimator = KeypointEstimator(onnx, device="CPU" if device == "AUTO" else device,
                                  flip=flip, subpixel=subpixel)
    ap = KeypointAP()
    ids = sorted(images)[:limit] if limit else sorted(images)
    started = time.time()
    for k, image_id in enumerate(ids):
        anns = people.get(image_id, [])
        if gt_boxes:
            labelled = [a for a in anns if not a.get("iscrowd") and a.get("num_keypoints", 0) > 0
                        and a["bbox"][2] > 1 and a["bbox"][3] > 1]
            if not labelled:
                continue
            img = cv2.imread(str(images[image_id]))
            boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in
                              (a["bbox"] for a in labelled)], np.float32)
            box_conf = np.ones(len(boxes), np.float32)
        else:
            img = cv2.imread(str(images[image_id]))
            r = model(img, conf=0.05, iou=0.7, max_det=100, verbose=False)[0]
            rows = person_rows(r.names, r.boxes.cls)
            boxes, box_conf = r.boxes.xyxy[rows], r.boxes.conf[rows]
        xy, conf = estimator(img, boxes)
        score = np.array([b * (c[c > 0.2].mean() if (c > 0.2).any() else 0.0)
                          for b, c in zip(box_conf, conf, strict=True)])
        gt = np.array([a["keypoints"] for a in anns], np.float64).reshape(len(anns), 17, 3)
        ap.add(xy, score, gt, np.array([a["area"] for a in anns]),
               np.array([a.get("iscrowd", 0) for a in anns], bool))
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(ids)}  {time.time() - started:.0f}s", flush=True)
    result = ap.compute()
    how = ("flip, " if flip else "") + ("sub-pixel" if subpixel else "argmax")
    print(f"{onnx.name} on {detector} boxes ({how}), {len(ids)} pictures: "
          + ", ".join(f"{k} {v:.4f}" for k, v in result.items()))
    return result


def export(ckpt_path: Path, out: Path | None = None) -> Path:
    """``best.pt`` → ``pose-<size>.onnx`` (input ``image`` (B, 3, 256, 192) RGB 0-255),
    checked against PyTorch on ONNX Runtime."""
    import warnings

    import torch

    from easydetect.nn.posenet import PoseNet
    from easydetect.pose import INPUT, decode

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = PoseNet(ckpt["size"])
    net.load_state_dict(ckpt["model"])
    net.eval()
    out = out or ckpt_path.with_name(f"pose-{ckpt['size']}.onnx")
    dummy = torch.rand(2, 3, *INPUT) * 255
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(net, dummy, str(out), input_names=["image"],
                          output_names=["x", "y"], opset_version=17, dynamo=False,
                          dynamic_axes={"image": {0: "people"}, "x": {0: "people"},
                                        "y": {0: "people"}})
        ref = [t.numpy() for t in net(dummy)]
    import onnxruntime as ort

    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
        None, {"image": dummy.numpy()})
    a, b = decode(*ref)[0], decode(*got)[0]
    off = float(np.abs(a - b).max())
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB, keypoints within {off:.2f} px of PyTorch")
    if off > 1.0:
        raise SystemExit("the ONNX file does not match PyTorch")
    return out


def train(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    from easydetect.data.dataset import worker_context
    from easydetect.data.keypoints import DESCRIPTION, KeypointDataset
    from easydetect.nn.posenet import simcc_loss
    from easydetect.pose import KEYPOINT_NAMES
    from easydetect.trainer import ModelEMA, _one_thread_per_worker, seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.resume or Path(args.out) / args.size)
    out.mkdir(parents=True, exist_ok=True)
    train_ds = KeypointDataset(args.coco, "train", augment=True, limit=args.limit)
    val_ds = KeypointDataset(args.coco, "val", augment=False, limit=args.val_limit)
    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=_one_thread_per_worker if args.workers else None,
        multiprocessing_context=worker_context() if args.workers else None)
    steps_per_epoch = len(loader)
    total = steps_per_epoch * args.epochs

    last = out / "last.pt"
    resuming = bool(args.resume) and last.exists()
    net = build(args.size, "none" if resuming else args.init)
    if args.freeze:
        net.freeze(args.freeze)
    if device.type == "cpu":  # oneDNN's convolutions run faster on NHWC
        net = net.to(memory_format=torch.channels_last)
    net = net.to(device)
    opt = torch.optim.AdamW(param_groups(net, args.weight_decay), lr=args.lr)
    ema = ModelEMA(net, decay=0.9998, warmups=2000)
    start, best = 0, -1.0
    if resuming:
        state = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(state["net"])
        ema.module.load_state_dict(state["model"])
        ema.updates = state["ema_updates"]
        opt.load_state_dict(state["optimizer"])
        start, best = state["epoch"] + 1, state["best"]
        print(f"resuming {out} at epoch {start + 1}, best AP so far {best:.4f}")
    else:
        (out / "run.json").write_text(json.dumps({
            "task": "pose", "size": args.size, "init": args.init, "epochs": args.epochs,
            "batch": args.batch, "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup, "clip": args.clip, "freeze": args.freeze,
            "train_people": len(train_ds),
            "val_people": len(val_ds), "augment": DESCRIPTION,
            "keypoints": list(KEYPOINT_NAMES), "device": str(device)}, indent=2))
    ema.module.to(device)
    amp = device.type == "cuda" and not args.no_amp
    print(f"{len(train_ds)} people to train on, {len(val_ds)} to score; "
          f"{steps_per_epoch} steps an epoch on {device}{' (bf16)' if amp else ''}")

    csv_path = out / "results.csv"
    began, longest = time.time(), 0.0
    for epoch in range(start, args.epochs):
        if args.hours and longest and \
                time.time() - began + 1.1 * longest > args.hours * 3600:
            print(f"stopping before epoch {epoch + 1}: it would not end within "
                  f"{args.hours} h. Continue with --resume {out}", flush=True)
            return
        net.train()
        started, seen, running = time.time(), 0, 0.0
        for k, (crops, xy, weight, _) in enumerate(loader):
            step = epoch * steps_per_epoch + k
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.warmup, args.lr)
            crops, xy, weight = (t.to(device, non_blocking=True) for t in (crops, xy, weight))
            if device.type == "cpu":
                crops = crops.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                x, y = net(crops)
            loss = simcc_loss(x, y, xy, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            ema.update(net)
            running += loss.item() * len(crops)
            seen += len(crops)
            if k % 100 == 0:
                print(f"  epoch {epoch + 1}/{args.epochs} step {k}/{steps_per_epoch} "
                      f"loss {running / seen:.4f}", flush=True)
        row = {"epoch": epoch + 1, "loss": round(running / max(seen, 1), 5),
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - started, 1)}
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            # both the EMA and the network itself: after a loss spike the EMA
            # still holds weights from both sides of it and can score well below
            # the network for a couple of epochs (0.518 vs 0.534 measured)
            workers = min(args.workers, 8)
            scored = {"ema": evaluate(ema.module, val_ds, device, workers=workers),
                      "net": evaluate(net, val_ds, device, workers=workers)}
            net.train()
            which = max(scored, key=lambda k: scored[k]["ap"])
            metrics = scored[which]
            row.update({k: round(v, 4) for k, v in metrics.items()})
            row.update(ap_ema=round(scored["ema"]["ap"], 4), ap_net=round(scored["net"]["ap"], 4),
                       weights=which)
            if metrics["ap"] > best:
                best = metrics["ap"]
                chosen = ema.module if which == "ema" else net
                torch.save({"kind": "pose", "size": args.size, "model": chosen.state_dict(),
                            "weights": which, "keypoints": list(KEYPOINT_NAMES),
                            "epoch": epoch + 1, "ap": best}, out / "best.pt")
        print(f"epoch {epoch + 1}: " + ", ".join(f"{k} {v}" for k, v in row.items()
                                                  if k != "epoch"), flush=True)
        append_row(csv_path, row)
        torch.save({"net": net.state_dict(), "model": ema.module.state_dict(),
                    "ema_updates": ema.updates, "optimizer": opt.state_dict(),
                    "epoch": epoch, "best": best, "size": args.size}, last)
        longest = max(longest, time.time() - started)
    if (out / "best.pt").exists():
        print(f"best OKS AP {best:.4f}")
        export(out / "best.pt")
        (out / "finished").write_text(f"{best:.4f}\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--coco", help="COCO 2017 folder (images/, annotations/)")
    p.add_argument("--size", default="s", choices=["s", "m"])
    p.add_argument("--init", help="dfine-s / dfine-m (default: same size), imagenet, none, "
                                  "a D-FINE .pt, or a keypoint best.pt to train further")
    p.add_argument("--epochs", type=int, default=210)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1000, help="steps")
    p.add_argument("--clip", type=float, default=3.0,
                   help="gradient norm cap: typical steps measure 1.4-3.4 (COCO, batch 64), "
                        "so 3 trims the rare spikes and leaves ordinary steps alone")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--freeze", type=int, default=0,
                   help="keep the stem and this many backbone stages as they start")
    p.add_argument("--hours", type=float,
                   help="stop after the last epoch that ends within this time (resume later)")
    p.add_argument("--device")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/pose")
    p.add_argument("--resume", help="a run folder to continue (its last.pt)")
    p.add_argument("--limit", type=int, help="train on the first N people (a smoke test)")
    p.add_argument("--val-limit", type=int, help="score on the first N people")
    p.add_argument("--export", type=Path, help="only export this best.pt to ONNX")
    p.add_argument("--eval", type=Path, help="only score this .onnx behind a detector")
    p.add_argument("--detector", default="dfine-m",
                   help="the detector --eval runs first; gt: the labelled boxes")
    p.add_argument("--no-flip", action="store_true",
                   help="--eval: read each crop once, not also mirrored (as pose_flip=False)")
    p.add_argument("--no-subpixel", action="store_true",
                   help="--eval: decode to the peak bin only")
    args = p.parse_args(argv)
    if args.export:
        export(args.export)
        return 0
    if args.eval:
        if not args.coco:
            p.error("--coco is required to evaluate")
        evaluate_pipeline(args.coco, args.eval, args.detector, args.device or "AUTO",
                          args.val_limit, flip=not args.no_flip, subpixel=not args.no_subpixel)
        return 0
    if not args.coco:
        p.error("--coco is required to train")
    args.init = args.init or f"dfine-{args.size}"
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
