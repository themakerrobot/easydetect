# Apache-2.0
"""Training the top-down keypoint network: on COCO's person keypoints
(tools/train_pose.py) or on a keypoint dataset of your own
(``Detector.train`` on a data.yaml with ``kpt_shape``).

The loop is the same for both: crops of the labelled boxes, SimCC targets,
AdamW with a cosine schedule, an EMA of the weights, and every few epochs OKS
AP on the val split with the labelled boxes — the EMA's and the network's,
keeping the better as ``best.pt`` — and at the end ``best.pt`` exported to
ONNX with its keypoint set (names, flip pairs, skeleton, classes) inside.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .pose import COCO, ONNX_SPEC_KEY, KeypointSpec


def build(size: str, init: str, num_keypoints: int = len(COCO), log=print):
    """A PoseNet of ``size`` with ``num_keypoints``, started from ``init``:
    "dfine-s"/"dfine-m"/"dfine-l" (a detector's backbone, downloaded), a
    detector or keypoint ``.pt`` (a keypoint model gives every weight whose
    shape fits — all of them for the same keypoint set), "imagenet" or "none"."""
    from . import downloads
    from .nn.posenet import PoseNet

    net = PoseNet(size, num_keypoints=num_keypoints, pretrained_backbone=init == "imagenet")
    if init in ("imagenet", "none"):
        return net
    import torch

    path = Path(init) if str(init).endswith(".pt") else downloads.download_checkpoint(init)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("kind") == "pose":
        mine = net.state_dict()
        fits = {k: v for k, v in state["model"].items() if k in mine and mine[k].shape == v.shape}
        mine.update(fits)
        net.load_state_dict(mine)
        log(f"{len(fits)} of {len(mine)} weights from {path} "
            f"(epoch {state.get('epoch')}, AP {state.get('ap')})")
    else:
        taken = net.load_detector_backbone(state["model"])
        log(f"backbone from {path.name}: {taken} tensors")
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
    """OKS AP of ``net`` on ``ds`` (a val KeypointDataset), using its labelled
    boxes and its keypoint set's OKS sigmas."""
    import torch
    from torch.utils.data import DataLoader

    from .data.dataset import worker_context
    from .pose import KeypointAP, apply, decode, invert

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
    k = len(ds.spec)
    ap = KeypointAP(sigmas=ds.spec.sigmas)
    for image_id in sorted({image_id for image_id, _ in ds.items}):
        anns = ds.people[image_id]
        found = per_image.get(image_id, [])
        gt = np.array([a["keypoints"] for a in anns], np.float64).reshape(len(anns), k, 3)
        ap.add(np.array([f[0] for f in found]).reshape(len(found), k, 2),
               np.array([f[1] for f in found]), gt,
               np.array([a["area"] for a in anns]), np.array([a.get("iscrowd", 0) for a in anns]))
    return ap.compute()


def export(ckpt_path: Path, out: Path | None = None, log=print) -> Path:
    """``best.pt`` → ``pose-<size>.onnx`` (input ``image`` (B, 3, 256, 192) RGB
    0-255), its keypoint set in the metadata, checked against PyTorch on ONNX
    Runtime."""
    import warnings

    import onnx
    import torch

    from .nn.posenet import PoseNet
    from .pose import INPUT, decode

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    spec = KeypointSpec.from_dict(ckpt["spec"]) if ckpt.get("spec") else COCO
    net = PoseNet(ckpt["size"], num_keypoints=len(spec))
    net.load_state_dict(ckpt["model"])
    net.eval()
    out = Path(out or ckpt_path.with_name(f"pose-{ckpt['size']}.onnx"))
    dummy = torch.rand(2, 3, *INPUT) * 255
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(net, dummy, str(out), input_names=["image"],
                          output_names=["x", "y"], opset_version=17, dynamo=False,
                          dynamic_axes={"image": {0: "people"}, "x": {0: "people"},
                                        "y": {0: "people"}})
        ref = [t.numpy() for t in net(dummy)]
    if spec != COCO:  # COCO's model reads as COCO without it, as before
        model = onnx.load(str(out))
        entry = model.metadata_props.add()
        entry.key, entry.value = ONNX_SPEC_KEY, json.dumps(spec.to_dict())
        onnx.save(model, str(out))
    import onnxruntime as ort

    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
        None, {"image": dummy.numpy()})
    a, b = decode(*ref)[0], decode(*got)[0]
    off = float(np.abs(a - b).max())
    log(f"{out}: {out.stat().st_size / 1e6:.1f} MB, keypoints within {off:.2f} px of PyTorch")
    if off > 1.0:
        raise RuntimeError("the ONNX file does not match PyTorch")
    return out


def train_keypoints(train_ds, val_ds, out: str | Path, *, size: str = "s", init: str = "dfine-s",
                    epochs: int = 210, batch: int = 256, lr: float = 2e-3,
                    weight_decay: float = 0.05, warmup: int = 1000, clip: float = 3.0,
                    workers: int = 4, val_every: int = 10, freeze: int = 0,
                    hours: float | None = None, device=None, amp: bool = True, seed: int = 0,
                    resume: bool = False, log=print, on_progress=None) -> Path | None:
    """Train a keypoint network on ``train_ds`` (a KeypointDataset; its
    ``spec`` is the keypoint set), score it on ``val_ds``, write the run to
    ``out`` (results.csv, run.json, last.pt, best.pt) and export ``best.pt``
    to ``pose-<size>.onnx``. Returns that ``.onnx``; None when ``hours`` ran
    out first (``resume=True`` continues). ``on_progress`` hears about once a
    second ``{"phase": "keypoints", "epoch", "epochs", "step", "steps",
    "seconds"}``, and at each scored epoch's end the same with ``"ap"``."""
    import torch
    from torch.utils.data import DataLoader

    from .data.dataset import worker_context
    from .data.keypoints import DESCRIPTION
    from .nn.posenet import simcc_loss
    from .trainer import ModelEMA, _one_thread_per_worker, seed_everything

    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = train_ds.spec
    batch = max(1, min(batch, len(train_ds)))
    loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, drop_last=len(train_ds) > batch,
        num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=_one_thread_per_worker if workers else None,
        multiprocessing_context=worker_context() if workers else None)
    steps_per_epoch = len(loader)
    total = steps_per_epoch * epochs
    warmup = min(warmup, max(total // 10, 1))

    last = out / "last.pt"
    resuming = resume and last.exists()
    net = build(size, "none" if resuming else init, len(spec), log=log)
    if freeze:
        net.freeze(freeze)
    if device.type == "cpu":  # oneDNN's convolutions run faster on NHWC
        net = net.to(memory_format=torch.channels_last)
    net = net.to(device)
    opt = torch.optim.AdamW(param_groups(net, weight_decay), lr=lr)
    ema = ModelEMA(net, decay=0.9998, warmups=min(2000, max(total // 5, 1)))
    start, best = 0, -1.0
    if resuming:
        state = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(state["net"])
        ema.module.load_state_dict(state["model"])
        ema.updates = state["ema_updates"]
        opt.load_state_dict(state["optimizer"])
        start, best = state["epoch"] + 1, state["best"]
        log(f"resuming {out} at epoch {start + 1}, best AP so far {best:.4f}")
    else:
        (out / "run.json").write_text(json.dumps({
            "task": "pose", "size": size, "init": str(init), "epochs": epochs,
            "batch": batch, "lr": lr, "weight_decay": weight_decay,
            "warmup_steps": warmup, "clip": clip, "freeze": freeze,
            "train_people": len(train_ds), "val_people": len(val_ds),
            "augment": DESCRIPTION, "keypoints": spec.to_dict(), "device": str(device)},
            indent=2))
    ema.module.to(device)
    amp = device.type == "cuda" and amp
    log(f"{len(train_ds)} to train on, {len(val_ds)} to score, {len(spec)} keypoints; "
        f"{steps_per_epoch} steps an epoch on {device}{' (bf16)' if amp else ''}")

    def save_best(model, which, epoch, ap):
        torch.save({"kind": "pose", "size": size, "model": model.state_dict(), "weights": which,
                    "spec": spec.to_dict(), "keypoints": list(spec.names), "epoch": epoch,
                    "ap": ap}, out / "best.pt")

    csv_path = out / "results.csv"
    began, longest = time.time(), 0.0
    for epoch in range(start, epochs):
        if hours and longest and time.time() - began + 1.1 * longest > hours * 3600:
            log(f"stopping before epoch {epoch + 1}: it would not end within {hours} h. "
                f"Continue with --resume {out}")
            return None
        net.train()
        started, seen, running = time.time(), 0, 0.0
        reported = 0.0
        for k, (crops, xy, weight, _) in enumerate(loader):
            step = epoch * steps_per_epoch + k
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, lr)
            crops, xy, weight = (t.to(device, non_blocking=True) for t in (crops, xy, weight))
            if device.type == "cpu":
                crops = crops.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                x, y = net(crops)
            loss = simcc_loss(x, y, xy, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            ema.update(net)
            running += loss.item() * len(crops)
            seen += len(crops)
            if k % 100 == 0:
                log(f"  epoch {epoch + 1}/{epochs} step {k}/{steps_per_epoch} "
                    f"loss {running / seen:.4f}")
            if on_progress is not None and (time.time() - reported >= 1.0
                                            or k + 1 == steps_per_epoch):
                reported = time.time()
                on_progress({"phase": "keypoints", "epoch": epoch + 1, "epochs": epochs,
                             "step": k + 1, "steps": steps_per_epoch,
                             "seconds": reported - started})
        row = {"epoch": epoch + 1, "loss": round(running / max(seen, 1), 5),
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - started, 1)}
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            # both the EMA and the network itself: after a loss spike the EMA
            # still holds weights from both sides of it and can score well below
            # the network for a couple of epochs (0.518 vs 0.534 measured)
            n = min(workers, 8)
            scored = {"ema": evaluate(ema.module, val_ds, device, workers=n),
                      "net": evaluate(net, val_ds, device, workers=n)}
            net.train()
            which = max(scored, key=lambda w: scored[w]["ap"])
            metrics = scored[which]
            row.update({m: round(v, 4) for m, v in metrics.items()})
            row.update(ap_ema=round(scored["ema"]["ap"], 4), ap_net=round(scored["net"]["ap"], 4),
                       weights=which)
            if metrics["ap"] > best:
                best = metrics["ap"]
                save_best(ema.module if which == "ema" else net, which, epoch + 1, best)
            if on_progress is not None:
                on_progress({"phase": "keypoints", "epoch": epoch + 1, "epochs": epochs,
                             "step": steps_per_epoch, "steps": steps_per_epoch,
                             "seconds": time.time() - started, "ap": round(metrics["ap"], 4)})
        log(f"epoch {epoch + 1}: " + ", ".join(f"{m} {v}" for m, v in row.items() if m != "epoch"))
        append_row(csv_path, row)
        torch.save({"net": net.state_dict(), "model": ema.module.state_dict(),
                    "ema_updates": ema.updates, "optimizer": opt.state_dict(),
                    "epoch": epoch, "best": best, "size": size, "spec": spec.to_dict()}, last)
        longest = max(longest, time.time() - started)
    if not (out / "best.pt").exists():
        return None
    log(f"best OKS AP {best:.4f}")
    onnx_path = export(out / "best.pt", log=log)
    (out / "finished").write_text(f"{best:.4f}\n")
    return onnx_path
