# Apache-2.0
"""Training loop (PyTorch). Kept deliberately small and readable.

AdamW with a lower LR on the backbone, the LR set from the batch size unless
given, linear warmup into a cosine decay, AMP
on CUDA, ``last.pt``/``best.pt`` after every epoch, early stop on ``patience``,
and ``resume=True`` to pick a killed run back up where it stopped.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import augment as aug
from .data.dataset import DetDataset, load_data_yaml, worker_context
from .utils.checkpoint import load_checkpoint
from .utils.loss import DetectionLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#: The learning rate a batch of 4 fine-tunes well at: D-FINE from its COCO
#: weights reached 0.765 mAP50-95 on a dataset where RT-DETR's own run gave 0.762.
REF_BATCH, REF_LR = 4, 1e-4


#: Warm-up steps D-FINE uses (``LinearWarmup``); see ``Trainer.warmup_for``.
WARMUP_STEPS = 500
BASE_SIZE_REPEAT = 3  # how often imgsz itself is listed among the multi-scale sizes
EMA_RESTART_STEP = 0.0001  # the average's decay is lowered this much after the best is reloaded


def _float32(out, _seen=None):
    """The network's outputs, every floating tensor in them as float32. A
    tensor that appears twice is converted once and stays one object (the loss
    tells a layer that is its own teacher by identity)."""
    seen = {} if _seen is None else _seen
    if isinstance(out, torch.Tensor):
        if id(out) not in seen:
            seen[id(out)] = (out, out.float() if out.is_floating_point() else out)
        return seen[id(out)][1]
    if isinstance(out, dict):
        return {k: _float32(v, seen) for k, v in out.items()}
    if isinstance(out, (list, tuple)):
        return type(out)(_float32(v, seen) for v in out)
    return out


def _rng_state() -> dict:
    """Every random stream training draws from, in types ``torch.load(...,
    weights_only=True)`` accepts (numpy's key as a tensor)."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state = {"python": random.getstate(),
             "numpy": (name, torch.from_numpy(keys.astype(np.int64)), int(pos), int(has_gauss),
                       float(cached)),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    try:
        random.setstate(state["python"])
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, np.int64).astype(np.uint32), pos,
                             has_gauss, cached))
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except (KeyError, ValueError, RuntimeError, TypeError):
        pass  # another machine or version: the run continues, just not bit-identical


def auto_lr(batch: int) -> float:
    """The learning rate for a batch size: ``1e-4 × √(batch / 4)``.

    A bigger batch means fewer optimizer steps an epoch, and Adam moves each
    weight by about the learning rate per step whatever the batch, so at one
    fixed rate a batch of 32 ends the same epochs having learned an eighth as
    much. The square root is the usual rule for Adam (four times the batch,
    twice the rate); at 32 it gives 2.8e-4, beside the 2–2.5e-4 D-FINE's own
    configs train with at their batch of 32.
    """
    return REF_LR * math.sqrt(max(int(batch), 1) / REF_BATCH)


#: The backbone learns slower than the rest — D-FINE's per-size ratios.
BACKBONE_LR_MULT = {"n": 0.5, "s": 0.5, "m": 0.1, "l": 0.05, "x": 0.01}


class ModelEMA:
    """An exponential moving average of the weights — what gets validated and saved.

    The decay ramps up over ``warmups`` updates, so early epochs are not
    dominated by the random start: ``decay * (1 - exp(-updates / warmups))``.
    """

    def __init__(self, net, decay=0.9999, warmups=1000, updates=0):
        import copy

        self.module = copy.deepcopy(net).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.warmups, self.updates = decay, warmups, updates

    @torch.no_grad()
    def update(self, net):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmups))
        current = net.state_dict()
        mine = self.module.state_dict()
        floats = [k for k, v in mine.items() if v.dtype.is_floating_point]
        # one fused call over every tensor instead of ~700 small ones a step
        # (230 -> 90 ms on a CPU; fewer kernel launches on a GPU)
        torch._foreach_lerp_([mine[k] for k in floats],
                             [current[k].detach() for k in floats], 1 - d)
        for k, v in mine.items():
            if not v.dtype.is_floating_point:
                v.copy_(current[k])


#: Shorthands accepted by ``freeze``.
FREEZE_GROUPS = {
    "backbone": ("backbone",),
    "encoder": ("encoder",),
    "backbone+encoder": ("backbone", "encoder"),
}


def freeze_modules(net, freeze):
    """Stop training the named parts. Returns the modules to keep in eval mode.

    ``freeze="backbone"`` is the useful one: on a small dataset it trains
    roughly twice as fast and overfits less, because the features it starts
    with are already good.
    """
    if not freeze:
        return []
    if freeze is True:
        freeze = "backbone"
    prefixes = FREEZE_GROUPS.get(freeze, (freeze,) if isinstance(freeze, str) else tuple(freeze))
    unknown = [p for p in prefixes if not hasattr(net, p.split(".")[0])]
    if unknown:
        raise ValueError(f"nothing called {unknown} to freeze; try {sorted(FREEZE_GROUPS)}")

    frozen_params = 0
    for name, param in net.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = False
            frozen_params += param.numel()
    modules = [getattr(net, p.split(".")[0]) for p in prefixes]
    print(f"[easydetect] frozen: {', '.join(prefixes)} ({frozen_params / 1e6:.1f}M parameters)")
    return modules


def _split_counts(ds, num_classes: int) -> dict:
    """Images, boxes and boxes per class in one split — label files only, no pixels."""
    per_class = [0] * num_classes
    boxes = empty = 0
    for f in ds.files:
        labels = ds._load_labels(f)
        boxes += len(labels)
        empty += len(labels) == 0
        for c in labels[:, 0].astype(int) if len(labels) else ():
            if 0 <= c < num_classes:
                per_class[c] += 1
    return {"images": len(ds.files), "boxes": boxes, "background_images": empty,
            "per_class": per_class}


def data_warnings(data: dict, names, train_files=(), val_files=()) -> list[str]:
    """What in the data is likely to give a disappointing model, in plain words.

    ``data`` is run.json's ``data`` (``_split_counts`` per split). Nothing here
    stops training; each line is printed once and kept in run.json.
    """
    out = []
    train, val = data.get("train", {}), data.get("val")
    if train.get("images", 0) < 100:
        out.append(f"only {train.get('images', 0)} training pictures: expect a rough model; "
                   "a few hundred per class is where it starts to hold up")
    per_class = train.get("per_class", [])
    for c, n in enumerate(per_class):
        name = names[c] if isinstance(names, (list, tuple)) else (names or {}).get(c, c)
        if n == 0:
            out.append(f"class {c} ({name}) has no boxes in train: the model cannot learn it")
        elif n < 10:
            out.append(f"class {c} ({name}) has only {n} boxes in train")
    seen = [n for n in per_class if n]
    if len(seen) > 1 and max(seen) >= 50 * min(seen):
        out.append(f"classes are very uneven ({min(seen)} to {max(seen)} boxes): the rare ones "
                   "will score low; more of them helps more than anything else")
    if train.get("images") and train.get("boxes", 0) == 0:
        out.append("no boxes in any training label file: are the labels next to the pictures?")
    if val is None:
        out.append("no val split: the best epoch is picked blind (the last one is kept)")
    elif train_files and val_files:
        shared = {Path(f).resolve() for f in train_files} & {Path(f).resolve() for f in val_files}
        if shared and len(shared) == len(val_files):
            out.append("val is the same pictures as train: its scores will look far better "
                       "than the model is; hold some pictures out")
        elif shared:
            out.append(f"{len(shared)} val pictures are also in train: val scores read high")
    return out


#: The loss parts logged per epoch (see utils.loss).
LOGGED = ("vfl", "l1", "giou", "fgl", "ddf")


def _param_groups(net, lr, lr_backbone):
    """D-FINE's grouping: the backbone at its own rate, and no weight decay on
    norms and biases outside it (nor on the backbone's norms)."""
    groups = {"backbone": [], "backbone_norm": [], "no_decay": [], "rest": []}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        is_norm = "norm" in name or ".bn." in name or name.endswith(".bn")
        if name.startswith("backbone"):
            groups["backbone_norm" if is_norm else "backbone"].append(p)
        elif is_norm or name.endswith(".bias"):
            groups["no_decay"].append(p)
        else:
            groups["rest"].append(p)
    out = [
        {"params": groups["rest"], "lr": lr},
        {"params": groups["no_decay"], "lr": lr, "weight_decay": 0.0},
        {"params": groups["backbone"], "lr": lr_backbone},
        {"params": groups["backbone_norm"], "lr": lr_backbone, "weight_decay": 0.0},
    ]
    return [g for g in out if g["params"]]


def _one_thread_per_worker(_worker_id: int) -> None:
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)


class Trainer:
    # the recipe's extras, off unless asked for
    multiscale, mosaic, mixup = False, 0.0, 0.0

    def __init__(
        self,
        net,
        data,
        epochs=100,
        imgsz=640,
        batch=8,
        lr=None,
        lr_backbone_mult=None,
        weight_decay=1e-4,
        warmup_epochs=1,
        device=None,
        workers=4,
        project="runs",
        name="train",
        amp=True,
        resume=False,
        patience=50,
        seed=0,
        freeze=None,
        augment=True,
        multiscale=False,
        mosaic=0.0,
        mixup=0.0,
        on_epoch_end=None,
        on_progress=None,
        origin=None,
    ):
        seed_everything(seed)
        self.augment = augment
        self.multiscale, self.mosaic, self.mixup = bool(multiscale), float(mosaic), float(mixup)
        self.seed, self.freeze, self.origin = seed, freeze, origin
        self.lr_auto = lr is None
        lr = auto_lr(batch) if lr is None else float(lr)
        self.lr, self.lr_backbone_mult, self.weight_decay = lr, lr_backbone_mult, weight_decay
        self.net = net
        self.frozen = freeze_modules(net, freeze)
        self.on_epoch_end = on_epoch_end
        self.on_progress = on_progress
        self.data_yaml = data
        self.cfg = load_data_yaml(data)
        self.epochs, self.imgsz, self.batch = epochs, imgsz, batch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.workers = workers
        self.amp = amp and self.device.type == "cuda"
        if self.device.type == "cuda":
            # the input size is fixed (or one of a few, multi-scale): let cuDNN
            # time its convolution algorithms once and keep the fastest; and
            # let float32 matmuls (the loss, the parts autocast leaves) use TF32
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # bfloat16 where the GPU has it (RTX 30 and later): float32's range, so
        # no loss scaling and no overflow; float16 with a GradScaler otherwise
        self.amp_dtype = (torch.bfloat16 if self.amp and torch.cuda.is_bf16_supported()
                          else torch.float16)
        self.patience = patience
        self.start_epoch = 0
        self.best = -1.0

        run_root = Path(project) / name
        self.save_dir = self._resume_dir(run_root) if resume else self._unique_dir(run_root)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "weights").mkdir(exist_ok=True)

        # on the device before the optimizer exists: loading a resumed optimizer
        # state puts each tensor beside its parameter, and a parameter still on
        # the CPU left Adam's moments there — "cuda:0 and cpu" on the first step
        self.net = net = net.to(self.device)

        if lr_backbone_mult is None:
            lr_backbone_mult = BACKBONE_LR_MULT.get(getattr(net, "variant", ""), 0.1)
        self.lr_backbone_mult = lr_backbone_mult
        if self.device.type == "cuda":
            # convolutions in channels-last (NHWC): what the GPU's tensor cores
            # want, so cuDNN no longer converts every activation back and forth
            # (an RTX 5090, dfine-s at batch 16: 117 -> 100 ms a step)
            net.to(self.device, memory_format=torch.channels_last)
        self.channels_last = self.device.type == "cuda"
        self.opt = torch.optim.AdamW(
            _param_groups(net, lr, lr * lr_backbone_mult), lr=lr, weight_decay=weight_decay,
            # one kernel for every parameter on a GPU, instead of a few per tensor
            fused=self.device.type == "cuda",
        )
        self.criterion = DetectionLoss(net.num_classes, reg_max=net.decoder.reg_max)
        self.ema = ModelEMA(net)
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]
        self.scaler = torch.amp.GradScaler(enabled=self.amp and self.amp_dtype == torch.float16)
        self.since_improved = 0
        if resume:
            self._load_resume_state()

    @staticmethod
    def _unique_dir(p: Path) -> Path:
        if not p.exists():
            return p
        i = 2
        while (q := p.with_name(f"{p.name}{i}")).exists():
            i += 1
        return q

    @staticmethod
    def _resume_dir(p: Path) -> Path:
        """The most recent ``name``/``name2``/... directory that has a last.pt."""
        candidates = [
            d
            for d in sorted(p.parent.glob(f"{p.name}*"))
            if (d / "weights" / "last.pt").exists()
        ]
        return candidates[-1] if candidates else p

    def _load_resume_state(self) -> None:
        last = self.save_dir / "weights" / "last.pt"
        if not last.exists():
            print(f"[easydetect] resume: nothing to resume from in {self.save_dir}, starting fresh")
            return
        ckpt = load_checkpoint(last)
        # "model" is the averaged weights everyone uses; "raw" what training was at
        self.net.load_state_dict(ckpt.get("raw") or ckpt["model"])
        self.ema.module.load_state_dict(ckpt["model"])
        self.ema.updates = int(ckpt.get("ema_updates", 0))
        if ckpt.get("optimizer"):
            self.opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best = float(ckpt.get("best", -1.0))
        # a resumed run carries on as the uninterrupted one would: the same
        # random stream, and early stopping counting from where it was
        self.since_improved = int(ckpt.get("since_improved", 0))
        if ckpt.get("rng"):
            _set_rng_state(ckpt["rng"])
        print(f"[easydetect] resuming {self.save_dir} "
              f"at epoch {self.start_epoch + 1}/{self.epochs}")

    def _restart_from_best(self, net) -> None:
        """Start the plain-picture epochs from the best weights so far, as
        D-FINE does: the strong-augment phase may have peaked and wandered
        off before it ended. The average then follows a little faster."""
        best = self.save_dir / "weights" / "best.pt"
        if not best.exists():
            return
        ckpt = load_checkpoint(best)
        if int(ckpt.get("epoch", -1)) + 1 >= self.clean_from:
            return                           # the last epoch was the best: nothing to undo
        net.load_state_dict(ckpt.get("raw") or ckpt["model"])
        self.ema.module.load_state_dict(ckpt["model"])
        self.ema.updates = int(ckpt.get("ema_updates", self.ema.updates))
        self.ema.decay -= EMA_RESTART_STEP
        print(f"[easydetect] restarting from the best epoch so far "
              f"({int(ckpt.get('epoch', -1)) + 1}, mAP50-95 {float(ckpt.get('best', 0)):.4f})")

    # ------------------------------------------------------------- the record

    def _write_setup(self, ds) -> dict:
        """``run.json``: what this run is, written before the first batch.

        Everything someone reading the result later would ask — where it
        started, on what data, with which settings, on what machine — so a
        model card or a report never has to guess.
        """
        import platform
        import sys

        from . import __version__

        previous = self.save_dir / "run.json"
        old = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
        params = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        freeze = self.freeze if not isinstance(self.freeze, (list, tuple)) else list(self.freeze)
        setup = {
            "variant": self.net.variant,
            "num_classes": self.net.num_classes,
            "names": self.cfg["names"],
            # a resumed run started where its first attempt did
            "start": old.get("start") or self.origin,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "epochs": self.epochs,
            "patience": self.patience,
            "seed": self.seed,
            "workers": self.workers,
            "freeze": "backbone" if freeze is True else freeze or None,
            "params": params,
            "trainable_params": trainable,
            "optimizer": {
                "name": "AdamW",
                "lr": self.lr,
                "lr_from": (f"batch {self.batch}: {REF_LR:g} × √({self.batch}/{REF_BATCH})"
                            if self.lr_auto else "set"),
                "lr_backbone": self.lr * self.lr_backbone_mult,
                "weight_decay": self.weight_decay,
                "schedule": "linear warmup, then cosine decay to 1%",
                "warmup_epochs": round(self.warmup_for(max(len(ds) // self.batch, 1)), 2),
                "grad_clip": 0.1,
            },
            "amp": self.amp,
            "augment": self._augment_lines(),
            "data": {"train": _split_counts(ds, self.net.num_classes)},
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "cpu": platform.processor() or platform.machine(),
            "versions": {
                "easydetect": __version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda if self.device.type == "cuda" else None,
                "python": sys.version.split()[0],
            },
            "started": old.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "resumed_at_epoch": self.start_epoch + 1 if self.start_epoch else None,
        }
        val_files = []
        try:
            val = DetDataset(self.data_yaml, "val", self.imgsz, augment=False)
            setup["data"]["val"] = _split_counts(val, self.net.num_classes)
            val_files = val.files
        except (ValueError, FileNotFoundError, KeyError):
            pass                                     # no val split: nothing to count
        setup["warnings"] = data_warnings(setup["data"], self.cfg["names"], ds.files, val_files)
        for line in setup["warnings"]:
            print(f"[easydetect] warning: {line}", flush=True)
        previous.write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")
        return setup

    def _outcome(self, setup: dict, started: float, stopped_early: bool) -> dict:
        """How it went, read back from results.csv so a resumed run counts every epoch."""
        rows = []
        path = self.save_dir / "results.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        scored = [r for r in rows if r.get("map50_95")]
        best = max(scored, key=lambda r: float(r["map50_95"])) if scored else None
        last = rows[-1] if rows else {}
        seconds = sum(float(r.get("seconds") or 0) for r in rows)
        return {
            "run": setup,
            "best_epoch": int(best["epoch"]) if best else None,
            "stopped_early": stopped_early,
            "train_seconds": round(seconds, 1),
            "epoch_seconds": round(seconds / len(rows), 1) if rows else None,
            "final": {k: float(last[k]) for k in ("loss", *LOGGED) if last.get(k)},
            "wall_seconds": round(time.time() - started, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def warmup_for(self, steps_per_epoch: int) -> float:
        """Warm-up length in epochs: ``warmup_epochs``, or longer when that is
        only a handful of steps — 100 pictures at batch 16 make a 7-step epoch,
        no warm-up at all. D-FINE warms up over 500 steps; here up to 500, but
        never more than a twentieth of the run."""
        total = max(steps_per_epoch * self.epochs, 1)
        steps = min(WARMUP_STEPS, total // 20)
        return max(self.warmup_epochs, steps / max(steps_per_epoch, 1))

    def _set_lr(self, epoch, step, steps_per_epoch):
        t = epoch + step / max(steps_per_epoch, 1)
        warmup = self.warmup_for(steps_per_epoch)
        if t < warmup:
            scale = t / max(warmup, 1e-8)
        else:
            prog = (t - warmup) / max(self.epochs - warmup, 1e-8)
            scale = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
            scale = max(scale, 0.01)
        for g, base in zip(self.opt.param_groups, self.base_lrs, strict=False):
            g["lr"] = base * scale

    def _augment_lines(self) -> list[str]:
        """What the pictures go through, for run.json and the model card."""
        *strong, flip = aug.DESCRIPTION
        lines = list(strong) if self.augment else []
        if lines and self.mosaic:
            lines.append(f"mosaic of four pictures (p={self.mosaic:g})")
        if lines and self.mixup:
            lines.append(f"mixup of two (p={self.mixup:g})")
        if lines and self.multiscale:
            sizes = self.train_sizes()
            lines.append(f"each batch at a random size, {min(sizes)}–{max(sizes)} px")
        if lines and self.clean_from < self.epochs:
            lines.append(f"those off for the last {self.epochs - self.clean_from} epoch(s)")
        return lines + [flip, f"resize to {self.imgsz}×{self.imgsz}"]

    def train_sizes(self) -> list[int]:
        """Multi-scale training sizes, as D-FINE's batch collation draws them:
        ±25% of imgsz in steps of 32, with imgsz itself listed three times so
        a quarter of the batches train at the size the model will run at."""
        low, high = int(self.imgsz * 0.75) // 32 * 32, int(self.imgsz * 1.25) // 32 * 32
        below = [s for s in range(max(low, 32), self.imgsz, 32)]
        above = [s for s in range(max(high, 32), self.imgsz, -32)]
        return below + [self.imgsz] * BASE_SIZE_REPEAT + above

    @property
    def clean_from(self) -> int:
        """The epoch the strong augmentation stops: the last tenth trains on plain pictures."""
        return self.epochs - (max(1, round(self.epochs * 0.1)) if self.epochs >= 4 else 0)

    def _loader(self, ds):
        return DataLoader(
            ds,
            batch_size=self.batch,
            shuffle=True,
            drop_last=len(ds) > self.batch,
            num_workers=self.workers,
            collate_fn=DetDataset.collate,
            pin_memory=self.device.type == "cuda",
            # workers live for the whole run instead of being forked again for
            # every epoch, and each decodes on one thread: N workers each using
            # every core would fight over the CPU that feeds the GPU
            persistent_workers=self.workers > 0,
            worker_init_fn=_one_thread_per_worker if self.workers > 0 else None,
            multiprocessing_context=worker_context() if self.workers > 0 else None,
        )

    def train(self, val_fn=None):
        ds = DetDataset(self.data_yaml, "train", self.imgsz, augment=True,
                        mosaic=self.mosaic, mixup=self.mixup)
        ds.strong = self.augment and self.start_epoch < self.clean_from
        dl = self._loader(ds)
        net = self.net.to(self.device)
        since_improved = self.since_improved
        if self.start_epoch >= self.epochs:
            # resuming a run that already finished its schedule: there is nothing
            # to do until someone asks for more epochs
            print(
                f"[easydetect] already trained {self.start_epoch} epochs; "
                f"raise epochs above {self.epochs} to continue"
            )
            return self.save_dir / "weights" / "best.pt"
        print(
            f"[easydetect] training on {self.device}, {len(ds)} images, "
            f"{self.epochs} epochs -> {self.save_dir}"
        )
        setup = self._write_setup(ds)
        started, stopped_early = time.time(), False

        logs = dict.fromkeys(LOGGED, 0.0)
        for epoch in range(self.start_epoch, self.epochs):
            if ds.strong and epoch >= self.clean_from:
                # persistent workers hold their own copy of the dataset: a
                # new loader is how they hear that the strong augments are off
                ds.strong = False
                dl = self._loader(ds)
                print(f"[easydetect] epoch {epoch + 1}: zoom, crop and colour jitter off "
                      "for the last epochs")
                if val_fn is not None:
                    self._restart_from_best(net)
            net.train()
            for module in self.frozen:  # frozen batch norms must not keep adapting
                module.eval()
            t0, running, reported = time.time(), 0.0, 0.0
            for step, (imgs, targets) in enumerate(dl):
                self._set_lr(epoch, step, len(dl))
                imgs = DetDataset.as_float(imgs.to(self.device, non_blocking=True))
                if self.multiscale and ds.strong:  # boxes are normalised: only pixels change
                    side = random.choice(self.train_sizes())
                    if side != imgs.shape[-1]:
                        imgs = torch.nn.functional.interpolate(
                            imgs, size=(side, side), mode="bilinear", align_corners=False)
                if self.channels_last:
                    imgs = imgs.contiguous(memory_format=torch.channels_last)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype,
                                        enabled=self.amp):
                    out = net(imgs, targets)       # targets feed the denoising queries
                # the matching and the losses in float32, as D-FINE does: in
                # float16 a confident logit's sigmoid rounds to 1, the focal
                # cost's log(1 - p) to -inf, and the box costs stop counting
                # in the matching; the IoU targets lose precision too
                loss, logs = self.criterion(_float32(out), targets)

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.ema.update(net)
                running += loss.detach()
                if self.on_progress is not None and (
                    time.time() - reported >= 1.0 or step + 1 == len(dl)
                ):
                    reported = time.time()  # about once a second, not every batch
                    self.on_progress({
                        "phase": "train", "epoch": epoch + 1, "epochs": self.epochs,
                        "step": step + 1, "steps": len(dl), "seconds": reported - t0,
                    })

            avg = float(running) / max(len(dl), 1)
            msg = (
                f"epoch {epoch + 1}/{self.epochs}  loss {avg:.3f}  "
                + " ".join(f"{k} {float(logs.get(k, 0.0)):.3f}" for k in LOGGED) + "  "
                f"{time.time() - t0:.1f}s"
            )

            metric = None
            if val_fn is not None:
                if self.on_progress is not None:
                    self.on_progress({
                        "phase": "val", "epoch": epoch + 1, "epochs": self.epochs,
                        "seconds": time.time() - t0,
                    })
                metric = val_fn(self.ema.module)
                msg += f"  mAP50-95 {metric:.4f}"
            print("[easydetect] " + msg)

            row = {
                "epoch": epoch + 1,
                "loss": round(avg, 5),
                **{k: round(float(logs.get(k, 0.0)), 5) for k in LOGGED},
                "map50_95": round(metric, 5) if metric is not None else "",
                "lr": self.opt.param_groups[0]["lr"],
                "seconds": round(time.time() - t0, 2),
            }
            self._append_results(row)

            improved = metric is None or metric > self.best
            if improved:
                self.best = metric if metric is not None else self.best
                since_improved = 0
            else:
                since_improved += 1
            self.since_improved = since_improved
            self._save(net, epoch, "last.pt")
            if improved:
                self._save(net, epoch, "best.pt")
            # after the checkpoints: a callback that stops the run (a cancel
            # button) must not throw away the epoch that just finished
            if self.on_epoch_end is not None:
                self.on_epoch_end(dict(row, epochs=self.epochs, save_dir=str(self.save_dir)))
            if val_fn is not None and self.patience and since_improved >= self.patience:
                print(
                    f"[easydetect] early stop: no mAP improvement for {self.patience} epochs "
                    f"(best {self.best:.4f})"
                )
                stopped_early = True
                break

        best_path = self.save_dir / "weights" / "best.pt"
        (self.save_dir / "summary.json").write_text(
            json.dumps(
                {
                    "best_map50_95": self.best if self.best >= 0 else None,
                    "epochs_run": epoch + 1,
                    "weights": str(best_path),
                    "imgsz": self.imgsz,
                    "names": self.cfg["names"],
                    **self._outcome(setup, started, stopped_early),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[easydetect] done. weights: {best_path}")
        return best_path

    def _append_results(self, row):
        """One CSV line per epoch, so a dashboard can just tail the file."""
        path = self.save_dir / "results.csv"
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save(self, net, epoch, fname):
        torch.save(
            {
                "model": self.ema.module.state_dict(),
                "raw": net.state_dict(),
                "ema_updates": self.ema.updates,
                "optimizer": self.opt.state_dict(),
                "scaler": self.scaler.state_dict() if self.amp else None,
                "variant": net.variant,
                "num_classes": net.num_classes,
                "names": self.cfg["names"],
                "epoch": epoch,
                "best": self.best,
                "since_improved": self.since_improved,
                "imgsz": self.imgsz,
                # resuming reads last.pt only; best.pt is what people load
                **({"rng": _rng_state()} if fname == "last.pt" else {}),
            },
            self.save_dir / "weights" / fname,
        )
