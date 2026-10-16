#!/usr/bin/env python3
# Apache-2.0
"""Where a training step's time goes: data, forward, loss, backward, update.

    python tools/profile_train.py --data data.yaml --model s --batch 16

Runs the trainer's own pieces (loader, network, loss, optimiser, average) for
``--steps`` steps after a warm-up, with the GPU synchronised between phases,
and prints the mean of each phase. ``--trace`` adds PyTorch's profiler table
of the operators that take the most time over a few steps. Nothing is saved;
the weights are random (the time a step takes does not depend on them).
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time

import torch

from easydetect.data.dataset import DetDataset, load_data_yaml
from easydetect.nn import DFINENet
from easydetect.trainer import Trainer, _float32


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="s", help="n, s, m, l or x")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--trace", action="store_true", help="also PyTorch's operator table")
    parser.add_argument("--syncs", action="store_true",
                        help="list the lines of easydetect where a step waits for the GPU")
    parser.add_argument("--channels-last", action="store_true",
                        help="network and pictures in channels-last memory layout")
    args = parser.parse_args()

    nc = load_data_yaml(args.data)["nc"]
    net = DFINENet(args.model, nc)
    with tempfile.TemporaryDirectory() as tmp:
        tr = Trainer(net, args.data, epochs=1, imgsz=args.imgsz, batch=args.batch,
                     workers=args.workers, project=tmp, amp=not args.no_amp)
        ds = DetDataset(args.data, "train", args.imgsz, augment=True)
        dl = tr._loader(ds)
        net = tr.net.to(tr.device).train()
        cuda = tr.device.type == "cuda"
        layout = torch.channels_last if args.channels_last else torch.contiguous_format
        if args.channels_last:
            net = net.to(memory_format=torch.channels_last)

        def sync():
            if cuda:
                torch.cuda.synchronize()

        phases = {k: [] for k in ("data", "forward", "loss", "backward", "update")}

        def step(imgs, targets, times):
            t = time.perf_counter()
            imgs = DetDataset.as_float(imgs.to(tr.device, non_blocking=True))
            imgs = imgs.contiguous(memory_format=layout)
            targets = [{k: v.to(tr.device) for k, v in tt.items()} for tt in targets]
            with torch.amp.autocast(tr.device.type, dtype=tr.amp_dtype, enabled=tr.amp):
                out = net(imgs, targets)
            sync()
            times["forward"] = time.perf_counter() - t
            t = time.perf_counter()
            loss, _ = tr.criterion(_float32(out), targets)
            sync()
            times["loss"] = time.perf_counter() - t
            t = time.perf_counter()
            tr.opt.zero_grad(set_to_none=True)
            tr.scaler.scale(loss).backward()
            sync()
            times["backward"] = time.perf_counter() - t
            t = time.perf_counter()
            tr.scaler.unscale_(tr.opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            tr.scaler.step(tr.opt)
            tr.scaler.update()
            tr.ema.update(net)
            sync()
            times["update"] = time.perf_counter() - t

        total, done, it = args.warmup + args.steps, 0, iter(dl)
        started = None
        while done < total:
            t = time.perf_counter()
            try:
                imgs, targets = next(it)
            except StopIteration:
                it = iter(dl)
                continue
            times = {"data": time.perf_counter() - t}
            step(imgs, targets, times)
            done += 1
            if done == args.warmup:
                started = time.perf_counter()
            elif done > args.warmup:
                for k, v in times.items():
                    phases[k].append(v * 1000)
        wall = (time.perf_counter() - started) / args.steps * 1000

        name = torch.cuda.get_device_name(tr.device) if cuda else "CPU"
        print(f"\n{args.model}, batch {args.batch}, {args.imgsz} px, {args.workers} workers, "
              f"{name}, amp {tr.amp_dtype if tr.amp else 'off'}")
        for k, v in phases.items():
            print(f"  {k:9s} {statistics.mean(v):7.1f} ms")
        print(f"  {'step':9s} {wall:7.1f} ms  ({1000 / wall:.1f} steps/s, "
              f"{args.batch * 1000 / wall:.0f} pictures/s)")
        print("  data = waiting for the loader; a big number there means more --workers "
              "(or a faster disk) would help")

        if args.syncs and cuda:
            import collections
            import traceback
            import warnings

            where = collections.Counter()

            def note(message, category, filename, lineno, file=None, line=None):
                frames = [f for f in traceback.extract_stack()
                          if "easydetect" in f.filename and "profile_train" not in f.filename]
                if frames:
                    f = frames[-1]
                    where[f"{f.filename.split('easydetect/')[-1]}:{f.lineno}  {f.line}"] += 1

            warnings.showwarning = note
            warnings.simplefilter("always")
            imgs, targets = next(it)
            torch.cuda.set_sync_debug_mode(1)
            try:
                step(imgs, targets, {})
            finally:
                torch.cuda.set_sync_debug_mode(0)
            print(f"\nGPU waits in one step: {sum(where.values())}")
            for place, n in where.most_common(30):
                print(f"  {n:4d}  {place}")

        if args.trace:
            from torch.profiler import ProfilerActivity, profile

            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
            with profile(activities=acts) as prof:
                for _ in range(5):
                    try:
                        imgs, targets = next(it)
                    except StopIteration:
                        it = iter(dl)
                        imgs, targets = next(it)
                    step(imgs, targets, {})
            key = "self_cuda_time_total" if cuda else "self_cpu_time_total"
            print(prof.key_averages().table(sort_by=key, row_limit=25))
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
