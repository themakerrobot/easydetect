#!/usr/bin/env python3
# Apache-2.0
"""A week of unattended runs on one GPU machine, one step after another.

    python tools/long_run.py                 # everything; again after a crash or reboot
    python tools/long_run.py --status        # what is done, what failed
    python tools/long_run.py --only coco-n416 bench-n320

Steps (each in a process of its own, its output in ``<root>/logs/<step>.log``):

* ``coco-data``     COCO 2017 downloaded (resuming), unzipped, converted
* ``bench-*``       the fine-tuning benchmarks against YOLO26 (tools/bench_finetune.py)
* ``coco-<v><s>``   the COCO checkpoints fine-tuned at 320, 416 and 640 px
                    (dfine-n and dfine-s), so a small input does not lose most
                    of what the 640 px weights know
* ``coco-score``    every COCO model, the original and the fine-tuned, as
                    OpenVINO IR at its size (and with 100 queries) scored on all
                    5000 val pictures; YOLO11/YOLO26 ONNX beside them; the IR
                    files gathered in ``<root>/pi-kit-coco`` for the Pi

A finished step leaves ``<root>/state/<step>.done`` and is skipped next time;
a training that stopped half way resumes from its ``last.pt``. A step whose
log stays silent for ``--stall`` seconds is stopped and tried again (three
tries); a failed step does not stop the ones after it, and the failed ones get
another pass at the end. ``<root>/REPORT.md`` is rewritten after every step.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip":
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
COCO_TRAIN_PICTURES = 118287
SIZES = (320, 416, 640)
VARIANTS = {"n": 32, "s": 16}          # batch on a 32 GB GPU
COCO_EPOCHS, COCO_PATIENCE = 60, 12


def bench(name: str, *flags: str) -> tuple[str, list[str]]:
    return name, ["bench", name, *flags]


BENCHES = [
    bench("bench-n320", "--dfine", "dfine-n", "--yolo", "yolo26n.pt", "--imgsz", "320",
          "--epochs", "30", "--batch", "8"),
    bench("bench-n320-class", "--dfine", "dfine-n", "--yolo", "yolo26n.pt", "--imgsz", "320",
          "--epochs", "30", "--batch", "8", "--queries", "100", "--denoising", "30",
          "--freeze", "backbone"),
    # the lab's classroom preset as it is: 35 epochs, no zoom/crop/colour
    bench("bench-n320-classroom", "--dfine", "dfine-n", "--yolo", "yolo26n.pt", "--imgsz",
          "320", "--epochs", "35", "--batch", "8", "--queries", "100", "--denoising", "30",
          "--freeze", "backbone", "--no-augment"),
    # the small datasets only, twice the epochs: does a little data want longer?
    bench("bench-n320-class-70ep", "--dfine", "dfine-n", "--yolo", "yolo26n.pt", "--imgsz",
          "320", "--epochs", "70", "--batch", "8", "--queries", "100", "--denoising", "30",
          "--freeze", "backbone", "--datasets", "signature", "medical-pills"),
    bench("bench-n640", "--dfine", "dfine-n", "--yolo", "yolo26n.pt", "--imgsz", "640",
          "--epochs", "100", "--batch", "16"),
    bench("bench-s640", "--dfine", "dfine-s", "--yolo", "yolo26s.pt", "--imgsz", "640",
          "--epochs", "100", "--batch", "16"),
    bench("bench-m640", "--dfine", "dfine-m", "--yolo", "yolo26m.pt", "--imgsz", "640",
          "--epochs", "100", "--batch", "16"),
]
N320 = ("--dfine", "dfine-n", "--yolo", "yolo26n.pt", "--imgsz", "320", "--batch", "8")
CLASS = ("--queries", "100", "--denoising", "30", "--freeze", "backbone")
SMALL = ("--datasets", "signature", "medical-pills")   # and 100 african-wildlife pictures
#: the same runs with other seeds: a dataset of 23 val pictures moves a lot
#: from one run to the next, and one run cannot tell a change from luck.
#: "-seedN" runs are summed up with the run of the same name without it.
REPEATS = []
for _seed in (1, 2):
    REPEATS += [
        bench(f"bench-n320-seed{_seed}", *N320, "--epochs", "30", "--seed", str(_seed)),
        bench(f"bench-n320-class-seed{_seed}", *N320, *CLASS, "--epochs", "30",
              "--seed", str(_seed)),
        bench(f"bench-n320-class-70ep-seed{_seed}", *N320, *CLASS, "--epochs", "70", *SMALL,
              "--seed", str(_seed)),
    ]
for _seed in (0, 1, 2):
    REPEATS += [
        # the batch-norm change of 0.7.5 against the way before it
        bench(f"bench-n320-adaptbn-seed{_seed}", *N320, "--epochs", "30", *SMALL,
              "--clean-bn", "adapt", "--seed", str(_seed)),
        bench(f"bench-n320-class-adaptbn-seed{_seed}", *N320, *CLASS, "--epochs", "30", *SMALL,
              "--clean-bn", "adapt", "--seed", str(_seed)),
        # how long a classroom run should be
        bench(f"bench-n320-class-100ep-seed{_seed}", *N320, *CLASS, "--epochs", "100", *SMALL,
              "--seed", str(_seed)),
        bench(f"bench-n320-classroom-70ep-seed{_seed}", *N320, *CLASS, "--epochs", "70", *SMALL,
              "--no-augment", "--seed", str(_seed)),
    ]

#: the classroom runs again, started from the COCO weights fine-tuned at 320
#: with 100 queries instead of the 640 px ones: whether that start is the
#: better default for 수업용 (against bench-n320-classroom-70ep-seed*)
FROM320 = [bench(f"bench-n320-classroom-70ep-from320-seed{_seed}", "--dfine",
                 "@coco/dfine-nq-320/weights/best.pt", "--label", "dfine-n-from320",
                 *N320[2:], *CLASS, "--epochs", "70", *SMALL, "--no-augment",
                 "--seed", str(_seed))
           for _seed in (0, 1, 2)]

#: also trained with 100 decoder queries from the start (what the Pi runs fastest)
QUERY_RUNS = [("n", 320), ("n", 416), ("s", 320), ("s", 416)]


def plan() -> list[tuple[str, list[str], list[str]]]:
    """(step, what it runs, steps it needs) in the order they run."""
    steps = [("coco-data", ["coco-data"], [])]
    steps += [(name, args, []) for name, args in BENCHES[:4]]
    for v in VARIANTS:
        for size in SIZES:
            steps.append((f"coco-{v}{size}", ["coco-train", v, str(size)], ["coco-data"]))
    for v, size in QUERY_RUNS:
        steps.append((f"coco-{v}{size}-q100", ["coco-train", v, str(size), "100"],
                      ["coco-data"]))
    steps.append(("coco-score", ["coco-score"], ["coco-data"]))
    steps += [(name, args, []) for name, args in REPEATS]
    steps += [(name, args, ["coco-n320-q100"]) for name, args in FROM320]
    steps += [(name, args, []) for name, args in BENCHES[4:]]
    return steps


# ---------------------------------------------------------------- the steps


def step_coco_data(root: Path, args) -> None:
    coco = Path(args.coco).expanduser()
    (coco / "images").mkdir(parents=True, exist_ok=True)
    zips = coco / "zips"
    zips.mkdir(exist_ok=True)
    wants = {"train2017.zip": coco / "images" / "train2017",
             "val2017.zip": coco / "images" / "val2017",
             "annotations_trainval2017.zip": coco / "annotations" / "instances_train2017.json"}
    for name, there in wants.items():
        if there.exists() and (there.is_file() or _count(there) > 1000):
            print(f"{there}: there already", flush=True)
            continue
        free = shutil.disk_usage(coco).free / 2**30
        if free < 45:
            raise SystemExit(f"only {free:.0f} GB free under {coco}; COCO wants about 45 GB "
                             "while it unzips (19 GB after)")
        z = zips / name
        print(f"downloading {name} (resumes where it stopped)", flush=True)
        _run(["wget", "-c", "--progress=dot:giga", "--tries=20", "--waitretry=30",
              "-O", str(z), COCO_URLS[name]])
        dest = coco if name.startswith("annotations") else coco / "images"
        print(f"unzipping {name}", flush=True)
        try:
            _run(["unzip", "-q", "-o", str(z), "-d", str(dest)])
        except subprocess.CalledProcessError:
            z.unlink(missing_ok=True)       # broken: the next try downloads it again
            raise
        z.unlink()                          # 18 GB back
    n = _count(coco / "images" / "train2017")
    if n < COCO_TRAIN_PICTURES:
        raise SystemExit(f"train2017 has {n} pictures, not {COCO_TRAIN_PICTURES}")
    # both splits, every time: an earlier data.yaml may point train at val2017
    _run([PY, str(HERE / "coco2yolo.py"), str(coco), "--splits", "train2017", "val2017"])
    labels = _count(coco / "labels" / "train2017")
    print(f"labels: {labels} train files", flush=True)
    if labels < COCO_TRAIN_PICTURES:
        raise SystemExit("coco2yolo wrote too few train labels")


def step_bench(root: Path, args, name: str, *flags: str) -> None:
    out = root / "bench" / name.removeprefix("bench-")
    # "@coco/..." is a file this run made: under the root, wherever that is
    flags = tuple(str(root / f[1:]) if f.startswith("@") else f for f in flags)
    _run([PY, "-u", str(HERE / "bench_finetune.py"), "--root", str(Path(args.bench_root)
         .expanduser()), "--out", str(out), "--workers", str(args.workers), *flags],
         cwd=root / "bench")


def step_coco_train(root: Path, args, v: str, size: str, queries: str | None = None) -> None:
    from easydetect import Detector

    coco_yaml = Path(args.coco).expanduser() / "data.yaml"
    # "dfine-nq-320", not "dfine-n-320-q100": resuming looks for "<name>*" folders
    project = root / "coco"
    name = f"dfine-{v}q-{size}" if queries else f"dfine-{v}-{size}"
    run = project / name
    resume = (run / "weights" / "last.pt").exists()
    if run.exists() and not resume:
        shutil.rmtree(run)                  # died before its first epoch: start clean
    best = Detector(f"dfine-{v}").train(
        data=str(coco_yaml), epochs=args.coco_epochs, imgsz=int(size), batch=VARIANTS[v],
        workers=args.workers, project=str(project), name=name, resume=resume,
        patience=COCO_PATIENCE, queries=int(queries) if queries else None)
    print(f"BEST {best}", flush=True)


def step_coco_score(root: Path, args) -> None:
    sys.path.insert(0, str(HERE))
    from pi_kit import score

    from easydetect import Detector

    coco_yaml = str(Path(args.coco).expanduser() / "data.yaml")
    kit, record = root / "pi-kit-coco", root / "coco" / "scores.json"
    kit.mkdir(parents=True, exist_ok=True)
    scores = json.loads(record.read_text()) if record.exists() else {}

    models = []                                     # (label, weights, size)
    for v in VARIANTS:
        for size in SIZES:
            models.append((f"dfine-{v} (640 px weights) at {size}", f"dfine-{v}", size))
            best = root / "coco" / f"dfine-{v}-{size}" / "weights" / "best.pt"
            if best.exists():
                models.append((f"dfine-{v} fine-tuned at {size}", str(best), size))
            best = root / "coco" / f"dfine-{v}q-{size}" / "weights" / "best.pt"
            if best.exists():   # trained with 100 queries: it runs with 100 already
                models.append((f"dfine-{v} fine-tuned at {size} with 100 queries", str(best),
                               size))
    for label, weights, size in models:
        for queries in ((None,) if "with 100 queries" in label else (None, 100)):
            key = label + (", 100 queries" if queries else "")
            if key in scores:
                continue
            tag = (f"dfine-{weights[-1]}-coco{size}" if weights.startswith("dfine-")
                   else f"{Path(weights).parents[1].name}-ft") + (f"-q{queries}" if queries else "")
            xml = Path(Detector(weights, verbose=False).export(
                format="openvino", imgsz=size, queries=queries, out_dir=kit / tag,
                verbose=False))
            xml.with_suffix(".onnx").unlink(missing_ok=True)
            scores[key] = {"file": f"{tag}/{xml.name}", **score(xml, coco_yaml, None)}
            record.write_text(json.dumps(scores, indent=2))
            print(f"{key}: {scores[key]}", flush=True)

    # YOLO beside them, exported by Ultralytics at each size, scored the same way
    if shutil.which("yolo"):
        from compare_yolo import Yolo, evaluate

        ds, truth = _truth(coco_yaml)
        work = root / "yolo"
        work.mkdir(exist_ok=True)
        for net in ("yolo11n", "yolo11s", "yolo26n", "yolo26s"):
            for size in SIZES:
                key = f"{net} at {size}"
                if key in scores:
                    continue
                onnx = work / f"{net}-{size}.onnx"
                if not onnx.exists():
                    _run(["yolo", "export", f"model={net}.pt", "format=onnx", f"imgsz={size}"],
                         cwd=work, timeout=900)
                    (work / f"{net}.onnx").rename(onnx)
                m50, m, ms = evaluate(key, Yolo(str(onnx), "CPU", size), ds.files, truth, ds.nc)
                scores[key] = {"map50": round(m50, 4), "map": round(m, 4), "ms_here": round(ms, 1)}
                record.write_text(json.dumps(scores, indent=2))
                print(f"{key}: {scores[key]}", flush=True)
    pibo = root / "yolo11s.onnx"                     # the Pibo's own, if copied here
    if pibo.exists() and "Pibo's yolo11s.onnx" not in scores:
        from compare_yolo import Yolo, evaluate

        ds, truth = _truth(coco_yaml)
        runner = Yolo(str(pibo), "CPU", 320)
        m50, m, ms = evaluate("pibo", runner, ds.files, truth, ds.nc)
        scores["Pibo's yolo11s.onnx"] = {"map50": round(m50, 4), "map": round(m, 4),
                                         "ms_here": round(ms, 1), "size": runner.size}
        record.write_text(json.dumps(scores, indent=2))


def _truth(data_yaml):
    import numpy as np

    from easydetect.data.dataset import DetDataset

    ds = DetDataset(data_yaml, "val", 64, augment=False)
    truth = []
    for f in ds.files:
        lab = ds._load_labels(f)
        cx, cy, bw, bh = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
        truth.append((np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1),
                      lab[:, 0].astype(int)))
    return ds, truth


# ---------------------------------------------------------------- running them


def _count(folder: Path) -> int:
    return sum(1 for _ in folder.iterdir()) if folder.is_dir() else 0


def _run(cmd, cwd=None, timeout=None) -> None:
    print("$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd, timeout=timeout)


def run_step(root: Path, name: str, what: list[str], args) -> bool:
    """One try of a step in a process of its own, stopped if it goes silent."""
    log = root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", str(Path(__file__).resolve()), "--root", str(root), "--coco", args.coco,
           "--bench-root", args.bench_root, "--workers", str(args.workers),
           "--coco-epochs", str(args.coco_epochs), "--internal", *what]
    with log.open("a") as out:
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} try\n")
        out.flush()
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                start_new_session=True)  # its own group: stopped as one
        while proc.poll() is None:
            time.sleep(30)
            if time.time() - log.stat().st_mtime > args.stall:
                out.write(f"\n===== silent for {args.stall} s: stopping it\n")
                out.flush()
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return False
        out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} exit {proc.returncode}\n")
    return proc.returncode == 0


def report(root: Path) -> None:
    lines = [f"# long_run report ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    state = root / "state"
    for name, _, _ in plan():
        mark = ("done" if (state / f"{name}.done").exists() else
                "FAILED" if (state / f"{name}.failed").exists() else "waiting")
        lines.append(f"- {name}: {mark}")
    lines += _repeats(root)
    for md in sorted((root / "bench").glob("*/results.md")):
        lines += ["", f"## {md.parent.name}", "", md.read_text()]
    scores = root / "coco" / "scores.json"
    if scores.exists():
        lines += ["", "## COCO val2017 (5000 pictures), OpenVINO CPU in float32", "",
                  "| model | mAP50-95 | mAP50 | ms on this PC |", "| --- | --- | --- | --- |"]
        for k, v in json.loads(scores.read_text()).items():
            lines.append(f"| {k} | {v['map']:.3f} | {v['map50']:.3f} | {v['ms_here']} |")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def _repeats(root: Path) -> list[str]:
    """Each benchmark's runs over seeds: mean ± spread a dataset."""
    import re
    import statistics

    groups: dict[str, list[dict]] = {}
    for rec in sorted((root / "bench").glob("*/results.json")):
        key = re.sub(r"-seed\d+$", "", rec.parent.name)
        groups.setdefault(key, []).append(json.loads(rec.read_text()))
    lines = []
    for key, runs in groups.items():
        if len(runs) < 2:
            continue
        lines += ["", f"## {key}: {len(runs)} runs (seeds), mAP50-95 mean ± spread", "",
                  "| dataset | runs | YOLO | D-FINE | D-FINE − YOLO |",
                  "| --- | --- | --- | --- | --- |"]
        for ds in runs[0]:
            y = [r[ds]["yolo"]["map"] for r in runs if ds in r]
            d = [r[ds]["dfine"]["map"] for r in runs if ds in r]

            def fmt(v):
                return f"{statistics.mean(v):.3f} ± {statistics.pstdev(v):.3f}"

            diff = statistics.mean(d) - statistics.mean(y)
            lines.append(f"| {ds} | {len(d)} | {fmt(y)} | {fmt(d)} | {diff:+.3f} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="~/longrun", help="where runs, logs and REPORT.md go")
    parser.add_argument("--coco", default="~/datasets/coco")
    parser.add_argument("--bench-root", default="~/datasets/bench")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--coco-epochs", type=int, default=COCO_EPOCHS)
    parser.add_argument("--stall", type=int, default=3 * 3600,
                        help="seconds of a silent log before a step counts as stuck")
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument("--pause", type=int, default=120, help="seconds between tries")
    parser.add_argument("--only", nargs="*", help="these steps only")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--internal", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    (root / "bench").mkdir(parents=True, exist_ok=True)

    if args.internal:                               # inside one step's process
        kind, *rest = args.internal
        if kind == "coco-data":
            step_coco_data(root, args)
        elif kind == "bench":
            step_bench(root, args, *rest)
        elif kind == "coco-train":
            step_coco_train(root, args, *rest)
        elif kind == "coco-score":
            step_coco_score(root, args)
        return

    if args.status:
        report(root)
        print((root / "REPORT.md").read_text())
        return

    lock = open(root / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"already running (another long_run holds {root / '.lock'})") from None

    state = root / "state"
    state.mkdir(exist_ok=True)
    steps = [s for s in plan() if not args.only or s[0] in args.only]
    for rnd in range(2):                            # a second pass for what failed
        for name, what, needs in steps:
            done, failed = state / f"{name}.done", state / f"{name}.failed"
            if done.exists():
                continue
            if any(not (state / f"{n}.done").exists() for n in needs):
                print(f"{name}: waits for {', '.join(needs)}", flush=True)
                continue
            for attempt in range(1, args.tries + 1):
                print(f"{time.strftime('%m-%d %H:%M')} {name} (try {attempt}), "
                      f"log {root / 'logs' / (name + '.log')}", flush=True)
                if run_step(root, name, what, args):
                    done.touch()
                    failed.unlink(missing_ok=True)
                    break
                time.sleep(args.pause)
            else:
                failed.touch()
                print(f"{name}: failed {args.tries} times; going on", flush=True)
            report(root)
        if all((state / f"{n}.done").exists() for n, _, _ in steps):
            break
        if rnd == 0:
            print(f"some steps failed: one more pass in {args.pause * 15 // 60} minutes",
                  flush=True)
            time.sleep(args.pause * 15)
    report(root)
    print((root / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
