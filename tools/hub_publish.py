#!/usr/bin/env python3
# Apache-2.0
"""One trained model into leeyunjai/easydetect-hub, ready to download and run.

    pip install -U "huggingface_hub[cli]" && hf auth login      # once, a write token
    python tools/hub_publish.py --weights runs/ppe-s640/weights/best.pt \\
        --data ~/datasets/bench/construction-ppe/bench.yaml --dataset construction-ppe

The dataset's name, licence, official page, credit and paper come from
``hub_datasets.py`` (``--title``/``--license``/``--source`` override them).

Writes ``<stage>/<dataset>/ed-<v>-<size>/`` and uploads it to the same path in
the repo:

Files are named ``<dataset>-ed-<v>-<size>.*`` (``taco-ed-s-640.onnx``), so
models of several datasets can share a folder:

* ``ed-<v>-<size>.pt``    the checkpoint without its optimizer, to fine-tune from
* ``ed-<v>-<size>.onnx``  ONNX Runtime, class names and the F1-best threshold inside
* ``ed-<v>-<size>.xml/.bin``  OpenVINO IR (Intel CPU, GPU, NPU)
* ``ed-<v>-<size>-int8.onnx`` at 320 px or less: 8 bits for a Raspberry Pi or Pibo
  (convolutions only, the stem in float, as Pibo's own model)
* ``model.json``          dataset, classes, size, task, COCO-style score on its val split

``--task segment`` / ``--task pose`` with ``--head`` (a mask_decoder.onnx or a
pose.onnx, see hub_tasks.py) publishes the same detector with that network
beside it, in a folder of its own: ``<dataset>/ed-<v>-<size>-seg/`` or
``-pose/``, scored the task's way (mask mAP50-95, OKS AP).

Then ``index.json`` (every model in the repo), ``<dataset>/README.md`` and the
front ``README.md`` are rebuilt from it and uploaded. ``--dry-run`` stops
before uploading.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "leeyunjai/easydetect-hub"
KEEP = ("model", "variant", "num_classes", "decoder_act", "qat", "queries", "denoising",
        "names", "imgsz", "epochs")


def slim(src: Path, dst: Path) -> dict:
    """``src`` without what only resuming needs (optimizer, raw weights)."""
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({k: ckpt[k] for k in KEEP if k in ckpt}, dst)
    return ckpt


THUMB = (320, 240)

NOTICE = "https://github.com/themakerrobot/easydetect/blob/main/NOTICE"
#: the Apache-2.0 models each hub model is built from (their notices: NOTICE)
BUILT_FROM = ["D-FINE (Apache-2.0)", "PP-HGNetV2 from PaddleDetection (Apache-2.0)"]
BUILT_FROM_TASK = {"segment": ["MobileSAM (Apache-2.0)"]}

#: a hub model's task: its folder's suffix, the file beside the detector, its score
TASKS = {"detect": ("", None, "mAP50-95"),
         "segment": ("-seg", "mask_decoder.onnx", "mask mAP50-95"),
         "pose": ("-pose", "pose.onnx", "OKS AP")}


def fit(image, center: tuple[float, float] | None = None):
    """``image`` cut to 4:3 around ``center`` (its middle by default) and
    scaled to THUMB, so every dataset's thumbnail has the same size."""
    import cv2

    h, w = image.shape[:2]
    cx, cy = center or (w / 2, h / 2)
    if w * 3 > h * 4:                                   # too wide: cut the sides
        cw = int(round(h * 4 / 3))
        x0 = int(min(max(cx - cw / 2, 0), w - cw))
        image = image[:, x0:x0 + cw]
    else:                                               # too tall: cut top and bottom
        ch = int(round(w * 3 / 4))
        y0 = int(min(max(cy - ch / 2, 0), h - ch))
        image = image[y0:y0 + ch]
    return cv2.resize(image, THUMB, interpolation=cv2.INTER_AREA)


def thumbnail(onnx: Path, data: str, out: Path, look: int = 200, task: str = "detect") -> Path:
    """One val picture with the model's boxes on it, 4:3 around the boxes.
    Of the first ``look``, one with 3-8 boxes found where they are largest (the
    clearest to look at); failing that, the one with the most, up to 20. Black
    bands baked into the picture are cut off."""
    import cv2
    import numpy as np

    from easydetect import Detector
    from easydetect.data.dataset import DetDataset

    model = Detector(str(onnx), task=task, verbose=False)
    clear, clearest, busy, most, calm, fewest = None, -1.0, None, -1, None, None
    for f in DetDataset(data, "val", 64, augment=False).files[:look]:
        result = model(str(f), verbose=False)[0]
        found = len(result.boxes)
        if 3 <= found <= 8:
            wh = result.boxes.xywhn[:, 2:]
            area = float((wh[:, 0] * wh[:, 1]).mean())
            if area > clearest:
                clear, clearest = result, area
        if most < found <= 20:
            busy, most = result, found
        if fewest is None or found < fewest:         # every picture crowded: the least
            calm, fewest = result, found
    best = clear or busy or calm
    image = best.plot(conf=False, line_width=2)
    lit = best.orig_img.max(axis=2) > 12
    rows, cols = np.flatnonzero(lit.any(axis=1)), np.flatnonzero(lit.any(axis=0))
    top, left = 0, 0
    if len(rows) and len(cols):
        image = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        top, left = int(rows[0]), int(cols[0])
    center = None
    if len(best.boxes):
        xyxy = best.boxes.xyxy
        center = ((xyxy[:, 0].min() + xyxy[:, 2].max()) / 2 - left,
                  (xyxy[:, 1].min() + xyxy[:, 3].max()) / 2 - top)
    image = fit(image, center)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def build(weights: Path, data: str, folder: Path, size: int, limit: int | None,
          score_data: str | None = None, int8_from: Path | None = None,
          task: str = "detect", head: Path | None = None, truth: str | None = None,
          coco: Path | None = None) -> dict:
    """The files of one model in ``folder``; ``int8_from``: an 8-bit .onnx made
    already (a quantization-aware one) instead of quantizing this one. For
    ``task`` "segment" or "pose", ``head`` goes beside the detector and the
    task's score is taken: on ``truth`` (hub_tasks.py's val.json) for masks,
    on COCO's val2017 in ``coco`` for keypoints."""
    sys.path.insert(0, str(HERE))
    from pi_kit import ort_int8, score
    from pibo_conf import pick

    from easydetect import Detector
    from easydetect.data.dataset import DetDataset
    from easydetect.exporter import embed_metadata

    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    # files named after the dataset too, so several datasets' sit in one folder
    name = f"{folder.parent.name}-{folder.name}"
    ckpt = slim(weights, folder / f"{name}.pt")
    model = Detector(str(folder / f"{name}.pt"), verbose=False)

    work = folder.parent / f".{name}-work"
    shutil.rmtree(work, ignore_errors=True)
    onnx = Path(model.export(format="onnx", imgsz=size, out_dir=work, verbose=False))
    shutil.move(str(onnx), folder / f"{name}.onnx")
    onnx = folder / f"{name}.onnx"
    xml = Path(model.export(format="openvino", imgsz=size, out_dir=work, verbose=False))
    for part in (xml, xml.with_suffix(".bin")):
        shutil.move(str(part), folder / f"{name}{part.suffix}")
    # an IR holds no class names: OpenVINO reads them beside it
    shutil.move(str(xml.with_suffix(".names.json")), folder / f"{name}.names.json")

    conf, _ = pick(onnx, data, limit)
    embed_metadata(onnx, {"easydetect.conf": f"{conf:.2f}"})
    score_data = score_data or data
    scores = {"onnx": score(onnx, score_data, limit, backend="onnxruntime")}
    files = [f"{name}.pt", f"{name}.onnx", f"{name}.xml", f"{name}.bin", f"{name}.names.json"]
    if size <= 320:
        if int8_from:
            int8 = Path(shutil.copyfile(int8_from, folder / f"{name}-int8.onnx"))
        else:
            calib = DetDataset(data, "train", size, augment=False).files
            calib = calib[:: max(1, len(calib) // 100)][:100]
            int8 = ort_int8(onnx, work / f"{name}-int8.onnx", calib, size, parts=("",),
                            percentile=0.9999, skip=("stem",))
            int8 = Path(shutil.move(str(int8), folder / f"{name}-int8.onnx"))
        conf8, _ = pick(int8, data, limit)
        embed_metadata(int8, {"easydetect.conf": f"{conf8:.2f}"})
        scores["int8"] = score(int8, score_data, limit, backend="onnxruntime")
        files.append(int8.name)
    shutil.rmtree(work, ignore_errors=True)
    task_scores = {}
    if task != "detect":
        from hub_tasks import score_pose, score_seg

        beside = folder / f"{name}.{TASKS[task][1]}"   # what easydetect looks for beside it
        shutil.copyfile(head, beside)
        files.append(beside.name)
        runs = [("onnx", onnx)] + ([("int8", folder / f"{name}-int8.onnx")] if size <= 320 else [])
        for kind, path in runs:
            if task == "segment":
                r = score_seg(str(path), str(beside), truth, limit, backend="onnxruntime")
                task_scores[kind] = r["mask_map50_95"]
            else:
                r = score_pose(str(path), str(beside), coco, limit)
                task_scores[kind] = r["oks_ap"]
    thumbnail(onnx, score_data, folder / "sample.jpg", task=task)
    files.append("sample.jpg")

    names = ckpt["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    box, box8 = scores["onnx"]["map_official"], scores.get("int8", {}).get("map_official")
    return {
        "model": f"ed-{ckpt.get('variant', 's')}", "imgsz": size, "files": files,
        "classes": {int(k): v for k, v in names.items()}, "conf": round(conf, 2),
        "task": task, "metric": TASKS[task][2],
        "weights_license": "Apache-2.0", "built_from": BUILT_FROM + BUILT_FROM_TASK.get(task, []),
        "score": task_scores.get("onnx", box), "score_int8": task_scores.get("int8", box8),
        # the detector's own boxes, whatever the task
        "map50_95": box, "map50_95_int8": box8,
        "queries": ckpt.get("queries"), "epochs": ckpt.get("epochs"),
    }


def _by_dataset(index: dict) -> dict:
    grouped = {}
    for key, m in sorted(index.items()):
        grouped.setdefault(key.split("/", 1)[0], {})[key] = m
    return grouped


HEADER = "| | dataset | classes | 320&nbsp;px (8&#8209;bit) | 640&nbsp;px | licence |"


def short_title(title: str) -> str:
    """What fits a table cell: 'TACO (Trash Annotations in Context)' -> 'TACO'."""
    if title.startswith("Roboflow 100: "):
        return title[len("Roboflow 100: "):] + " (RF100)"
    return title.split(" (")[0].replace(" Dataset", "")


def short_license(text: str) -> str:
    """'CC BY 4.0 (annotations); pictures ...' -> 'CC BY 4.0 *' (the rest on its page)."""
    head = re.split(r" \(|;", text)[0].strip()
    return head if head == text.strip() else head + " *"


def front_readme(index: dict, repo: str = REPO) -> str:
    sys.path.insert(0, str(HERE))
    from hub_datasets import CATEGORIES, category

    sections = {}
    for dataset, models in _by_dataset(index).items():
        first = next(iter(models.values()))
        thumb = f"https://huggingface.co/{repo}/resolve/main/{dataset}/thumb.jpg"
        cells = []
        for size in (320, 640):
            m = next((m for m in models.values()
                      if m["imgsz"] == size and m.get("task", "detect") == "detect"), None)
            cells.append("—" if m is None else f"{m['map50_95']:.3f}" + (
                f" ({m['map50_95_int8']:.3f})" if m.get("map50_95_int8") else ""))
        group = category(dataset)
        tasks = " · ".join(task_of(index, dataset))
        sections.setdefault(group, []).append(
            f"| <img src=\"{thumb}\" width=\"120\" height=\"90\"> | "
            f"[{short_title(first['title'])}]({dataset}/)<br><sub>{group} · {tasks}</sub> | "
            f"{len(first['classes'])} | {cells[0]} | {cells[1]} | "
            f"{short_license(first['license']).replace(' ', '&nbsp;')} |")
    order = [c for c in [*CATEGORIES, "Other"] if c in sections]
    contents = " · ".join(f"{c} ({len(sections[c])})" for c in order)
    # one table (every row the same widths), grouped by category, each row
    # naming its own under the dataset
    tables = [HEADER, "| --- | --- | --- | --- | --- | --- |",
              *(row for c in order for row in sections[c]), ""]
    others = []
    for dataset, models in _by_dataset(index).items():
        for task in ("segment", "pose"):
            cells = []
            for size in (320, 640):
                m = next((m for m in models.values()
                          if m["imgsz"] == size and m.get("task") == task), None)
                cells.append("—" if m is None else f"{m['score']:.3f}" + (
                    f" ({m['score_int8']:.3f})" if m.get("score_int8") else ""))
            if cells != ["—", "—"]:
                first = next(iter(models.values()))
                others.append(f"| [{short_title(first['title'])}]({dataset}/) | {task} | "
                              f"{TASKS[task][2]} | {cells[0]} | {cells[1]} |")
    if others:
        others = ["## Segmentation and keypoints", "",
                  "The same detectors with a second network beside them, trained on the",
                  "dataset's masks or keypoints, each in a folder of its own",
                  "(`ed-s-<size>-seg`, `ed-s-<size>-pose`). Download the folder and use the",
                  "file, its network beside it, with `task=`:", "",
                  "```python",
                  'Detector("coco-ed-s-640-seg.onnx", task="segment")  # .mask_decoder.onnx',
                  'Detector("coco-ed-s-640-pose.onnx", task="pose")    # .pose.onnx beside',
                  "# online, a hub name works too (easydetect 0.7.17 or later fetches both)",
                  '# Detector("hub:coco/ed-s-640-seg")',
                  "```", "",
                  "| dataset | task | score | 320&nbsp;px (8&#8209;bit) | 640&nbsp;px |",
                  "| --- | --- | --- | --- | --- |", *others, ""]
    big = [k for k in sorted(index) if k.endswith("-640")]
    example = next((k for k in big if k.startswith("taco/")), (big or sorted(index))[0])
    small = next((k for k in sorted(index) if k.endswith("-320")), example)
    datasets, models = len(_by_dataset(index)), len(index)
    return "\n".join([
        "---", "license: apache-2.0", "library_name: easydetect",
        "pipeline_tag: object-detection", "tags:", "  - object-detection", "  - detr",
        "  - onnx", "  - openvino", "  - int8", "  - raspberry-pi", "  - edge", "---", "",
        "# easydetect hub", "",
        f"**Ready-to-run object detectors for {datasets} public datasets. No PyTorch, no GPU:**",
        "**`pip install easydetect` and point it at a file.**", "",
        f"- **{models} models across {datasets} datasets, one download each:** "
        f"{', '.join(c.lower() for c in order)}.",
        "- **Runs anywhere.** ONNX Runtime on any CPU (Windows, macOS, Linux, Raspberry Pi);",
        "  OpenVINO on an Intel CPU, GPU or NPU.",
        "- **Small and fast.** ed-s (10M parameters), a real-time DETR: no NMS to tune.",
        "  320 px for small devices, with an 8-bit file for a Raspberry Pi; 640 px for accuracy.",
        "- **Apache-2.0 weights**, built from Apache-2.0 models (D-FINE, PP-HGNetV2, MobileSAM).",
        "  The training data's terms still apply: each dataset's licence and credit are in",
        "  the table and on its page; keep the credit, and check them before commercial use.",
        "- **Your own data next.** Every model comes with the checkpoint to fine-tune from.", "",
        "## Quick start (no PyTorch)", "",
        "```bash", "pip install -U easydetect        # 0.7.17 or later", "```", "",
        f"Download a model's folder (its page, or `hf download {repo} --include "
        f"'{example}/*' --local-dir hub`), then:", "",
        "```python", "from easydetect import Detector", "",
        "# the file you downloaded: names and threshold inside",
        f"model = Detector(\"{example.replace('/', '-')}.onnx\")",
        "# online, a hub name works too (downloads once)",
        f"# model = Detector(\"hub:{example}\")",
        "result = model(\"photo.jpg\")[0]",
        "result.show()                # or result.save(\"out.jpg\")",
        "print(result.summary())      # [{'name': ..., 'confidence': ..., 'box': {...}}, ...]",
        "```", "",
        "Or from the command line, on a picture, a folder, a video or a webcam (`source=0`):", "",
        "```bash",
        f"easydetect predict model={example.replace('/', '-')}.onnx source=photo.jpg",
        f"# easydetect predict model=hub:{example} source=photo.jpg     # online",
        "easydetect hub search=plant      # what is here, with scores", "```", "",
        "`pip install easydetect` brings numpy, OpenCV, ONNX Runtime and OpenVINO, nothing",
        "more. Only fine-tuning needs PyTorch.", "",
        "## On an Intel NPU or a Raspberry Pi", "",
        "```python",
        f"Detector(\"{example.replace('/', '-')}.xml\", device=\"NPU\")  # OpenVINO, .bin beside",
        f"Detector(\"{small.replace('/', '-')}-int8.onnx\", backend=\"onnxruntime\")  # 8 bits",
        f"# online: Detector(\"hub:{example}.xml\", device=\"NPU\"), "
        f"Detector(\"hub:{small}-int8\", ...)",
        "```", "",
        "## Fine-tune on your own pictures", "",
        "```bash", "pip install -U \"easydetect[train]\"      # adds PyTorch", "```", "",
        "```python",
        f"model = Detector(\"{example.replace('/', '-')}.pt\")",
        f"# model = Detector(\"hub:{example}.pt\")     # online",
        "model.train(data=\"my_data.yaml\", epochs=50, imgsz=640)   # YOLO-format folders",
        "```", "",
        "Or without code, in the browser:",
        "[easydetect lab](https://github.com/themakerrobot/easydetect-lab) (label, train,",
        "test and export, offline).", "",
        "## Without easydetect", "",
        "Each `.onnx` runs in plain ONNX Runtime (input: RGB, 0-1, stretched to its size;",
        "outputs: `boxes` as centre and size, 0-1, and `scores` per class).", "",
        "```bash",
        f"hf download {repo} --include '{example}/*' --local-dir hub", "```", "",
        "## What is in each folder", "",
        "| file | for |", "| --- | --- |",
        "| `<dataset>-ed-s-<size>.onnx` | ONNX Runtime, any CPU; names and threshold inside |",
        "| `<dataset>-ed-s-<size>.xml` + `.bin` | OpenVINO: Intel CPU, GPU, NPU |",
        "| `<dataset>-ed-s-320-int8.onnx` | 8 bits for a Raspberry Pi (320 px only) |",
        "| `<dataset>-ed-s-<size>.pt` | the checkpoint to fine-tune from (needs PyTorch) |",
        "| `<model>.mask_decoder.onnx`, `<model>.pose.onnx` | -seg / -pose: the network beside "
        "the detector |",
        "| `model.json` | classes, size, score, dataset licence |", "",
        "## Models", "",
        "ed-s. mAP50-95 on each dataset's own val split, COCO's way; in brackets",
        "the 8-bit file.", "",
        f"{datasets} datasets: {contents}", "", *tables,
        "\\* more terms on the dataset's page (the pictures' own licences).", "", *others,
        "## Licence", "",
        "The weights (what was made here) are Apache-2.0. They are built from Apache-2.0",
        "models: D-FINE, PP-HGNetV2 (PaddleDetection) and, for masks, MobileSAM; their",
        f"notices are in [easydetect's NOTICE]({NOTICE}).", "",
        "The training data's terms still apply to a model trained on it. Each dataset's",
        "licence, the credit it asks for and the paper to cite are on its page (click its",
        "name): keep the credit with the model, and check the dataset's and its pictures'",
        "terms before commercial use.",
        "", "Made with [easydetect](https://github.com/themakerrobot/easydetect).", ""])


def dataset_readme(dataset: str, index: dict) -> str:
    models = {k: m for k, m in index.items() if k.startswith(dataset + "/")}
    first = next(iter(models.values()))
    names = ", ".join(first["classes"].values())

    def row(k, m):
        score = m.get("score", m["map50_95"])
        int8 = m.get("score_int8", m.get("map50_95_int8"))
        task = m.get("task", "detect")
        return (f"| `{k.split('/', 1)[1]}` | {task} | {m['imgsz']} | "
                f"{m.get('metric', TASKS[task][2])} | {score:.3f} | "
                f"{f'{int8:.3f}' if int8 else '—'} |")

    order = {t: i for i, t in enumerate(TASKS)}
    rows = [row(k, m) for k, m in sorted(models.items(), key=lambda km: (
        order.get(km[1].get("task", "detect"), 9), km[1]["imgsz"]))]
    sys.path.insert(0, str(HERE))
    from hub_datasets import category

    lines = [f"# {first['title']}", "", f"Category: {category(dataset)}", "",
             "![sample](thumb.jpg)", "",
             f"Source: {first['source']}  ",
             f"Licence of the data: {first['license']}  ",
             f"Licence of the weights: Apache-2.0, built from D-FINE, PP-HGNetV2 and (masks) "
             f"MobileSAM, all Apache-2.0 ([NOTICE]({NOTICE})). The data's terms above still "
             "apply: keep its credit, and check them before commercial use.", ""]
    if first.get("attribution"):
        lines += [f"Credit (keep it with the model): {first['attribution']}", ""]
    if first.get("citation"):
        lines += [f"Cite: {first['citation']}", ""]
    if first.get("note"):
        lines += [first["note"], ""]
    return "\n".join([*lines, f"{len(first['classes'])} classes: {names}", "",
                      "| model | task | size | score | value | 8-bit |",
                      "| --- | --- | --- | --- | --- | --- |", *rows,
                      ""])


def _fill_categories(index: dict) -> None:
    """Every entry with its category (the web page groups by it) and its task
    and score, including entries published before those were recorded."""
    sys.path.insert(0, str(HERE))
    from hub_datasets import category

    for key, entry in index.items():
        entry["category"] = category(key.split("/", 1)[0])
        entry.setdefault("task", "detect")
        entry.setdefault("metric", TASKS[entry["task"]][2])
        entry.setdefault("score", entry.get("map50_95"))
        entry.setdefault("score_int8", entry.get("map50_95_int8"))


def task_of(index: dict, dataset: str) -> list[str]:
    """The tasks a dataset has models for, in TASKS' order."""
    have = {m.get("task", "detect") for k, m in index.items() if k.split("/", 1)[0] == dataset}
    return [t for t in TASKS if t in have]


@contextlib.contextmanager
def _publishing(stage: Path):
    """One process at a time on the hub's list and front page."""
    import fcntl

    stage.mkdir(parents=True, exist_ok=True)
    with open(stage / ".publish.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def refresh_thumbs(repo: str, stage: Path, dry_run: bool) -> int:
    """Every dataset's thumb.jpg on the hub cut to THUMB, and the front page
    rebuilt, in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    with _publishing(stage):
        return _refresh_thumbs(repo, stage, dry_run, CommitOperationAdd, HfApi, hf_hub_download)


def _refresh_thumbs(repo, stage, dry_run, CommitOperationAdd, HfApi, hf_hub_download) -> int:
    import cv2

    index = json.loads(Path(hf_hub_download(repo, "index.json", force_download=True)).read_text())
    ops = []
    for dataset in _by_dataset(index):
        thumb = Path(hf_hub_download(repo, f"{dataset}/thumb.jpg", force_download=True))
        out = stage / dataset / "thumb.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), fit(cv2.imread(str(thumb))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        ops.append(CommitOperationAdd(f"{dataset}/thumb.jpg", str(out)))
        print(f"{dataset}: {THUMB[0]}x{THUMB[1]}", flush=True)
    _fill_categories(index)
    (stage / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    ops.append(CommitOperationAdd("index.json", str(stage / "index.json")))
    (stage / "README.md").write_text(front_readme(index, repo))
    ops.append(CommitOperationAdd("README.md", str(stage / "README.md")))
    if not dry_run:
        HfApi().create_commit(repo, ops, commit_message="thumbnails: one size")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", help="best.pt of the training")
    parser.add_argument("--data", help="its data.yaml (val split scored)")
    parser.add_argument("--dataset", help="folder name in the repo, e.g. voc")
    parser.add_argument("--refresh-thumbs", action="store_true",
                        help="only cut every published thumbnail to the same size and "
                             "rebuild the front page")
    parser.add_argument("--score-data", help="data.yaml whose val split scores it "
                        "(default --data; RF100: its score.yaml, the test split)")
    parser.add_argument("--title", help="instead of hub_datasets.py's")
    parser.add_argument("--license", help="instead of hub_datasets.py's")
    parser.add_argument("--source", help="instead of hub_datasets.py's")
    parser.add_argument("--size", type=int, help="input size (default: the checkpoint's)")
    parser.add_argument("--int8", help="an 8-bit .onnx made already (e.g. quantization-aware) "
                        "instead of quantizing the float one; 320 px or less")
    parser.add_argument("--task", default="detect", choices=list(TASKS),
                        help="segment / pose: the detector with --head beside it")
    parser.add_argument("--head", help="--task segment: mask_decoder.onnx; pose: pose.onnx")
    parser.add_argument("--truth", help="--task segment: hub_tasks.py's val.json to score on")
    parser.add_argument("--coco", default="~/datasets/coco",
                        help="--task pose: COCO 2017 (val2017 person keypoints) to score on")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--stage", default="~/hub-stage", help="where the files are built")
    parser.add_argument("--limit", type=int, help="val pictures to score (a quick try)")
    parser.add_argument("--dry-run", action="store_true", help="build, do not upload")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if args.refresh_thumbs:
        return refresh_thumbs(args.repo, Path(args.stage).expanduser(), args.dry_run)
    if not (args.weights and args.data and args.dataset):
        parser.error("--weights, --data and --dataset are needed")
    if args.task != "detect" and not args.head:
        parser.error(f"--task {args.task} needs --head")
    if args.task == "segment" and not args.truth:
        parser.error("--task segment needs --truth (hub_tasks.py seg-data writes it)")

    import torch

    sys.path.insert(0, str(HERE))
    from hub_datasets import category, lookup

    facts = lookup(args.dataset)
    for field in ("title", "license", "source"):
        if getattr(args, field):
            facts[field] = getattr(args, field)
    weights = Path(args.weights).expanduser()
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    size = args.size or int(ckpt.get("imgsz") or 640)
    variant = ckpt.get("variant", "s")
    key = f"{args.dataset}/ed-{variant}-{size}{TASKS[args.task][0]}"
    stage = Path(args.stage).expanduser()
    folder = stage / key
    score_data = str(Path(args.score_data).expanduser()) if args.score_data else None
    int8_from = Path(args.int8).expanduser() if args.int8 else None
    info = build(weights, str(Path(args.data).expanduser()), folder, size, args.limit, score_data,
                 int8_from, task=args.task,
                 head=Path(args.head).expanduser() if args.head else None,
                 truth=str(Path(args.truth).expanduser()) if args.truth else None,
                 coco=Path(args.coco).expanduser())
    info.update(facts, category=category(args.dataset))
    (folder / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "classes"}, ensure_ascii=False))

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi()
    if not args.dry_run:
        api.upload_folder(folder_path=str(folder), path_in_repo=key, repo_id=args.repo,
                          commit_message=f"{key}")
    # the list and the front page: one publisher at a time, or two at once
    # would each write the list without the other's model
    with _publishing(stage):
        try:
            index = json.loads(Path(hf_hub_download(args.repo, "index.json",
                                                    force_download=True)).read_text())
        except EntryNotFoundError:
            index = {}
        index[key] = info
        _fill_categories(index)
        (stage / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
        (stage / "README.md").write_text(front_readme(index, args.repo))
        # the dataset's picture is its detector's (a mask or keypoint model
        # draws it only for a dataset that has no detector of its own)
        own_thumb = args.task == "detect" or "detect" not in task_of(index, args.dataset)
        if own_thumb:
            shutil.copyfile(folder / "sample.jpg", stage / args.dataset / "thumb.jpg")
        (stage / args.dataset / "README.md").write_text(dataset_readme(args.dataset, index))
        if args.dry_run:
            print(f"built {folder} (not uploaded)")
            return 0
        ops = [CommitOperationAdd(name, str(stage / name)) for name in
               ("index.json", "README.md", f"{args.dataset}/README.md",
                *([f"{args.dataset}/thumb.jpg"] if own_thumb else []))]
        api.create_commit(args.repo, ops, commit_message=f"{key}: list and pages")
    print(f"https://huggingface.co/{args.repo}/tree/main/{key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
