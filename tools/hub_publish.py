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

* ``ed-<v>-<size>.pt``    the checkpoint without its optimizer, to fine-tune from
* ``ed-<v>-<size>.onnx``  ONNX Runtime, class names and the F1-best threshold inside
* ``ed-<v>-<size>.xml/.bin``  OpenVINO IR (Intel CPU, GPU, NPU)
* ``ed-<v>-<size>-int8.onnx`` at 320 px or less: 8 bits for a Raspberry Pi or Pibo
  (convolutions only, the stem in float, as Pibo's own model)
* ``model.json``          dataset, classes, size, COCO-style mAP50-95 on its val split

Then ``index.json`` (every model in the repo), ``<dataset>/README.md`` and the
front ``README.md`` are rebuilt from it and uploaded. ``--dry-run`` stops
before uploading.
"""

from __future__ import annotations

import argparse
import json
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


def thumbnail(onnx: Path, data: str, out: Path, look: int = 200) -> Path:
    """One val picture with the model's boxes on it, its longer side 640 px:
    of the first ``look``, the one with the most boxes found, up to 20."""
    import cv2

    from easydetect import Detector
    from easydetect.data.dataset import DetDataset

    model = Detector(str(onnx), verbose=False)
    best, most = None, -1
    for f in DetDataset(data, "val", 64, augment=False).files[:look]:
        result = model(str(f), verbose=False)[0]
        found = len(result.boxes)
        if most < found <= 20:
            best, most = result, found
    image = best.plot(line_width=2)
    scale = 640 / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def build(weights: Path, data: str, folder: Path, size: int, limit: int | None,
          score_data: str | None = None) -> dict:
    sys.path.insert(0, str(HERE))
    from pi_kit import ort_int8, score
    from pibo_conf import pick

    from easydetect import Detector
    from easydetect.data.dataset import DetDataset
    from easydetect.exporter import embed_metadata

    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    name = folder.name
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

    conf, _ = pick(onnx, data, limit)
    embed_metadata(onnx, {"easydetect.conf": f"{conf:.2f}"})
    score_data = score_data or data
    scores = {"onnx": score(onnx, score_data, limit, backend="onnxruntime")}
    files = [f"{name}.pt", f"{name}.onnx", f"{name}.xml", f"{name}.bin"]
    if size <= 320:
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
    thumbnail(onnx, score_data, folder / "sample.jpg")
    files.append("sample.jpg")

    names = ckpt["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    return {
        "model": name.rsplit("-", 1)[0], "imgsz": size, "files": files,
        "classes": {int(k): v for k, v in names.items()}, "conf": round(conf, 2),
        "map50_95": scores["onnx"]["map_official"],
        "map50_95_int8": scores.get("int8", {}).get("map_official"),
        "queries": ckpt.get("queries"), "epochs": ckpt.get("epochs"),
    }


def _by_dataset(index: dict) -> dict:
    grouped = {}
    for key, m in sorted(index.items()):
        grouped.setdefault(key.split("/", 1)[0], {})[key] = m
    return grouped


def front_readme(index: dict, repo: str = REPO) -> str:
    rows = []
    for dataset, models in _by_dataset(index).items():
        first = next(iter(models.values()))
        thumb = f"https://huggingface.co/{repo}/resolve/main/{dataset}/thumb.jpg"
        cells = []
        for size in (320, 640):
            m = next((m for m in models.values() if m["imgsz"] == size), None)
            cells.append("—" if m is None else f"{m['map50_95']:.3f}" + (
                f" (8-bit {m['map50_95_int8']:.3f})" if m.get("map50_95_int8") else ""))
        rows.append(f"| <img src=\"{thumb}\" width=\"160\"> | [{first['title']}]({dataset}/) | "
                    f"{len(first['classes'])} | {cells[0]} | {cells[1]} | {first['license']} |")
    example = next(iter(sorted(index)), "construction-ppe/ed-s-640")
    return "\n".join([
        "---", "license: apache-2.0", "library_name: easydetect",
        "pipeline_tag: object-detection", "tags:", "  - object-detection", "  - detr",
        "  - onnx", "  - openvino", "---", "",
        "# easydetect hub — detectors trained on public datasets", "",
        "Models trained with [easydetect](https://github.com/themakerrobot/easydetect) on",
        "public datasets whose terms allow commercial use. Each folder holds the",
        "checkpoint to fine-tune from, an ONNX file for ONNX Runtime (class names and",
        "threshold inside), OpenVINO IR, and at 320 px an 8-bit ONNX for a Raspberry Pi.",
        "", "```bash", "pip install easydetect huggingface_hub",
        f"hf download {repo} --include '{example}/*' --local-dir hub",
        "```", "", "```python", "from easydetect import Detector", "",
        f"model = Detector(\"hub/{example}/{example.split('/')[1]}.onnx\")",
        "model(\"photo.jpg\")[0].show()", "```", "",
        "ed-s, mAP50-95 on each dataset's own val split, COCO's way.", "",
        "| | dataset | classes | 320 px | 640 px | dataset licence |",
        "| --- | --- | --- | --- | --- | --- |", *rows, "",
        "The weights are Apache-2.0. Each dataset's licence and attribution are in its",
        "folder's README; keep them when you use a model trained on it.", ""])


def dataset_readme(dataset: str, index: dict) -> str:
    models = {k: m for k, m in index.items() if k.startswith(dataset + "/")}
    first = next(iter(models.values()))
    names = ", ".join(first["classes"].values())
    rows = [f"| `{k.split('/', 1)[1]}` | {m['imgsz']} | {m['map50_95']:.3f} | "
            f"{m['map50_95_int8']:.3f} |" if m.get("map50_95_int8") else
            f"| `{k.split('/', 1)[1]}` | {m['imgsz']} | {m['map50_95']:.3f} | — |"
            for k, m in sorted(models.items())]
    lines = [f"# {first['title']}", "", "![sample](thumb.jpg)", "",
             f"Source: {first['source']}  ",
             f"Licence: {first['license']}", ""]
    if first.get("attribution"):
        lines += [f"Credit (keep it with the model): {first['attribution']}", ""]
    if first.get("citation"):
        lines += [f"Cite: {first['citation']}", ""]
    if first.get("note"):
        lines += [first["note"], ""]
    return "\n".join([*lines, f"{len(first['classes'])} classes: {names}", "",
                      "| model | size | mAP50-95 | 8-bit |", "| --- | --- | --- | --- |", *rows,
                      ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help="best.pt of the training")
    parser.add_argument("--data", required=True, help="its data.yaml (val split scored)")
    parser.add_argument("--dataset", required=True, help="folder name in the repo, e.g. voc")
    parser.add_argument("--score-data", help="data.yaml whose val split scores it "
                        "(default --data; RF100: its score.yaml, the test split)")
    parser.add_argument("--title", help="instead of hub_datasets.py's")
    parser.add_argument("--license", help="instead of hub_datasets.py's")
    parser.add_argument("--source", help="instead of hub_datasets.py's")
    parser.add_argument("--size", type=int, help="input size (default: the checkpoint's)")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--stage", default="~/hub-stage", help="where the files are built")
    parser.add_argument("--limit", type=int, help="val pictures to score (a quick try)")
    parser.add_argument("--dry-run", action="store_true", help="build, do not upload")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    import torch

    sys.path.insert(0, str(HERE))
    from hub_datasets import lookup

    facts = lookup(args.dataset)
    for field in ("title", "license", "source"):
        if getattr(args, field):
            facts[field] = getattr(args, field)
    weights = Path(args.weights).expanduser()
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    size = args.size or int(ckpt.get("imgsz") or 640)
    variant = ckpt.get("variant", "s")
    key = f"{args.dataset}/ed-{variant}-{size}"
    stage = Path(args.stage).expanduser()
    folder = stage / key
    score_data = str(Path(args.score_data).expanduser()) if args.score_data else None
    info = build(weights, str(Path(args.data).expanduser()), folder, size, args.limit, score_data)
    info.update(facts)
    (folder / "model.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "classes"}, ensure_ascii=False))

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi()
    try:
        index = json.loads(Path(hf_hub_download(args.repo, "index.json")).read_text())
    except EntryNotFoundError:
        index = {}
    index[key] = info
    (stage / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    (stage / "README.md").write_text(front_readme(index, args.repo))
    shutil.copyfile(folder / "sample.jpg", stage / args.dataset / "thumb.jpg")
    (stage / args.dataset / "README.md").write_text(dataset_readme(args.dataset, index))
    if args.dry_run:
        print(f"built {folder} (not uploaded)")
        return 0
    api.upload_folder(folder_path=str(folder), path_in_repo=key, repo_id=args.repo,
                      commit_message=f"{key}")
    for name in ("index.json", "README.md", f"{args.dataset}/README.md",
                 f"{args.dataset}/thumb.jpg"):
        api.upload_file(path_or_fileobj=str(stage / name), path_in_repo=name,
                        repo_id=args.repo, commit_message=f"{key}: {name}")
    print(f"https://huggingface.co/{args.repo}/tree/main/{key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
