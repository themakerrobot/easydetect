---
license: apache-2.0
library_name: easydetect
pipeline_tag: object-detection
tags:
  - object-detection
  - d-fine
  - detr
  - openvino
  - npu
  - coco
---

# easydetect weights — D-FINE for `pip install easydetect`

Real-time object detectors, ready to run: **ONNX** for ONNX Runtime on any CPU,
**OpenVINO IR** for an Intel CPU, GPU or NPU, and the **PyTorch checkpoint** to
fine-tune from. These
are the files the [easydetect](https://github.com/themakerrobot/easydetect)
package downloads on first use — you never have to fetch them by hand.

Apache-2.0 end to end: the code, the COCO weights, and the models fine-tuned here.

```bash
pip install easydetect          # OpenVINO and ONNX Runtime, no PyTorch
```

```python
from easydetect import Detector

model = Detector("dfine-s")                 # downloads dfine-s/ from this repo once
results = model("photo.jpg", conf=0.5)
results[0].save("result.jpg")

for r in model.predict(0, stream=True, show=True):   # webcam; q or Esc quits
    pass
```

No NMS step to tune: D-FINE predicts its set of boxes end to end; the package
drops the rare second box on one object (IoU > 0.7) and handles resize, decode
and drawing.

## COCO models — 80 classes

| folder | backbone | decoder | params | COCO mAP50-95 |
| --- | --- | --- | --- | --- |
| [`dfine-n`](./dfine-n) | HGNetv2-B0 | 3 layers | 4M | 42.8 |
| [`dfine-s`](./dfine-s) | HGNetv2-B0 | 3 layers | 10M | 48.5 |
| [`dfine-m`](./dfine-m) | HGNetv2-B2 | 4 layers | 19M | 52.3 |
| [`dfine-l`](./dfine-l) | HGNetv2-B4 | 6 layers | 31M | 54.0 |
| [`dfine-x`](./dfine-x) | HGNetv2-B5 | 6 layers | 62M | 55.8 |

Every folder holds the same files:

| file | what it is |
| --- | --- |
| `<name>.xml` + `<name>.bin` | OpenVINO IR — what `Detector("<name>")` runs on OpenVINO |
| `<name>.onnx` | the same network for ONNX Runtime — the light install, e.g. a Raspberry Pi |
| `<name>.pt` | PyTorch checkpoint — the starting point for `model.train(...)` |
| `labels.txt` | class names, one per line |

`dfine-s` at 640 × 640 on a Core Ultra 5 250K Plus, whole pipeline:

| device | latency |
| --- | --- |
| `CPU` | 36 ms (28 FPS) |
| `NPU` | 38 ms (26 FPS) — the CPU stays free |

More in [performance](https://github.com/themakerrobot/easydetect/blob/main/docs/performance.md).

## Fine-tuned models — [`models/`](./models)

Detectors trained on one job, each with its own README: classes, scores, the data
it learned from, and how it was trained.

| folder | finds | base | val mAP50-95 |
| --- | --- | --- | --- |

<!-- add a row per model folder; its README.md already has the numbers -->

Use one with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
from easydetect import Detector

root = snapshot_download("leeyunjai/easydetect", allow_patterns="models/<name>/*")
model = Detector(f"{root}/models/<name>/best.xml", device="AUTO")   # CPU · GPU · NPU
model.predict("photo.jpg", save=True)
```

## Train your own

```bash
pip install "easydetect[train]"
```

```python
from easydetect import Detector

model = Detector("dfine-s")                        # starts from the COCO weights above
model.train(data="data.yaml", epochs=50, imgsz=640)
model.export(format="openvino")                    # best.xml + best.bin + labels.txt
```

`data.yaml` is the common images/ + labels/ layout — a Roboflow export works as
it is. Or do it all in a browser: the
[easydetect lab](https://github.com/themakerrobot/easydetect-lab) collects
and labels images, trains, shows the numbers, and writes the upload folder for
`models/` with its README filled in from the run.

## Layout

```
dfine-n/   dfine-n.xml  dfine-n.bin  dfine-n.pt  labels.txt
dfine-s/   …
dfine-m/   …
dfine-l/   …
dfine-x/   …
mobile_sam/  encoder.onnx  decoder.onnx  LICENSE    the segmenter (task="segment")
pose/        pose-s.onnx  pose-m.onnx              the keypoint models (task="pose")
models/
  <name>/   best.xml  best.bin  labels.txt  README.md
```

`mobile_sam/` is MobileSAM (Apache-2.0,
[ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)) exported
to ONNX by `tools/convert_sam.py`: given the boxes a detector found, it outlines
what is inside each.

`pose/` is easydetect's own keypoint network (Apache-2.0, 3.7 M parameters),
trained by `tools/train_pose.py` on COCO 2017 person keypoints (CC BY 4.0) from
the dfine-s backbone: given a person box, it places COCO's 17 body keypoints.
OKS AP on COCO val2017, reading each person mirrored too (the package's
default): 0.637 with the labelled boxes, 0.588 behind dfine-s.

Keep these names: the package builds its download URLs from them
(`<repo>/resolve/main/<name>/<name>.xml`). To host a copy elsewhere, mirror the
same layout and point `$EASYDETECT_ASSETS_URL` at it.

## License and credit

* The COCO weights are the official D-FINE checkpoints by Yansong Peng et al.,
  released under Apache-2.0 at [Peterande/D-FINE](https://github.com/Peterande/D-FINE),
  converted unchanged. Only the COCO-trained checkpoints are used. COCO
  annotations are CC BY 4.0.
* The package and the fine-tuned models are Apache-2.0. Each model's README says
  where its training images came from; those images keep their own license.

```bibtex
@misc{peng2024dfine,
  title         = {D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
  author        = {Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
  year          = {2024},
  eprint        = {2410.13842},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
