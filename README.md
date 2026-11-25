<div align="center">

# easydetect

**Easy real-time object detection you can actually ship.**
Train in PyTorch, run on OpenVINO or ONNX Runtime — a CPU, an Intel GPU or NPU,
a Raspberry Pi. Apache-2.0 code and COCO weights.

[![PyPI](https://img.shields.io/pypi/v/easydetect?color=2b7489)](https://pypi.org/project/easydetect/)
[![Python](https://img.shields.io/pypi/pyversions/easydetect)](https://pypi.org/project/easydetect/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/themakerrobot/easydetect/blob/main/LICENSE)
[![CI](https://github.com/themakerrobot/easydetect/actions/workflows/ci.yml/badge.svg)](https://github.com/themakerrobot/easydetect/actions/workflows/ci.yml)

![detections on a street scene](https://raw.githubusercontent.com/themakerrobot/easydetect/main/docs/assets/demo.jpg)

</div>

## Why

* **Three lines** to detect, train and export.
* **Free for products**: Apache-2.0 code and COCO weights, no source to publish.
* **Accurate for its size**: ed-s, 0.485 COCO mAP50-95, 10M parameters, no NMS to tune.
* **Runs anywhere**: OpenVINO (CPU, Intel GPU, NPU) or ONNX Runtime (any CPU, a Raspberry Pi).

## Install

```bash
pip install easydetect              # inference: OpenVINO and ONNX Runtime
pip install "easydetect[train]"     # + training (PyTorch)
```

Inference needs no PyTorch; MobileSAM's encoder (masks) is in the package, so
nothing more is downloaded for it.

## Detect and train

```python
from easydetect import Detector

model = Detector("ed-s")           # COCO weights, downloaded on first use
r = model("photo.jpg", conf=0.5)[0]
r.boxes.xyxy, r.boxes.conf, r.boxes.cls   # plain numpy
r.save()

for r in model.predict(0, stream=True, show=True):   # webcam; also a folder,
    print(r.boxes.xyxyn)                             # a video, an RTSP stream
model.track("clip.mp4")                              # adds r.boxes.id

model.train(data="data.yaml", epochs=50, imgsz=640, device=0)
model.val(data="data.yaml").box.map                  # mAP50-95
model.export(format="openvino")                      # best.xml + .bin + labels.txt
```

```yaml
# data.yaml: one .txt per image, "cls cx cy w h" normalised (YOLO's layout)
path: /data/cans
train: images/train
val: images/val
names: {0: can, 1: bottle}
```

A YOLO-format zip (Roboflow's included) trains as it comes.
`task="segment"` adds masks (MobileSAM on each box, no mask labels needed);
`task="pose"` adds keypoints.

## Models

| name | backbone | params | COCO mAP50-95 |
| --- | --- | --- | --- |
| `ed-n` | HGNetv2-B0 | 4M | 0.428 |
| `ed-s` | HGNetv2-B0 | 10M | 0.485 — the default |
| `ed-m` | HGNetv2-B2 | 19M | 0.523 |
| `ed-l` | HGNetv2-B4 | 31M | 0.540 |
| `ed-x` | HGNetv2-B5 | 62M | 0.558 |

The original authors' numbers (640 px, COCO val2017).

### easydetect hub: models for public datasets

[easydetect hub](https://huggingface.co/leeyunjai/easydetect-hub): ed-s trained
on over a hundred public datasets, 320 px (with 8-bit) and 640 px, some with
masks or keypoints. Each follows its dataset's terms: check before commercial use.

```python
model = Detector("hub:taco/ed-s-640")          # ONNX, downloaded once
model = Detector("hub:bccd/ed-s-320-int8")     # 8 bits, for a Raspberry Pi
model = Detector("hub:ap10k/ed-s-640-pose")    # animals and their keypoints
model = Detector("hub:taco/ed-s-640.pt")       # the checkpoint, to fine-tune
```

[Browse and try them in your browser](https://themakerrobot.github.io/easydetect/).

## easydetect lab

[easydetect lab](https://github.com/themakerrobot/easydetect-lab): label, train
and test your own pictures in one app, offline too.

![easydetect lab](https://raw.githubusercontent.com/themakerrobot/easydetect-lab/main/docs/lab-en.jpg)

## Speed

ed-s at 640, whole pipeline, Core Ultra 5 250K Plus + RTX 5090 (`tools/bench.py`):

| `device=` | latency | FPS |
| --- | --- | --- |
| `"CPU"` | 36 ms | 28 |
| `"NPU"` | 38 ms | 26, and the CPU stays free |
| `"GPU"` (the RTX 5090 through OpenCL) | 14 ms | 72 |

## Compared with YOLO

mAP50-95 and OKS AP, 0 to 1.

**COCO val2017, 640 px**, as each project publishes it:

| | n | s | m | l | x |
| --- | --- | --- | --- | --- | --- |
| ed | **0.428** | **0.485** | 0.523 | 0.540 | 0.558 |
| YOLO26 (no NMS / NMS) | 0.401 / 0.409 | 0.478 / 0.486 | 0.525 / 0.531 | 0.544 / 0.550 | 0.569 / 0.575 |
| ed params | 4M | 10M | 19M | 31M | 62M |
| YOLO26 params | 2.4M | 9.5M | 20.4M | 24.8M | 55.7M |

**Keypoints**, COCO val2017 (detector + keypoint model):

| | s | m | l |
| --- | --- | --- | --- |
| easydetect (ed-s, then the keypoint model of that size) | **0.663** | **0.716** | **0.734** |
| YOLO26-pose | 0.630 | 0.688 | 0.704 |

**Fine-tuned**, each trainer's defaults (100 epochs, 640 px), one evaluator, YOLO26 → ed:

| dataset | train pictures | n | s | m |
| --- | --- | --- | --- | --- |
| african-wildlife | 1,052 | 0.817 → **0.848** | 0.830 → **0.869** | 0.814 → **0.890** |
| construction-ppe | 1,132 | **0.300** → 0.293 | 0.289 → **0.297** | 0.298 → **0.307** |
| brain-tumor | 893 | 0.417 → **0.449** | 0.357 → **0.470** | 0.360 → **0.466** |
| signature | 143 | 0.975 → **0.985** | 0.984 → **0.985** | 0.967 → **0.997** |
| medical-pills | 92 | **0.797** → 0.794 | **0.801** → 0.800 | 0.797 → **0.812** |
| african-wildlife, 100 at random | 100 | 0.711 → **0.802** | 0.674 → **0.837** | 0.613 → **0.858** |

**Roboflow 100**, 26 datasets, ≤300 pictures each, 3 seeds:

| ed against YOLO26, same size | ed ahead on | mean | median |
| --- | --- | --- | --- |
| **s, 640 px, 100 epochs** | **19 of 26** | **+0.019** | **+0.016** |
| n, 640 px, 100 epochs | 8 of 26 | −0.034 | −0.018 |
| s, 320 px, the lab's classroom preset | 6 of 26 | −0.054 | −0.049 |
| n, 320 px, the lab's classroom preset | 4 of 26 | −0.103 | −0.076 |

ed leads on large objects and few pictures; YOLO on small objects and in speed
(15–25% at n, s). Ultralytics YOLO is AGPL-3.0; this is Apache-2.0.

### Small inputs and small computers

COCO val2017, one class a box (reads below the authors' tool: ed-n 640 0.416 vs 0.428):

| input | ed-n, 640 px weights | ed-n, fine-tuned | ed-s, 640 px weights | ed-s, fine-tuned | YOLO11n | YOLO26n | YOLO11s | YOLO26s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 320 | 0.100 | **0.322** | 0.239 | **0.381** | 0.283 | 0.297 | 0.365 | **0.381** |
| 640 | **0.416** | 0.410 | **0.472** | 0.465 | 0.381 | 0.397 | 0.453 | 0.471 |

Raspberry Pi 4B, 4 threads, the model alone, same scoring:

| model | runtime | Pi 4B | COCO mAP50-95 |
| --- | --- | --- | --- |
| ed-n 320, 100 queries | OpenVINO | 122 ms | 0.322 |
| ed-s 320, 100 queries | OpenVINO | 347 ms | 0.381 |
| ed-s 320, float | ONNX Runtime | 396 ms | 0.381 |
| **ed-s 320, QAT 8-bit (Pibo's default)** | **ONNX Runtime** | **318 ms** | **0.376** |
| ed-s 640, QAT 8-bit (Pibo, accuracy first) | ONNX Runtime | 1,058 ms | 0.454 |
| YOLO11s 320 | ONNX Runtime | 300 ms | 0.365 |

## Command line

```bash
easydetect predict model=ed-s source=photo.jpg conf=0.5
easydetect train   model=ed-s data=data.yaml epochs=50
easydetect val     model=best.pt data=data.yaml
easydetect export  model=best.pt format=openvino half=true
easydetect hub     search=plant
```

## Docs

[Using the model](https://github.com/themakerrobot/easydetect/blob/main/docs/usage.md) ·
[Training](https://github.com/themakerrobot/easydetect/blob/main/docs/training.md) ·
[Keypoints](https://github.com/themakerrobot/easydetect/blob/main/docs/pose.md) ·
[Weights and offline use](https://github.com/themakerrobot/easydetect/blob/main/docs/weights.md) ·
[Performance](https://github.com/themakerrobot/easydetect/blob/main/docs/performance.md) ·
[Deploying](https://github.com/themakerrobot/easydetect/blob/main/docs/deploy.md) ·
[Design](https://github.com/themakerrobot/easydetect/blob/main/docs/design.md) ·
[Development](https://github.com/themakerrobot/easydetect/blob/main/docs/contributing.md)

Not here: classification, oriented boxes, multi-GPU training.

## 한국어

- 설치: `pip install easydetect`(추론) · `pip install "easydetect[train]"`(학습)
- 사용: `Detector("ed-s")("photo.jpg")`, 학습은 `model.train(data="data.yaml")`
- 라이선스: 코드·COCO 가중치 Apache-2.0(상용 가능). hub 모델은 데이터셋 조건을 따름
- **파이보 기본 모델: ed-s 320 QAT 8비트, 파이보 4B 318 ms, COCO 0.376** (YOLO11s 300 ms, 0.365)
- 내 사진으로 만들기: [easydetect lab](https://github.com/themakerrobot/easydetect-lab)

## Credits

The detector and its COCO weights come from
[D-FINE](https://github.com/Peterande/D-FINE) (Apache-2.0,
[arXiv:2410.13842](https://arxiv.org/abs/2410.13842)), which builds on
[RT-DETR](https://github.com/lyuwenyu/RT-DETR). This package adapts that network
and loss, and adds its own training loop, inference stack and tooling — see
[NOTICE](https://github.com/themakerrobot/easydetect/blob/main/NOTICE).

## License

Apache-2.0 — see [LICENSE](https://github.com/themakerrobot/easydetect/blob/main/LICENSE) and [NOTICE](https://github.com/themakerrobot/easydetect/blob/main/NOTICE).
The COCO weights are Apache-2.0 too, for any use, commercial included (credit
COCO, whose annotations are CC BY 4.0), and so is a model you train from them on
your own data. easydetect hub's models are the exception: each follows its
dataset's terms, shown on its page, to check before commercial use.
