# Sizes and speed

Back to the [README](../README.md).

## Which size

| size | backbone | decoder layers | params | COCO mAP50-95 |
| --- | --- | --- | --- | --- |
| `ed-n` | HGNetv2-B0, two levels | 3 | 4M | 42.8 |
| `ed-s` | HGNetv2-B0 | 3 | 10M | 48.5 |
| `ed-m` | HGNetv2-B2 | 4 | 19M | 52.3 |
| `ed-l` | HGNetv2-B4 | 6 | 31M | 54.0 |
| `ed-x` | HGNetv2-B5 | 6 | 62M | 55.8 |

`ed-s` is the default: the accuracy of a much larger model at a size a CPU
handles. `ed-n` is for small boards; `m`/`l`/`x` when a GPU or an NPU does
the work and every point of mAP matters.

## Measure your machine

```bash
python tools/bench.py                             # ed-s on CPU, GPU, NPU
python tools/bench.py --models ed-n ed-s ed-m
python tools/bench.py --models runs/train/weights/best.pt --devices CPU NPU
```

Whole pipeline (resize, inference, decode), median of 30 calls, and every
device's boxes checked against the CPU's.

## Measured

`ed-s` at 640 on a Core Ultra 5 250K Plus (18 cores, no bfloat16) with an RTX
5090, `python tools/bench.py` — whole pipeline, median of 30 calls, each device
checked against the CPU:

| `device=` | latency | FPS | vs CPU |
| --- | --- | --- | --- |
| `"CPU"` | 35.7 ms | 28.0 | reference |
| `"NPU"` | 37.8 ms | 26.5 | mean IoU 0.984 (worst 0.946) |
| `"GPU"` (the RTX 5090, see below) | 13.8 ms | 72.3 | identical |

The network alone is about 2 ms of that less (34 / 37 / 13 ms): resizing and
decoding cost little.

Against RT-DETR r18, which this package used to ship, timed the same way
(network only) on a 4-core cloud CPU without bfloat16: `ed-s` 116 ms, r18
198 ms — faster, more accurate (48.5 against 46.4 COCO mAP) and half the size.

* **The NPU runs the whole model**, deformable attention and the box
  distribution included, while leaving the CPU free. Its fp16 arithmetic moves
  boxes by a few pixels at most. For a box that runs detection all day, this is
  the device to use.
* **The `"GPU"` row is the 5090 reached through NVIDIA's OpenCL driver**,
  because OpenVINO found no Intel GPU on that machine. It works and matches the
  CPU, but it is not a supported combination and leaves most of the card unused
  — expect lines of `105 warnings generated.` from the OpenCL compiler on first
  load; they are harmless. For inference on an NVIDIA card,
  export to ONNX (`format="onnx"`) and use onnxruntime-gpu or TensorRT.

## Precision

OpenVINO picks the execution precision itself: on a CPU with bfloat16 it uses
that — much faster than fp32, at a small cost in exactness that can flip a
borderline detection near the threshold. Ask for exactness when you need it:

```python
Detector("ed-s", precision="f32")   # matches PyTorch to float noise
```

On a CPU without bfloat16 (like the one above) this changes nothing. When you
are comparing against a reference or debugging a threshold, use `f32`.

## Making it faster

1. **Shrink the input — after training at that size.** Compute scales with the
   pixel count: 320 is about a quarter of 640. But the COCO weights were trained
   at 640, and exporting them at 320 as they are loses most of their accuracy
   (see [Small inputs](#small-inputs-320) below). Train at the size you will
   run: `model.train(..., imgsz=320)`, and the IR it exports is 320 too.
2. **Pick the size.** `ed-n` is roughly half of `ed-s`'s work.
3. **Use another device.** `Detector("ed-s", device="NPU")` on a Core
   Ultra, `device="GPU"` for Intel graphics. `ov.Core().available_devices`
   lists what OpenVINO can see; see [NPU on Linux](#npu-on-linux) if the NPU
   is missing.
4. **Skip frames.** `predict(0, vid_stride=2, ...)` runs every other frame.
5. **FP16 is about size, not CPU speed.** `half=True` halves the weight file and
   helps on GPU; on a CPU it makes little difference.
6. **Fewer queries or decoder layers.** The decoder refines 300 candidate
   boxes through 3-6 layers (by size); each layer is trained to answer on its
   own, so an export can stop early or start from fewer candidates:
   `model.export(format="openvino", queries=100)` or `layers=1`. A scene with
   a handful of objects rarely needs 300. On the 4-core Xeon, float32:

   | | as trained | `queries=100` | `layers=1` |
   | --- | --- | --- | --- |
   | ed-n at 320 | 19.9 ms | 13.2 ms | 13.1 ms |
   | ed-s at 640 | 109.8 ms | 96.5 ms | 90.1 ms |

   What it costs in accuracy depends on the scene; measure on yours (below).
7. **INT8.** `model.export(format="openvino", int8=True, data="data.yaml")`
   calibrates on 300 of the training pictures and writes `best_int8.xml`.
   Where the CPU has bfloat16 (Xeon with AMX) OpenVINO already uses it and
   INT8 adds little; elsewhere — most laptops — it replaces float32: ed-s
   at 640 went from 106 ms to 45 ms, ed-n at 320 from 17 ms to 12 ms.
   Check the accuracy on your own validation pictures before shipping it.

`tools/eval_exports.py` exports each variant from one checkpoint (a COCO
name or your `best.pt`) and scores them on the same val pictures — mAP,
precision and recall at the default conf 0.5, and speed:

```bash
python tools/eval_exports.py --data data.yaml --model runs/train/weights/best.pt
python tools/eval_exports.py --data ~/datasets/coco/data.yaml --model ed-n --imgsz 320
```

## Small inputs (320)

Measured on COCO val2017 (5,000 pictures, `easydetect val` and
`tools/compare_yolo.py`, the same evaluator for every row), all at 320:

| model | how | mAP50-95 | mAP50 | model time, Core Ultra 5 CPU |
| --- | --- | --- | --- | --- |
| ed-n | COCO weights (trained at 640), run at 320 | 0.099 | 0.302 | |
| ed-n | + fine-tuned 12 epochs on COCO at 320 | 0.324 | 0.482 | 5.5 ms |
| YOLO11s | its COCO weights, exported at 320 | 0.374 | 0.520 | 7.2 ms + NMS |

For reference, ed-n at 640 scores 0.419 here. Two lessons:

* **Train at the size you deploy.** Fine-tuning at 320 took the score from
  0.099 to 0.313 in one epoch; the rest of the twelve added 0.011, so a few
  epochs are enough. `tools/coco2yolo.py` converts COCO for this.
* **At 320 on a CPU, a YOLO of similar speed is the more accurate choice.**
  ed-n is the smaller model (4M parameters against 9.4M) and a quarter
  faster here, but 0.05 lower. ed's lead is at 640 and on GPUs and NPUs.

`tools/compare_yolo.py` repeats the comparison for your own YOLO export and
checkpoints: `python tools/compare_yolo.py --data coco/data.yaml --yolo
yolo11s.onnx --ed runs/ed-n-320/weights/best.pt`.

## NPU on Linux

The kernel side ships with recent kernels: `lsmod | grep intel_vpu` and a
`/dev/accel/accel0` owned by the `render` group mean it is there. OpenVINO also
needs Intel's user-space driver and the Level Zero loader; without either, the
NPU is simply absent from `available_devices`, with no error.

```bash
# the ubuntu2404 tarball from https://github.com/intel/linux-npu-driver/releases
tar xzf linux-npu-driver-*-ubuntu2404.tar.gz
sudo apt install -y libtbb12 libze1      # libze1 is the Level Zero loader
sudo dpkg -i intel-fw-npu_*.deb intel-driver-compiler-npu_*.deb intel-level-zero-npu_*.deb
sudo usermod -aG render $USER            # then log in again
python -c "import openvino as ov; print(ov.Core().available_devices)"   # ... 'NPU'
```

`libze1` is the step most often missed: the NPU packages install cleanly
without it, and the device still does not appear. If Ubuntu's `libze1` is too
old for the driver, take `libze1_*.deb` from
[oneapi-src/level-zero releases](https://github.com/oneapi-src/level-zero/releases).

## Confidence and overlap defaults

What `predict` shows by default — `conf=0.5`, NMS at `iou=0.7`, `contain`
off — comes from one pass of `ed-m` over COCO val2017 (5,000 pictures,
36,334 boxes), every filter applied to the same raw boxes. A shown box is
right when it matches an unclaimed truth box of its class at IoU 0.5;
precision is the share of shown boxes that are right, recall the share of
truth boxes found.

| conf | precision | recall | boxes shown |
| --- | --- | --- | --- |
| 0.25 | 32% | 81% | 91,170 |
| 0.30 | 41% | 79% | 70,279 |
| 0.40 | 57% | 73% | 47,032 |
| **0.50** | **70%** | **67%** | 34,880 |
| 0.60 | 80% | 60% | 27,281 |

(NMS at 0.7.) ed gives an unsure box 0.3–0.5, where YOLO would give it
under 0.25, so YOLO's customary 0.25 here shows two wrong boxes for every
right one; 0.5 is where precision and recall meet. COCO leaves many real
objects unlabelled, so on a real picture precision runs somewhat higher.

| NMS `iou` | mAP50-95 | precision / recall at conf 0.5 |
| --- | --- | --- |
| none | 0.5152 | 68.7% / 67.6% |
| 0.5 | 0.5028 | 71.2% / 66.4% |
| 0.6 | 0.5079 | 70.5% / 66.8% |
| **0.7** | 0.5107 | 69.9% / 67.1% |
| 0.8 | 0.5118 | 69.5% / 67.2% |

Three in four boxes NMS removes are wrong ones, so it stays on; a lower
threshold buys under a point of precision for more lost mAP and more risk to
real overlapping objects, so it stays at 0.7, Ultralytics' value too.
`contain=0.8` costs 1.7 points of mAP (0.4936) and, at any recall, shows
more wrong boxes than simply raising `conf` would — COCO is full of real
objects inside others of their class — so it stays off, for scenes like the
half hidden chair below. `val` and the mAP a run reports use no threshold and
no filter.

## Duplicate boxes

ed has no NMS step: one-to-one matching during training teaches it to give
each object one query, and mostly it does. Below its confident answers,
though, a second query can land on the same object, often under a
neighbouring class. On eight test pictures, `ed-s` at `conf=0.25` gave 194
boxes with 20 pairs overlapping by more than IoU 0.7 — one vehicle as `truck`
0.83 and `car` 0.57 (IoU 0.93), one fridge three times — and every pair was
one object twice. Raising `conf` to 0.5 cleared the same-class pairs but not
the truck/car ones.

So `predict` (and `track`) keep the best box and drop any box that covers it by
more than `iou=0.7`, whatever the class — the same name and default as
Ultralytics' NMS threshold, applied here to ed's rare duplicates: 194 boxes became 176 with no pair left, and at
0.5, 66 became 60. Separate objects rarely overlap that much; for a scene where
they do, change or switch it off:

```python
model.predict("crowd.jpg", iou=0.85)   # only near-identical boxes merge
model.predict("crowd.jpg", iou=None)   # every box the model gave
```

IoU has a blind spot: a piece inside a whole. A chair half hidden behind a
person came back from `ed-m` as the whole chair (0.64) plus its two visible
pieces (0.67, 0.61); each piece overlaps the whole by an IoU of only its share
of the area, so NMS at any usual threshold keeps all three (YOLO's NMS would
too — it measures overlap the same way). `contain=0.8` measures it over the
smaller box instead: a box of the same class with 80% of the smaller one's area
shared is one object. The inner box goes when the enclosing one is about as
sure (within 0.1), so the whole chair stays and its pieces go; the enclosing
one goes when it is much less sure than the box inside, so a loose box around
two people cannot erase them. It is off by default because a real object
inside another of its class — a child held by an adult — goes too:

```python
model.predict(0, stream=True, show=True, contain=0.8)   # one box per chair
```

`val` and the mAP a run reports use the unfiltered boxes, as the original own
numbers do.
