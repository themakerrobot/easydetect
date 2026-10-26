# Deploying a model

What runs where, and what has been checked:

| Runtime | Devices | Status |
| --- | --- | --- |
| OpenVINO (`.xml`, `.onnx`) | Intel CPU, iGPU, NPU; other x86 and ARM CPUs | tested: every release, CI |
| ONNX Runtime (`.onnx`) | any CPU (a Raspberry Pi included) | tested: every release, CI |
| `easydetect serve` (HTTP) | wherever one of the two above runs | tested: CI |
| TensorRT, RKNN, TFLite, Core ML | NVIDIA GPU / Jetson, Rockchip NPU, mobile | **not tested** — notes below |

## An HTTP server: `easydetect serve`

```bash
easydetect serve model=best.onnx                       # http://127.0.0.1:8000
easydetect serve model=runs/train/weights/best.pt host=0.0.0.0 port=9000
easydetect serve model=ed-s task=segment
```

```bash
curl -F image=@bus.jpg http://127.0.0.1:8000/predict
curl --data-binary @bus.jpg -H "Content-Type: image/jpeg" \
     "http://127.0.0.1:8000/predict?conf=0.4&classes=0,2"
curl -F image=@bus.jpg "http://127.0.0.1:8000/predict?draw=1" -o drawn.jpg
curl http://127.0.0.1:8000/health
```

`POST /predict` takes one picture, as the raw body or as the `image` field of
a form, and answers with the same rows as `r.summary()` (numbers here for
illustration):

```json
{"detections": [{"name": "person", "class": 0, "confidence": 0.93,
                 "box": {"x1": 48.1, "y1": 398.6, "x2": 245.3, "y2": 902.4}}],
 "width": 810, "height": 1080, "ms": 41.2,
 "speed": {"preprocess": 1.9, "inference": 37.4, "postprocess": 0.6}}
```

`conf`, `iou` (`none` turns it off), `max_det` and `classes` go in the query
string; `draw=1` answers with the drawn picture as a JPEG instead. With
`task=segment` or `task=pose` each row also carries its mask or keypoints.
The model loads once at start (a named model downloads then), and requests are
served from several threads.

It is the standard library's HTTP server, so it needs nothing beyond
easydetect itself — but it has no authentication or TLS. It listens on
127.0.0.1 unless given `host=0.0.0.0`; anything reachable from outside belongs
behind a reverse proxy that has both.

In a container, for example (a sketch, not tested in CI):

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir easydetect && \
    apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
COPY best.onnx /model/best.onnx
EXPOSE 8000
CMD ["easydetect", "serve", "model=/model/best.onnx", "host=0.0.0.0", "port=8000"]
```

## The model file, for any other runtime

An exported `.onnx` (opset 17, one fixed input size) is the whole detector,
decoding included up to the scores:

- input `images`: `(1, 3, S, S)` float32, **RGB, 0–1, plain-resized** to S×S
  (no letterbox, no mean/std) — S is the export size, 640 unless chosen;
- output `boxes`: `(1, Q, 4)` box centre and size, `cx cy w h`, as fractions
  of the picture (so they map straight back to the original width and height);
- output `scores`: `(1, Q, C)` per-class probabilities, already through a
  sigmoid; Q is 300 unless exported with fewer queries.
- metadata `easydetect.names`: the class names as JSON.

Each query is one candidate object: take its best class, keep it above a
confidence, and scale. With ONNX Runtime and NumPy alone:

```python
import cv2, numpy as np, onnxruntime as ort

session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
size = session.get_inputs()[0].shape[-1]                  # 640 for the mirror's models
img = cv2.imread("bus.jpg")
h, w = img.shape[:2]
x = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB)
x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # plain resize, RGB 0-1, NCHW
boxes, scores = session.run(None, {"images": x})
cls, conf = scores[0].argmax(1), scores[0].max(1)          # one class per query
keep = conf >= 0.5
cx, cy, bw, bh = boxes[0][keep].T * [[w], [h], [w], [h]]   # fractions -> pixels
xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
```

These boxes match easydetect's own to within 1e-5 pixels; easydetect then
also drops a box that overlaps a surer one by more than IoU 0.7 (`iou=`, see
[performance.md](performance.md)) — worth doing too, but rarely needed.

## Other runtimes — not tested

These start from the same `.onnx`. None is part of easydetect's tests, so
treat the commands as a starting point. The operators that decide whether a
runtime can take the model are `GridSample` (the decoder's deformable
attention samples features at learned points), `TopK` and `GatherElements`
(choosing the queries) and `LayerNormalization` (opset 17); everything else is
convolutions and matrix products.

- **TensorRT** (NVIDIA GPUs, Jetson): `GridSample` and `LayerNormalization`
  are built-in from TensorRT 8.6. the original authors deploy this way:
  `trtexec --onnx=best.onnx --saveEngine=best.engine --fp16`. The engine
  takes and gives the same tensors as above.
- **RKNN** (Rockchip NPUs, via rknn-toolkit2): `GridSample` and `TopK` may
  not run on the NPU and fall back to the CPU, or not convert; check the
  toolkit's operator list for your chip before counting on it.
- **TFLite / LiteRT** (phones, microcontrollers): through `onnx2tf`, which
  has a `GridSample` conversion; expect to validate the outputs against
  ONNX Runtime on a few pictures, and int8 needs calibration there.
- **Core ML** (Apple): through `coremltools` from the ONNX or the PyTorch
  model; untried.

For any of them, the check that matters: run the same few pictures through
ONNX Runtime and through the converted model and compare `boxes` and
`scores` — they should agree to a few thousandths.
