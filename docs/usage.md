# Using the model

Everything `Detector(...)` can be pointed at, and everything a prediction gives
back. See also [training](training.md), [weights](weights.md),
[performance](performance.md).

## Predicting

Anything you can point it at:

| Source | Example |
| --- | --- |
| image | `model("bus.jpg")` |
| glob | `model("frames/*.jpg")` |
| folder | `model("dataset/images")` |
| list file | `model("images.txt")` |
| URL | `model("https://example.com/bus.jpg")` |
| video | `model("clip.mp4")` |
| stream | `model("rtsp://camera/live")` |
| webcam | `model(0)` |
| array | `model(numpy_bgr)` / `model(pil_image)` |
| list | `model(["a.jpg", "b.jpg"])` |

```python
model = Detector("dfine-s", device="NPU")     # AUTO, CPU, GPU, NPU
model = Detector("dfine-s", precision="f32")  # exact, ~3x slower on CPU

for r in model.predict("clip.mp4", conf=0.4, stream=True):   # generator, O(1) memory
    print(r.boxes.xyxyn)

model.predict("bus.jpg", save=True)     # writes runs/detect/predict/bus.jpg
model.predict("bus.jpg", iou=None)      # keep a second box on the same object (default 0.7 drops it)
model.predict("desk.jpg", contain=0.8)  # also merge the pieces of a half hidden object (off by default)
model.track("clip.mp4")                 # IoU tracker -> r.boxes.id (match_iou=0.3, max_age=30)
for r in model.predict(0, stream=True, show=True):  # webcam window, q or Esc quits
    pass                                           # (a generator only runs when iterated)
```

### Runtimes

Two runtimes can run the model, with the same preprocessing and decoding, so
the boxes match:

| `backend=` | reads | devices | installed size |
| --- | --- | --- | --- |
| `"openvino"` | `.xml`, `.onnx` | CPU, Intel GPU, Intel NPU | 180 MB |
| `"onnxruntime"` | `.onnx` | CPU | 67 MB |

Both come with `pip install easydetect`.

Left out, `Detector` uses OpenVINO when it is installed, ONNX Runtime
otherwise; `$EASYDETECT_BACKEND` sets the default. A named model downloads the
`.xml` or the `.onnx` to suit, a `.pt` exports whichever the runtime reads, and
an `.xml` on a machine without OpenVINO runs from the `.onnx` every export
writes beside it.

```python
model = Detector("dfine-s", backend="onnxruntime")
model = Detector("runs/train/weights/best.onnx")      # either runtime
```

An exported `.onnx` carries its class names inside it, so the file works on
its own: one download from a Hugging Face page, `Detector("best.onnx")`, and
the boxes come back named — on OpenVINO or ONNX Runtime. A
`<stem>.names.json` beside it still comes first; then the names inside; then a
`labels.txt` in its folder (the IR's `.xml` + `.bin` read their names from
there).

On an x86 CPU in float32 the two are close (dfine-s at 640: 118 ms OpenVINO,
132 ms ONNX Runtime, on a 4-core Xeon); OpenVINO's default drops to bfloat16
where the CPU supports it, 49 ms there. Measure on your own board before
choosing for speed.

One `Detector` can be shared between threads — a capture thread and a worker,
say — each gets its own inference request underneath, so calls neither block
nor collide.

A webcam loop you can copy, with an FPS counter and optional tracking, lives in
[`examples/webcam.py`](examples/webcam.py):

```bash
python examples/webcam.py --track          # camera 0
python examples/webcam.py --source rtsp://camera/live --conf 0.4 --save
```

The log line reads the way you expect:

```
image 1/1 bus.jpg: 640x640 4 persons, 1 bus, 12.3ms
```

### Masks: `task="segment"`

```python
model = Detector("dfine-s", task="segment")
r = model("photo.jpg")[0]
r.masks.data        # (N, H, W) bool, one per box, in the same order
r.masks.xy          # each mask's outline, (K, 2) pixel points
r.masks.area        # pixels inside each
r.save("result.jpg")   # boxes with their masks tinted in
```

The detector finds the boxes; MobileSAM (Apache-2.0) then outlines what is in
each one, prompted by the box. It knows nothing of classes, so a model trained
on your own boxes gets masks with no mask labels at all. It downloads once
(44 MB) and costs about 150 ms a picture plus 25 ms a box on a 4-core CPU —
fine for photos and recordings; on a live CPU webcam, segment every few
frames. `predict`, `track` and the CLI (`task=segment`) all take it.

### Keypoints: `task="pose"`

```python
model = Detector("dfine-s", task="pose")
r = model("people.jpg")[0]
r.keypoints.xy      # (N, 17, 2) COCO body keypoints, one row per box
r.keypoints.conf    # (N, 17) 0-1; zero for boxes that are not people
```

Each person box is cut out and easydetect's own keypoint network (trained on
COCO keypoints) places the 17 keypoints; about 13 ms a person on a 4-core
CPU, reading each person mirrored too (`model.pose_flip = False`: about 9 ms,
2 AP lower). How it works, how it was trained and how to train it again:
[pose.md](pose.md).

### Results

| Attribute | What you get |
| --- | --- |
| `r.boxes.xyxy` | `(N, 4)` pixel corners |
| `r.boxes.xywh` | `(N, 4)` pixel centre + size |
| `r.boxes.xyxyn` / `r.boxes.xywhn` | the same, normalized 0..1 |
| `r.boxes.conf` / `r.boxes.cls` | `(N,)` scores and class indices |
| `r.boxes.id` | track ids after `model.track(...)`, else `None` |
| `r.masks` | with `task="segment"`: `.data` `(N, H, W)` bool, `.xy` outlines, `.area`; else `None` |
| `r.keypoints` | with `task="pose"`: `.data` `(N, 17, 3)`, `.xy`, `.conf`; else `None` |
| `r.names` | `{0: "person", ...}` |
| `r.plot()` | annotated BGR ndarray |
| `r.save()` / `r.show()` | write / display it |
| `r.summary()` | detections as JSON-ready dicts |
| `r.speed` | `{"preprocess": ms, "inference": ms, "postprocess": ms}` |
