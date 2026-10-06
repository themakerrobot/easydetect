# Training, validating, exporting

Back to the [README](../README.md).

## Training

Labels are one `.txt` per image next to a `data.yaml`:

```yaml
path: /data/cans
train: images/train
val: images/val
names:
  0: can
  1: bottle
```

```python
model = Detector("dfine-s")
best = model.train(
    data="data.yaml", epochs=100, imgsz=640, batch=8,
    device=0, workers=4, project="runs", name="train",
    resume=False, patience=50, lr0=None, seed=0,
)
# runs/train/weights/best.pt  (last.pt too — resume=True picks the run back up)
```

D-FINE's recipe, one process: AdamW with the backbone at its own slower rate
(half for n/s, a tenth for m, less for l/x), no weight decay on norms and biases,
linear warmup into cosine decay, AMP on CUDA, and an exponential moving average
of the weights that is what gets validated and saved. The loss is D-FINE's —
varifocal classification, L1 + GIoU boxes, fine-grained localisation and
decoupled distillation, over every decoder layer and the denoising queries.
mAP50-95 after every epoch, early stop on `patience`.

A data.yaml with `kpt_shape` (Ultralytics' pose format) is a keypoint
dataset: training learns the boxes, then a keypoint network for them —
[pose.md](pose.md#keypoints-of-your-own). Polygon labels (a segmentation
dataset) train the detector on their bounding boxes, then fine-tune the mask
decoder task="segment" uses on the polygons —
[usage.md](usage.md#masks-of-your-own).

## Batch size and learning rate

`lr0=None` (the default) sets the learning rate from the batch size:
`1e-4 × √(batch / 4)` — 1e-4 at 4, 1.4e-4 at 8, 2e-4 at 16, 2.8e-4 at 32.
A bigger batch takes fewer optimizer steps an epoch, and Adam moves a weight by
about the learning rate each step, so at one fixed rate a batch of 32 ends the
same epochs having learned roughly an eighth as much: a fine-tune that reached
0.76 mAP50-95 at batch 4 stayed under 0.5 at batch 32 with the rate unchanged.
The square root is the usual rule for Adam, and at 32 it lands beside the
2–2.5e-4 D-FINE's own configs use at their batch of 32. `run.json` records
the rate and where it came from; `lr0=2e-4` sets it outright.

Other optimizer settings pass through `train()` as they are:
`lr_backbone_mult` (the backbone's rate as a fraction of `lr0`; by default
D-FINE's per-size ratio), `weight_decay=1e-4`, `warmup_epochs=1`, plus
`seed` and `amp`.

`device=0` uses the first CUDA GPU, `device="cpu"` forces CPU, and leaving it
out picks a GPU when there is one. CPU training is slow but real — at 320 px,
dfine-s manages an epoch of 240 images in a few minutes on 4 cores — and
fine-tuning from the COCO weights is what makes a short run enough.

No dataset yet? `python tools/make_toyset.py toyset` writes 300 labelled images of
squares and circles — enough to see a run learn, and to try every step below.

## Datasets from elsewhere

Any YOLO-format dataset trains as it is — the `.txt` layout above is what
labelling tools export. Downloaded datasets vary in the details, and these all
load without editing:

| in the download | what happens |
| --- | --- |
| `images/train` + `labels/train` | the standard layout |
| `train/images` + `train/labels`, yaml says `train: ../train/images` | a split per folder, as Roboflow exports it (pick a *YOLOv8* / *YOLOv11* TXT format) |
| `path:` naming someone else's machine (`/content/datasets/…`) | falls back to the folder the yaml is in |
| `train: [images/day, images/night]` | several folders make one split |
| polygon rows (`cls x1 y1 x2 y2 x3 y3 …`) | each becomes its bounding box |
| a sixth value on a box row | read as a confidence and ignored |

A label sits where the last `images` folder in the image's path becomes
`labels`, or beside the image when there is none — one rule, used by training,
[easydetect lab](https://github.com/themakerrobot/easydetect-lab) and its exports alike. A dataset with no `val:` is refused with a
message rather than validated on its training images.

```bash
unzip shapes.v1i.yolov11.zip -d shapes
easydetect train model=dfine-s data=shapes/data.yaml epochs=50 device=0
```

In [easydetect lab](https://github.com/themakerrobot/easydetect-lab), the same zip goes in through
*Datasets → zip* without unpacking.

## Starting point

`Detector("dfine-s")` warm-starts from the mirror's COCO weights. For a domain
COCO says nothing about (thermal, medical, satellite), or for a clean baseline,
start from the ImageNet-pretrained HGNetv2 backbone instead:

```python
Detector("dfine-s", pretrained=False).train(data="data.yaml", epochs=200)
```

## Freezing

```python
model.train(data="data.yaml", epochs=50, freeze="backbone")
```

Roughly twice as fast per epoch (measured on CPU: 1.90 s/step → 1.09 s/step at
640, batch 2) and it overfits less on a small dataset, because the features it
starts from are already good. `freeze` also takes `"encoder"`,
`"backbone+encoder"`, or a list of module prefixes. Frozen batch norms are held
in eval mode so their running statistics stop drifting.

## Augmentation

Training pictures go through D-FINE's recipe, each step on its own coin flip:
colour jitter (brightness, contrast, saturation, hue), a zoom-out that puts the
picture on a canvas up to 4× larger, an IoU-constrained random crop, and a
horizontal flip. Zoom-out teaches small objects and crop teaches large, partly
visible ones, so the model does not learn only the framing your photos had. The
last tenth of the epochs trains on plain pictures (flip only) so it settles on
what it will actually see; the log says when that starts.

```python
model.train(data="data.yaml", epochs=50, augment=False)   # flip only
```

`run.json` lists what a run used. Validation pictures are never augmented.

Three more, off unless asked for, and off for the last tenth with the rest:

```python
model.train(data="data.yaml", multiscale=True, mosaic=0.5, mixup=0.3)
```

- `multiscale=True` trains each batch at a random size within ±25% of
  `imgsz`, in steps of 32 (as D-FINE's own batch collation does), so the model
  does not tie an object's size to one input size.
- `mosaic=0.5` makes half the pictures a 2×2 mosaic of four, meeting at a
  random point: more objects a step, many of them small, in unusual company.
- `mixup=0.3` lays a second picture over 30% of them, at 40–60% opacity, with
  both pictures' boxes: finding an object through clutter that is not part of
  it.

They stay off by default because, measured, they did not help a fine-tune.
`.github/workflows/recipe.yml` trained `dfine-n` from its COCO weights on the
same 800 Pascal VOC 2007 pictures (`tools/voc2yolo.py`, 20 classes), 40 epochs
at 320, two seeds each, scored on 1000 held-out pictures (best mAP50-95):

| recipe | seed 0 | seed 1 | mean | CPU hours |
| --- | --- | --- | --- | --- |
| default | 0.5507 | 0.5507 | **0.551** | 1.2 |
| `mixup=0.3` | 0.5421 | 0.5529 | 0.548 | 1.9 |
| `multiscale=True` | 0.5456 | 0.5380 | 0.542 | 1.7 |
| `mosaic=0.5` | 0.5252 | 0.5368 | 0.531 | 1.8 |
| all three | 0.5022 | 0.5007 | 0.501 | 1.8 |

Seeds differ by up to 0.012, so mixup is a tie; the rest cost accuracy, all
three together 0.05, and each makes an epoch about half again as slow. They
are built for long training from scratch (YOLO trains 300 epochs or more), where
the model has time to learn from harder pictures; on a short fine-tune that
time is not there. Try them for long runs on a large dataset of your own, and
compare against a run without.

## Early stopping

`patience=50` stops a run once mAP has not improved for that many epochs and
keeps the best epoch's weights; `patience=0` runs every epoch. In
[easydetect lab](https://github.com/themakerrobot/easydetect-lab) it is the "안 나아지면 멈추기" field, set
by each preset.

## Another input size for the COCO models

The COCO weights are trained at 640. To run them at another size — 320 for a
small CPU — fine-tune them there on COCO itself:

```bash
python tools/coco2yolo.py ~/datasets/coco          # images/, annotations/ from cocodataset.org
easydetect train model=dfine-n data=$HOME/datasets/coco/data.yaml imgsz=320 epochs=4 batch=32
```

The converter keeps COCO's class order, the one the pretrained head uses, and
stops if the annotation file says otherwise. What to expect is in
[performance](performance.md#small-inputs-320).

## Watching a run

Every epoch appends a row to `runs/train/results.csv` and, if you pass one, calls
`on_epoch_end` with the same numbers — which is all a dashboard needs:

```python
model.train(data="data.yaml", epochs=100, on_epoch_end=lambda row: print(row))
# {'epoch': 1, 'loss': 24.7, 'vfl': .., 'l1': .., 'giou': .., 'map50_95': 0.31,
#  'lr': 0.0001, 'seconds': 12.4, 'epochs': 100, 'save_dir': 'runs/train'}
```

`summary.json` in the same folder holds the final numbers.

## Exporting and deploying

```python
model = Detector("runs/train/weights/best.pt")
xml = model.export(format="openvino", half=True, imgsz=640)
```

Writes `best.xml` + `best.bin`, plus **`labels.txt`** (one class name per line)
next to the IR — that is how downstream runtimes discover class names.

The exported graph emits **probabilities, not logits**. Anything that decodes it
must not apply a sigmoid a second time; this package checks the score range and
skips it (`easydetect/predictor.py`).

## CLI

`key=value` arguments:

```bash
easydetect predict model=dfine-s source=bus.jpg conf=0.5
easydetect train   model=dfine-s data=data.yaml epochs=100
easydetect val     model=best.pt data=data.yaml
easydetect export  model=best.pt format=openvino half=true
easydetect track   model=best.pt source=clip.mp4
```

