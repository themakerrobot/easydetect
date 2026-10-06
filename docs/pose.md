# Keypoints: `task="pose"`

```python
from easydetect import Detector

model = Detector("dfine-s", task="pose")
r = model("people.jpg")[0]
r.keypoints.xy      # (N, 17, 2) pixel positions, one row per box
r.keypoints.conf    # (N, 17) 0-1; all zero for boxes that are not people
r.save("result.jpg")   # boxes with each person's skeleton drawn in
```

The 17 keypoints are COCO's, in COCO's order
(`easydetect.pose.KEYPOINT_NAMES`: nose, eyes, ears, shoulders, elbows,
wrists, hips, knees, ankles). `r.summary()` lists them by name for every
person. `predict`, `track` and the CLI (`task=pose`) all take it.

## Accuracy

OKS AP on COCO val2017 (5,000 pictures), through the exported model the
package downloads, with easydetect's scorer (written to COCO's definition;
it can differ from pycocotools in the third decimal):

| | AP | AP50 | AP75 |
| --- | --- | --- | --- |
| the labelled person boxes (the keypoint model alone) | 0.637 | 0.890 | 0.708 |
| behind dfine-s (what you get) | 0.588 | 0.801 | 0.655 |
| behind dfine-s, `model.pose_flip = False` | 0.571 | 0.795 | 0.637 |

For scale, Ultralytics publishes 50.4 for YOLOv8n-pose and 60.0 for
YOLOv8s-pose (their own end-to-end models, scored with pycocotools). Behind
dfine-m instead of dfine-s the score is the same (0.571 either way, without
mirroring), so the faster detector is enough.

The released model was trained on GitHub's CPU runners in two runs: 40 epochs
with the stem and first two backbone stages kept from the detector (0.589
with the labelled boxes), then 20 more epochs from there with every layer
training, the learning rate starting again at 4e-4 (0.619). Both runs, their
logs and their scores are in the repository's Actions history
(`train-pose.yml`, `eval-pose.yml`).

## How it works

Top-down, in two steps. The detector finds the people; each person box is
widened to a 3:4 shape with a quarter of margin (a raised arm often sticks
out of the box), cut out at 256×192, and a small network places the 17
keypoints in the crop, which are mapped back to the picture. A box gets
keypoints when its class is named `person` — or every box does, when the
model has no such class (a model trained on "worker" or "player").

The keypoint network is easydetect's own (`easydetect/nn/posenet.py`, 3.7 M
parameters): D-FINE's HGNetv2-B0 backbone, started from the COCO detector's
weights so it already knows people, and a head that reads each keypoint's
position as two classifications — over the crop's columns and over its rows,
at half-pixel steps (the SimCC formulation, Li et al., ECCV 2022) — with one
transformer layer across the 17 keypoints so an elbow is placed knowing where
the shoulder and wrist are. A keypoint's confidence is how sharply peaked
those two distributions are.

Each person is read twice, as cropped and mirrored, and the two readings are
averaged (left and right swapped back): on COCO val2017 that is worth 2.5 OKS
AP with the labelled boxes (0.589 → 0.613) and 2.0 behind dfine-s (0.545 →
0.565). On a 4-core CPU (OpenVINO) it costs about 13 ms for one person instead
of 9, and 40 ms for four instead of 20, on top of the detector. For the faster
reading:

```python
model = Detector("dfine-s", task="pose")
model.pose_flip = False   # before the first prediction
```

Keypoints are placed between the network's half-pixel steps (a parabola
through each peak); measured, that is worth less than 0.001 AP, but costs
nothing.

### A larger keypoint model, or your own

```python
model = Detector("dfine-s", task="pose")
model.pose_model = "m"                          # HGNetv2-B2 instead of B0
model.pose_model = "l"                          # HGNetv2-B4
model.pose_model = "runs/pose/m/pose-m.onnx"    # or one you trained
```

| | backbone (from) | parameters | time a person, relative |
| --- | --- | --- | --- |
| `"s"` | HGNetv2-B0 (dfine-s) | 3.7 M | 1× |
| `"m"` | HGNetv2-B2 (dfine-m) | 8.1 M | about 2× |
| `"l"` | HGNetv2-B4 (dfine-l) | 15.8 M | about 4× |

Each is downloaded as `pose/pose-<size>.onnx` once that is on the mirror
(until then it says so). On the command line and in `easydetect serve`:
`pose_model=m`, `pose_flip=false`.

It is trained on COCO 2017's person keypoints (labels CC BY 4.0) and nothing
else, so the weights carry no research-only licence.

## Training it

210 epochs over COCO's 150,000 labelled people. On one RTX 5090 that should
take roughly 4–5 hours — an estimate: reading, cropping and jittering a person
costs about 4 ms of one CPU core, so 16 workers feed some 4,000 a second and
the loading, not the GPU, sets the pace. Give it as many `--workers` as you
have cores to spare.

```bash
git clone https://github.com/themakerrobot/easydetect && cd easydetect
pip install "easydetect[train]"

# COCO 2017: pictures and annotations (skip what you already have)
mkdir -p ~/datasets/coco/images && cd ~/datasets/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip -d images && unzip -q val2017.zip -d images
unzip -q annotations_trainval2017.zip            # annotations/person_keypoints_*.json
cd -

python tools/train_pose.py --coco ~/datasets/coco --workers 16
```

`tools/coco2yolo.py` reads the same folder, so a COCO copy made for it works
as it is. The run writes to `runs/pose/s/`: `results.csv` (loss each epoch,
OKS AP every `--val-every` epochs), `run.json` (the settings), `last.pt`
(everything needed to go on: `--resume runs/pose/s`) and `best.pt` (the
weights with the best AP). At the end `best.pt` is exported to
`pose-s.onnx` and checked against PyTorch on ONNX Runtime.

Each scoring rates both the EMA of the weights and the network itself
(`ap_ema`, `ap_net` in `results.csv`) and keeps the better. They usually agree
within a point, but the first CPU run had a short loss spike at epoch 14, and
for a few epochs after it the EMA, still averaging weights from both sides of
the spike, scored 0.518 where the network scored 0.534. Gradients are capped
at norm 3 (`--clip`): ordinary steps measured 1.4–3.4 on COCO at batch 64, so
the cap only trims spikes.

The AP printed during training is scored with COCO's own person boxes, which
measures the keypoint network alone. What a user gets depends on the detector
too; score the whole pipeline — the detector's boxes, missed people and false
boxes included — with

```bash
python tools/train_pose.py --coco ~/datasets/coco --eval runs/pose/s/pose-s.onnx --detector dfine-m
```

### Without a GPU

`.github/workflows/train-pose.yml` trains on GitHub's 4-core CPU runners
(free for a public repository). A job may run six hours and an epoch there
should take about an hour, so the run is a chain of up to 14 jobs, one after the
other: each downloads COCO, continues the previous job's `last.pt` (the
`pose-run` artifact), trains until the next epoch would not fit in 4.8 hours
(`--hours`), and hands the run on. The finishing job exports the ONNX file
and scores the pipeline behind dfine-s and dfine-m (`eval.txt`). Its default
is a shorter recipe for the CPU: 40 epochs at batch 64, and the stem and first
two backbone stages kept as the detector trained them (`--freeze 2`: their
backward pass is skipped, which makes a CPU step about 1.5× as fast; with
`channels_last`, 2× in all). Start it from the Actions tab (or `gh workflow
run train-pose.yml`); a run cut short continues from its artifact with
`resume_run=<run id>`. Nothing is uploaded to the mirror by it.

`--size m` and `--size l` train the larger ones on HGNetv2-B2 and B4, from
dfine-m's and dfine-l's backbones, written as `runs/pose/<size>/pose-<size>.onnx`.
On a CPU a training step of m costs about 2×, and of l about 4×, one of s.
`--limit 2000 --epochs 3` is a quick check that everything runs.

## Publishing it

`Detector(..., task="pose")` downloads `pose/pose-s.onnx` from the mirror.
A model trained by `train-pose.yml` goes there with `mirror.yml` and
`pose_run=<that run id>` (and `pose_size=m` or `l` for the larger ones): it checks the
model puts keypoints on the people in a test picture on both runtimes, uploads
`pose/pose-<size>.onnx` alone, and downloads it back the way users will. From
a machine that trained one:

```bash
pip install -U huggingface_hub
hf auth login                       # a write token for the mirror's account
hf upload leeyunjai/easydetect runs/pose/s/pose-s.onnx pose/pose-s.onnx
hf upload leeyunjai/easydetect runs/pose/m/pose-m.onnx pose/pose-m.onnx
hf upload leeyunjai/easydetect runs/pose/l/pose-l.onnx pose/pose-l.onnx
```

A model already downloaded stays in the cache (`~/.easydetect/pose/`): delete
the file there to fetch a newer upload.

Until it is there, `task="pose"` says so and where to put the file instead
(`~/.easydetect/pose/pose-s.onnx`).
