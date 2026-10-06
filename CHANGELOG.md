# Changelog

## Unreleased

- **Masks of your own.** A segmentation dataset (polygons, Ultralytics'
  format) trains the detector on the polygons' boxes, then fine-tunes
  MobileSAM's mask decoder on the polygons with jittered box prompts (SAM's
  focal + dice loss on the best of three masks, its IoU head alongside); the
  image encoder stays frozen and each picture's embedding is cached. Scored by
  val mean IoU from MobileSAM as it was (epoch 0); the best lands as
  `mask_decoder.onnx` beside `best.pt`, which `task="segment"` uses
  (`Detector.seg_model` to choose). `seg_epochs`, `seg_batch`, `seg_lr`,
  `seg=False`. SAM's prompt encoder and mask decoder are vendored as
  `easydetect.nn.sam_decoder` (Apache-2.0, Meta), matching segment-anything's
  outputs exactly on MobileSAM's weights; those come from the mirror's
  `mobile_sam/decoder.pt`, or MobileSAM's pinned, checksummed release until
  that is there.
- **Keypoints of your own.** A data.yaml with `kpt_shape` (Ultralytics' pose
  format, any number of keypoints, `flip_idx`, optional `kpt_names` and
  `skeleton`) trains the detector and then a keypoint network for its boxes,
  from the detector's own backbone; it lands as `pose.onnx` beside `best.pt`,
  carrying its keypoint set, and `task="pose"` uses it there. `summary()` names
  and `plot()` draws that set. `pose_epochs`, `pose_batch`, `pose_lr`,
  `pose_size`, `pose=False`. The training loop moved from tools/train_pose.py
  into `easydetect.pose_trainer`.
- Training's keypoint and mask stages report through `on_progress` like the
  detector's epochs (`"phase": "keypoints"` / `"masks"`, with `"ap"` /
  `"miou"` at a scored epoch's end, and `"stage": "encode"` while MobileSAM
  reads the pictures), and `export()` copies `pose.onnx` / `mask_decoder.onnx`
  next to the exported model, so the IR or ONNX keeps its keypoints and masks.
- Fixed: a keypoint label line with an even number of keypoints (4 × 3 + 5 =
  17 values) was read as a polygon; with `kpt_shape` in the data.yaml it is
  read as a box and keypoints.
- `Detector.pose_model`: `"s"` (default), `"m"` (HGNetv2-B2, 8.1 M parameters),
  `"l"` (HGNetv2-B4, 15.8 M, from dfine-l, whose backbone has no lab layers) or
  the path of a pose `.onnx` of your own; the CLI and `easydetect serve` take
  `pose_model=` and `pose_flip=`. Each is fetched as `pose/pose-<size>.onnx`;
  `train_pose.py --size l` trains the new one, and `mirror.yml` uploads any
  size (`pose_size`).

## 0.5.0

- **`task="pose"` works: easydetect's keypoint model is on the mirror**
  (`pose/pose-s.onnx`). OKS AP on COCO val2017: 0.637 with the labelled boxes,
  0.588 behind dfine-s; about 13 ms a person on a 4-core CPU. Trained on
  GitHub's CPU runners: 40 epochs with the early backbone kept from the
  detector, then 20 more with every layer training. docs/pose.md has the table.
- `mirror.yml` takes `pose_run=<train-pose run id>`: checks that model on both
  runtimes, uploads `pose/pose-s.onnx` alone, downloads it back.
- `task="pose"` reads each person mirrored too and averages the two: +2.5
  OKS AP on COCO val2017 with the labelled boxes (0.589 → 0.613), +2.0 behind
  dfine-s (0.545 → 0.565), for about 13 ms a person instead of 9 on a 4-core
  CPU (the two crops share one call). On by default; `model.pose_flip = False`
  for the faster reading. Keypoints also decode between the network's
  half-pixel bins (a parabola through each peak): measured, under 0.001 AP.
  `train_pose.py --eval` scores either way and with the labelled boxes
  (`--detector gt`); `eval-pose.yml` runs those on an earlier run's model.
- `train_pose.py --init` takes a keypoint `best.pt` to train further, and
  `train-pose.yml` starts from an earlier run's with `init_run`.
- Fixed: scoring the whole keypoint pipeline (`train_pose.py --eval`) failed
  on pictures with nobody in them — about half of COCO val2017. In
  `train-pose.yml` the error was hidden by a pipe (now `pipefail`), a finished
  but unscored run is scored when resumed (`resume_run`), and a later leg that
  finds nothing to continue stops instead of starting over.
- Keypoint training (`tools/train_pose.py`): each scoring rates the EMA and the
  network itself and keeps the better as `best.pt` (`ap_ema`, `ap_net`,
  `weights` in `results.csv`; an older file gains the columns on resume);
  gradients are capped at norm 3 (`--clip`, was a fixed 10 that never bit);
  half-body crops smaller than 32 px are skipped (they were blown up to 150×);
  `train-pose.yml` logs each runner's CPU. From an audit of the first CPU run,
  whose EMA scored 0.32 after a loss spike while the network was fine.
- docs/training.md: the measured recipe comparison — on a 40-epoch VOC
  fine-tune, multi-scale, mosaic and mixup did not beat the default (mixup a
  tie, all three together −0.05 mAP50-95), so they stay off.

## 0.4.1

- **Keypoints: `Detector(..., task="pose")` — code only, weights in 0.5.0.**
  The keypoint model is still training; until its weights are on the mirror,
  `task="pose"` stops with an error that says so. Every person box gets COCO's 17
  body keypoints (`r.keypoints`: `.xy`, `.conf`), drawn as a skeleton by
  `r.plot()` and listed by name in `r.summary()`. The detector finds the
  people; easydetect's own top-down network (HGNetv2-B0 from the detector's
  COCO backbone, a SimCC-style head, 3.7 M parameters) places the keypoints
  in each crop — about 7 ms a person on a 4-core CPU. Trained on COCO
  keypoints only by `tools/train_pose.py`, which also scores the whole
  pipeline (`--eval`); docs/pose.md.
- `KeypointAP`, COCO's OKS AP, for scoring keypoints without pycocotools.
- `.github/workflows/train-pose.yml` trains the keypoint model on GitHub's CPU
  runners, as a chain of five-hour jobs that hand the run on.
- **`easydetect serve`**: an HTTP server around a model — `POST /predict`
  with a picture (raw or a form's `image`) answers with `r.summary()` as JSON,
  or the drawn picture with `draw=1`; `GET /health`. Standard library only, so
  it runs wherever the model does; 127.0.0.1 unless `host=0.0.0.0`.
  docs/deploy.md: the server, the `.onnx` contract (with a NumPy decoder that
  matches easydetect's to 1e-5 px), and notes on TensorRT, RKNN, TFLite.
- **Training options `multiscale=True`, `mosaic=0.5`, `mixup=0.3`** (off by
  default): a random size per batch within ±25% of `imgsz`, 2×2 mosaics of four
  pictures, and two pictures laid over each other. `recipe.yml` measures them
  on a fixed slice of Pascal VOC 2007 (`tools/voc2yolo.py`).
- **Fixed: training scripts without `if __name__ == "__main__":` (and scripts
  piped in) failed in 0.4.0** — "DataLoader worker exited unexpectedly". 0.4.0
  started every DataLoader worker from a forkserver, which first runs the main
  script again. Workers are forked again, as up to 0.3, unless OpenVINO is
  loaded in the process (its threads are what made forking unsafe: the lab,
  which predicts and trains in one process).

## 0.4.0

- **Masks: `Detector(..., task="segment")`.** Every box gets a mask
  (`r.masks.data`, `.xy` outlines, `.area`), drawn by `r.plot()` and listed
  by `r.summary()`. The detector finds the boxes and MobileSAM (Apache-2.0)
  outlines what is inside each, so a model trained on boxes alone gets masks
  — no mask labels. Downloads once (44 MB); about 150 ms a picture plus 25 ms
  a box on a 4-core CPU. On drawn shapes the masks matched the true ones at
  IoU 0.99 on both runtimes. `tools/convert_sam.py` builds the files from a
  pinned commit and hash-checked weights; the mirror workflow builds, checks
  and uploads them.
- **Fixed: training after predicting in one process could crash.** DataLoader
  workers were forked, and a fork of a process whose OpenVINO thread pool is
  alive can inherit locks held by threads that no longer exist — easydetect
  lab, which serves predictions and trains in one server, was exposed. Workers
  now start from a forkserver (spawn on Windows, as before).

## 0.3.1

- **Lighter exports.** `export(queries=100)` starts the decoder from fewer
  candidate boxes and `export(layers=1)` stops it after fewer layers (each is
  trained to answer on its own); dfine-n at 320 on a 4-core CPU: 19.9 ms →
  13.2 ms. `export(format="openvino", int8=True, data="data.yaml")` writes an
  8-bit IR calibrated on 300 training pictures (NNCF, now in `[train]`):
  dfine-s at 640, 106 ms in float32 → 45 ms, on CPUs without bfloat16.
- `tools/eval_exports.py` scores such variants side by side — mAP, precision
  and recall at the default conf, speed — on COCO or your own val split.

## 0.3.0

- **Changed: `predict` and `track` default to `conf=0.5`** (was 0.25), and so
  does the CLI. D-FINE gives an unsure box 0.3–0.5 where YOLO gives it under
  0.25: on COCO val2017, dfine-m's boxes shown at 0.25 were 32% right, at 0.5
  70%, still finding 67% of the objects. `conf=0.25` brings back the old
  output. NMS stays at `iou=0.7` and `contain` stays off — both measured in
  docs/performance.md ("Confidence and overlap defaults").

## 0.2.3

- **`contain=0.8` merges the pieces of one object.** A half hidden object can
  come back as the whole of it plus its visible pieces (a chair behind a
  person: 0.64 whole, 0.67 and 0.61 pieces), and IoU-based NMS keeps them all
  because a piece overlaps the whole only by its share of the area. With
  `contain`, a box sharing at least that much of the smaller box's area with
  another of its class is one object: the inner box goes when the enclosing one
  is about as sure (within 0.1), the enclosing one when it is much less sure
  (a loose box around a group). Off by default — a child held by an adult is
  also a box inside a box — on `predict`, `track` and the CLI.

## 0.2.2

- **The learning rate follows the batch size.** `lr0=None`, now the default,
  means `1e-4 × √(batch / 4)`: unchanged at batch 4, 1.4e-4 at the default 8,
  2.8e-4 at 32. With one fixed rate a bigger batch took fewer steps and
  learned less in the same epochs (0.76 mAP50-95 at batch 4, under 0.5 at 32).
  **Changed:** a run at the default batch 8 now trains at 1.4e-4 instead of
  1e-4; `lr0=1e-4` keeps the old rate. `run.json` says which rate was used and
  why.
- **An `.onnx` works on its own.** Export writes the class names into the
  file (ONNX metadata), and loading reads them back — on OpenVINO and ONNX
  Runtime, without the `onnx` package — before any `labels.txt` in the folder.
  So one `best.onnx` downloaded from a Hugging Face page names its classes.

## 0.2.1

- **The browser app moved** to its own repository,
  [easydetect lab](https://github.com/themakerrobot/easydetect-lab). It runs on
  this package from PyPI; `platform/` is gone from here. A data folder from
  `platform/` keeps working: `python run.py --data <easydetect-platform folder>`.
- `easydetect.data.dataset.list_images` (was `_list_images`): the images of one
  split of a `data.yaml`, public because the lab uses it.

## 0.2.0

- **Two runtimes.** `pip install easydetect` now brings ONNX Runtime beside
  OpenVINO. `Detector(..., backend="onnxruntime")` runs the `.onnx` on any CPU
  (a Raspberry Pi included) with the same preprocessing and decoding, so the
  boxes match OpenVINO's; OpenVINO stays the default and the way to an Intel
  GPU or NPU. `$EASYDETECT_BACKEND` sets the default. A named model downloads
  the `.onnx` for ONNX Runtime (the mirror now carries one per size), a `.pt`
  exports whichever the runtime reads, and an `.xml` without OpenVINO runs
  from the `.onnx` beside it. `pip install "easydetect[train]"` adds training.
- **Validation on large sets.** `val` (and the per-epoch validation) no longer
  runs out of file descriptors on COCO-sized validation sets ("received 0
  items of ancdata").
- **Faster augmentation.** The zoom, crop and flip are rendered once at the
  training size: 3.1 ms a picture at 320, down from 25.4.
- **Tools.** `tools/coco2yolo.py` converts COCO 2017 for fine-tuning the COCO
  models at another input size; `tools/compare_yolo.py` scores a YOLO ONNX
  export and D-FINE checkpoints with one evaluator.

## 0.1.2

- `predict` and `track` drop a box that covers a higher-scoring one by more
  than `iou=0.7`, whatever the class — D-FINE's occasional second box on one
  object (a vehicle as both truck and car). `iou=None` keeps every box.
- **Changed:** `track(iou=)` is now that duplicate filter, as in Ultralytics;
  the tracker's matching threshold is `match_iou=0.3`.

## 0.1.1

- **Augmentation** from D-FINE's recipe — colour jitter, zoom-out, IoU crop,
  flip — with the last tenth of the epochs on plain pictures;
  `train(augment=False)` keeps only the flip.
- Platform: early stopping per run, the backbone-freeze choice in plain sight,
  a training run that can always be deleted, cut-off runs that keep their best
  epoch, and the themaker-ui design with a Korean / English switch.

## 0.1.0

- First release: D-FINE (n/s/m/l/x) with its Apache-2.0 COCO weights, training,
  OpenVINO export and inference, and the browser platform.
