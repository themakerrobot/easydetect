# Changelog

## 0.7.7

- `resume=True` carries on with the run's own frozen parts, queries and
  denoising when the call leaves them out (from `run.json`); before, a run
  resumed from a plain `Detector("dfine-n")` failed to load its optimizer or
  went on with 300 queries. It carries on `project/name` itself when that has
  a `last.pt` — never another run whose name starts the same (the lab's
  `job5` resumed `job57` when that one was newer) — and otherwise the newest
  of `name2`, `name3`, … Resumed with more epochs than planned, an
  early-stopped run gets its full patience again instead of stopping after
  one epoch (for a run saved before 0.7.7, the plan is read from `run.json`).
- The restart from the best epoch at the start of the plain-picture epochs
  resets the patience count: a run whose best came early could stop one
  epoch into them.
- Fine-tuning a checkpoint on the same classes in another order (`[cat, dog]`
  on `[dog, cat]`) gives each class its own row; before, the heads came
  across as they were and the classes swapped. The COCO start does the same
  for a dataset with COCO's 80 classes in another order. A new class with the same
  class count no longer inherits another's row. Re-heading keeps the
  checkpoint's queries and denoising. `run.json` records which classes
  started from a known row (`start.class_rows`); `train(match_names=False)`
  turns matching off. Names that mean other things as often are no longer
  matched: ship, plant, table, 가위, 연.
- A data.yaml without `val:` trains without validation and says so, instead
  of failing. A misspelt `clean_bn` is an error. `name=3` works.
- Label files with a byte-order mark (Windows Notepad) read correctly; a line
  that is not numbers is left out and the file named, instead of stopping the
  run. Image lists (`.txt`) with a byte-order mark too.
- `workers=None` (the default) is 4, or 0 on a Windows CPU, where a worker
  re-runs an unguarded script. Validation during training uses no more
  workers than training, and none for a small val set. A last batch is
  dropped only when it would hold one picture.
- A saved video keeps a Korean file name on Windows (written under a plain
  name, renamed at the end); two videos of the same name from different
  folders no longer overwrite each other; a writer that cannot open is said.
- `easydetect benchmark` on a folder leaves out the keypoint and mask models
  beside a detector, and a model that loads but fails to run is a row with
  its error instead of stopping the table.
- The mask decoder's best epoch is read back weights-only, like every other
  checkpoint.

## 0.7.6

- The plain-picture epochs keep the batch-norm statistics only when the whole
  network trains (`clean_bn="auto"`, the default). Three seeds on three small
  sets: training everything, keeping them was even or ahead; with the backbone
  frozen (`freeze="backbone"`, the lab's classroom preset), letting them adapt
  scored higher on all three (signature 0.885 → 0.907, medical-pills
  0.554 → 0.659, 100 african-wildlife pictures 0.628 → 0.660), so frozen runs
  now adapt them as before 0.7.5. `Trainer(clean_bn="keep" | "adapt")` forces
  either.
- Measured with the same runs: a small dataset wants more epochs. dfine-n at
  320 px, backbone frozen, 100 queries: 30 → 70 → 100 epochs gave 0.628 →
  0.730 → 0.751 on 100 african-wildlife pictures (YOLO26n 0.679), 0.885 →
  0.938 → 0.960 on signature, 0.554 → 0.693 → 0.746 on medical-pills.

## 0.7.5

- **Classes the COCO weights already know keep what they know.** Fine-tuning
  on a dataset with other classes gave every class a fresh head; now a class
  named like one of COCO's 80 (`apple`, `Bottles`, `mobile_phone`, and the
  Korean names, `사과`, `컵`, `사람` …) starts from that class's rows. On 100
  african-wildlife pictures (elephant and zebra are COCO classes) dfine-n at
  320 px scored 0.25 mAP50-95 after the first epoch instead of 0.02, and 0.40
  instead of 0.12 after six. `easydetect.class_names.match_classes`,
  `easydetect.model.copy_class_rows`.
- **Fixed: the last tenth of training (plain pictures) lost accuracy.** The
  batch-norm statistics were re-estimated from plain-picture batches and the
  score fell, every run (100 pictures: 0.48 → 0.40 mAP50-95 in three epochs).
  Those epochs now keep the statistics the augmented ones gathered (0.48 held).
  `best.pt` was never the worse model; the drop cost the chance to improve.
- `easydetect benchmark`: a model that fails says why (the cause, the last line
  of OpenVINO's message, not its first).
- `tools/long_run.py`: a week of unattended runs on one GPU machine — COCO
  fine-tuned at 320/416/640 px, every benchmark again — resuming after a crash
  or a reboot; the same benchmarks again with other seeds, summed up as
  mean ± spread. `tools/bench_finetune.py --no-augment --seed --clean-bn`
  (`Trainer(clean_bn="adapt")` re-estimates the statistics as before, to
  compare).

## 0.7.4

- **Fixed: pictures in a folder with Hangul in its path could not be read on
  Windows** (a desktop called `바탕 화면`, a Korean OneDrive): OpenCV's
  `imread` takes the path in the system code page and returned nothing, and
  training stopped with "unreadable pictures". Pictures are now read and
  written through Python (`easydetect.utils.imageio.imread` / `imwrite`), and
  videos open through the folder's short path on Windows (`cv2_path`).

## 0.7.3

- `train(queries=, denoising=)`: fewer decoder queries (of 300) and denoising
  queries (100) while training. On a CPU most of a step goes to them, not to
  the picture (dfine-n at 256, 320 or 416 px: about the same time a step);
  `queries=100, denoising=30` with `freeze="backbone"` trains about 2× faster
  on a laptop. The checkpoint keeps the query count, so the model also runs
  with it.
- `tools/profile_train.py --freeze --queries --denoising`, and the same options
  in `tools/bench_finetune.py`.

## 0.7.2

- **Fixed: downloads from the mirror could fail since 0.7.0.** The SHA-256
  check took the storage server's `ETag` (another hash, after Hugging Face's
  redirect) for the file's SHA-256 and threw a good file away. Only
  `X-Linked-Etag`, which Hugging Face sends with the redirect, is trusted now.
- `easydetect benchmark` takes model names (`dfine-n`) and says why one could
  not be fetched.

- `easydetect benchmark model=<files or folders> threads=4`: times every model
  on every runtime that can read it (OpenVINO, ONNX Runtime): preprocessing,
  inference, decoding and fps, fastest first. Needs no PyTorch, so it runs on
  the device itself (a Raspberry Pi).
- `tools/pi_kit.py` (on the PC): exports models × input sizes × lighter
  decoders, as OpenVINO IR, 8-bit IR and ONNX, and scores each on a val split,
  so a Pi's timings and the accuracy make one table.
- `OVPredictor(threads=)` caps OpenVINO's CPU threads.

## 0.7.1

- **Faster training steps on a GPU.** Every layer's matching now crosses to
  the CPU in one copy instead of one per layer; the matched indices go back to
  the GPU in one pinned copy instead of a blocking copy at each of the loss's
  dozens of indexing steps; two box counts no longer round-trip through the
  GPU; AdamW is fused; cuDNN picks its fastest convolutions (`benchmark`) and
  float32 matmuls use TF32; the encoder's position embedding and the
  decoder's anchors are built once per size (were rebuilt on the CPU every
  step); the network runs channels-last on a GPU; the loss no longer waits
  for the GPU at boolean selections, box checks and a value comparison (78
  waits a step before, 2 now). The losses are unchanged (tested equal).
  dfine-s at batch 16 on an RTX 5090: 172 -> 97 ms a step (93 -> 166
  pictures a second), 1.8× faster.
- `tools/profile_train.py`: where a training step's time goes (data, forward,
  loss, backward, update), `--trace` for PyTorch's operator table.

- `tools/bench_finetune.py`: each model is scored both as its own project
  scores it (a box counts for every class above conf) and as `predict` hands
  boxes out (one class a box); `--rescore` scores finished runs again; the
  low-data set is 100 pictures drawn at random (the first 100 of
  african-wildlife are all buffalo); D-FINE trains in its own process.
- Note on 0.7.0's COCO-style AP: before it, a class the model never predicted
  scored 0.5 instead of 0 (the precision curve was drawn down to recall 1),
  and every class whose recall stopped short of 1 scored a little high. mAP
  printed by 0.6 and earlier, during training and by `val`, reads higher than
  the same model scores now; compare runs with one version.

## 0.7.0

Tracking, masks and video

- **A new tracker** (ByteTrack's recipe): a Kalman filter predicts where each
  object moves, boxes are assigned globally, and boxes below `conf` (down to
  0.1) carry a track through a partial occlusion without being shown. A fast
  object (60 px a frame) keeps its id, crossing objects no longer swap ids
  (0 of 12 cases, was 9), and a one-frame flicker no longer starts a track.
  Matching ignores the class (`match_class=True` to match by class);
  `match_iou` defaults to 0.2.
- **Masks 3× faster, 40–55% less memory**: the mask decoder's small output is
  upscaled per box, only inside the box (4K, 30 people: 8.3 → 3.0 s on
  OpenVINO). Each mask is cut to its box grown by 10% (`box_pad`); a
  segmenter's `fill_holes` and `largest` clean masks further.
  `r.summary()` with masks 30× faster, drawing masks 6× faster.
- Fixed: a fine-tuned mask decoder exported with a fixed crop gave wrong
  full-size masks for pictures that are not 4:3. Mask fine-tuning now trains
  all four mask outputs, picks the better selection rule (best IoU or the
  single mask) on jittered boxes, and builds it into the exported decoder.
- Video: a webcam or stream is read on its own thread and only the newest
  frame is kept (no growing lag); files are read ahead; `vid_stride` skips
  frames without decoding them (stride 4: 27.7 → 16.0 ms a frame). Saved
  videos are H.264 when OpenCV has it (plays in browsers), else mp4v.

Accuracy

- **Fine-tuning computes its loss in float32.** Under float16 autocast a sure
  logit's focal cost became -inf and the matcher stopped weighing the boxes;
  the network still runs in mixed precision (bfloat16 where the GPU has it,
  so no loss scaling). Custom models train better, most with AMP on.
- **The plain-picture epochs start from the best weights so far** (as D-FINE
  does), the average then following a little faster.
- Multi-scale sizes drawn as D-FINE draws them (imgsz three times in 13), a
  mosaic tile is no longer zoomed out on top of being a quarter of the
  picture, and every shrink in augmentation is antialiased.
- Warmup is at least min(500 steps, 5% of the run): a small dataset no longer
  warms up in a handful of steps.
- Validation scores like COCO: the 300 best (query, class) pairs, up to 100
  boxes a picture, 101-point AP. Val boxes are clipped into the picture.
- `classes=[...]` keeps a query sure of a wanted class even when another class
  scores higher on it.
- Keypoint AP agrees with pycocotools: our pipeline is ahead of YOLO26-pose at
  every size (s 66.3 / 63.0, m 71.6 / 68.8, l 73.4 / 70.4). README and
  docs/pose.md updated.

Custom data

- Training prints (and run.json keeps) plain warnings: fewer than 100
  pictures, a class with no or very few boxes, very uneven classes, val the
  same pictures as train, no val split.
- A label with a class id outside the dataset's names is dropped with a
  warning instead of crashing a batch; an unreadable picture is skipped.
- Resume carries on exactly: random state and the early-stop count are saved.

Speed

- Pictures leave the loader as uint8 and become float on the GPU (53 → 147
  pictures a second fed on a 4-core CPU); the average update is one fused
  call (230 → 90 ms a step on a CPU); the loss is no longer synchronised
  every step.
- Preprocessing in one pass (1080p: 6.9 → 1.7 ms). OpenVINO caches compiled
  models (`$EASYDETECT_OV_CACHE=0` to turn off): the second load is about 3×
  faster. ONNX Runtime picks CUDA or DirectML by itself on `device="AUTO"`.
- An exported model is reused instead of exported again, keyed by the weights'
  path, size and time, so two `best.pt` never share a cache entry.

Safety

- Checkpoints load with `weights_only=True`: a `.pt` that would run code is
  refused unless `EASYDETECT_TRUST_CHECKPOINTS=1`.
- Downloads are checked: a cut-off transfer or one whose SHA-256 does not
  match the server's is an error, not a broken model file.
- ONNX `half=True` now writes float16 (needs `onnxconverter-common`); INT8
  export reports how many FP32 boxes it still finds on the calibration
  pictures and warns below 90%.
- `easydetect serve`: one inference at a time, at most 8 waiting (503 after),
  a 30 s read timeout, and a 500 with the reason when the model fails.

Benchmark

- `tools/bench_finetune.py` fine-tunes dfine-s and a YOLO on the same small
  public datasets and scores both with one evaluator. `compare_yolo.py`
  reads an end-to-end YOLO's ONNX (YOLO26).
- **Keypoint models s, m and l on the mirror**, each trained for 100 epochs
  with every layer learning on an RTX 5090. `eval-pose.yml` and `mirror.yml`
  take a model from a URL (`pose_url`, a release asset, drafts included).

## 0.6.0

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
  that is there. Measured on COCO val2017's polygons (1,200 pictures to
  train, 300 to score, every class as one): val mean IoU 0.752 as MobileSAM
  was, 0.810 after 6 epochs; on a 4-core CPU the pictures encode once in
  about 20 minutes and an epoch takes about 29.
- **Keypoints of your own.** A data.yaml with `kpt_shape` (Ultralytics' pose
  format, any number of keypoints, `flip_idx`, optional `kpt_names` and
  `skeleton`) trains the detector and then a keypoint network for its boxes,
  from the detector's own backbone; it lands as `pose.onnx` beside `best.pt`,
  carrying its keypoint set, and `task="pose"` uses it there. `summary()` names
  and `plot()` draws that set. `pose_epochs`, `pose_batch`, `pose_lr`,
  `pose_size`, `pose=False`. The training loop moved from tools/train_pose.py
  into `easydetect.pose_trainer`. Measured on COCO's five face keypoints as
  a dataset of their own (3,355 heads to train, 810 to score): OKS AP 0.946
  after 10 epochs of 95 s on a 4-core CPU — with Ultralytics' default
  per-keypoint sigma for a keypoint set of your own (1/K), more lenient than
  COCO's, so not comparable with COCO's body AP.
- Fixed: a mask-training warning about converting a tensor that needs grad.
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
