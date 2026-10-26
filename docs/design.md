# Design notes and provenance

Back to the [README](../README.md).

## Design notes

* **Backbone** HGNetv2 (PP-HGNetV2) B0 / B0 / B2 / B4 / B5 for n / s / m / l / x.
* **Encoder** an AIFI transformer layer on the coarsest level, then an ELAN
  (RepNCSPELAN4) FPN/PAN fusion with SCDown downsampling.
* **Decoder** two-stage: dense heads pick the top-300 encoder tokens as queries,
  3/3/4/6/6 layers refine them with multi-scale deformable cross-attention.
  Boxes are not regressed directly: each edge is a probability distribution
  over 33 bins, refined layer by layer (fine-grained distribution refinement).
* **Loss** Hungarian matching; varifocal classification, L1 + GIoU boxes,
  fine-grained localisation over the bins, and distillation from the last layer
  to the earlier ones — over every decoder layer, the encoder's proposals and
  the denoising queries.
* **Training** the original parameter groups and an EMA of the weights.
* **Validation** COCO-style mAP50 / mAP50-95, 101-point interpolation, no
  pycocotools dependency.
* **Inference** OpenVINO, plain resize (no letterbox) to match training.

The network under `easydetect/nn/` and the loss in `easydetect/utils/loss.py`
are adapted from [D-FINE](https://github.com/Peterande/D-FINE) (Apache-2.0),
itself built on [RT-DETR](https://github.com/lyuwenyu/RT-DETR), precisely so
The original COCO weights load here unchanged; see [NOTICE](../NOTICE).
Two changes, both outside the weights: anchors and position embeddings follow
the input instead of being cached for 640 × 640, so any input size works; and
an input with fewer tokens than queries selects what there is. Everything around
it — packaging, API, trainer, validator, exporter, predictor, CLI — is this
project's own. Nothing here derives from an AGPL-licensed project.

Rules that keep it that way, and keep it working:

* Never read or copy Ultralytics code or weights — the API shape is ours to
  choose, the implementation must stay clean-room.
* Use only the original COCO-trained checkpoints; the Objects365-pretrained ones may
  carry that dataset's terms.
* The exported IR already outputs sigmoid probabilities — never apply a second
  sigmoid in inference.
* The exporter writes `labels.txt` next to the IR; downstream tools read it.

Deformable attention is the `grid_sample` formulation, so the export path stays
plain ONNX (opset 17) → OpenVINO; a test pins it to within 1e-4 of eager
PyTorch. Against the reference implementation, all five sizes load strictly and
give the same detections (worst box IoU 0.999, confidence within 0.001); the
IR matches PyTorch on CPU, Intel GPU and NPU.

