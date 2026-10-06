# Apache-2.0
"""easydetect — real-time object detection you can ship, 100% Apache-2.0.

    pip install easydetect

    from easydetect import Detector

    model = Detector("dfine-s")                      # weights fetched from the mirror
    results = model("photo.jpg")                     # list[Results]
    results[0].boxes.xyxy, results[0].boxes.conf     # plain numpy
    results[0].save()

    model.train(data="data.yaml", epochs=50)         # your own dataset
    model.val(data="data.yaml").box.map50            # COCO-style mAP
    model.export(format="openvino", half=True)       # IR + labels.txt

The detector is D-FINE (Peterande/D-FINE, Apache-2.0), a real-time DETR, in
five sizes n/s/m/l/x; its released COCO weights load unchanged. The API,
trainer, validator, exporter, predictor and CLI are this project's own. No AGPL
code or weights anywhere, so this package can ship inside a product. Inference
needs numpy/opencv/openvino/pyyaml; training adds
``pip install "easydetect[train]"`` (torch, torchvision, scipy, onnx).
"""

from __future__ import annotations

__version__ = "0.6.0"

from .errors import DownloadError, EasyDetectError, ModelNotFoundError
from .metrics import BoxMetrics, DetMetrics
from .model import Detector
from .results import Boxes, Results

__all__ = [
    "Detector",
    "Results",
    "Boxes",
    "DetMetrics",
    "BoxMetrics",
    "EasyDetectError",
    "ModelNotFoundError",
    "DownloadError",
    "__version__",
]
