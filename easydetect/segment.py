# Apache-2.0
"""Masks for the boxes a detector found: MobileSAM, prompted with each box.

The detector says where and what; MobileSAM (Apache-2.0) outlines what is
inside each box, whatever the class — a model trained on your own boxes gets
masks without a single mask label. The image is encoded once (the heavy part:
about 150 ms on a 4-core CPU), then each box costs one small decoder pass
(about 26 ms). See tools/convert_sam.py for how the two ONNX files are built.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

SIZE = 1024  # MobileSAM's input: longest side 1024, zero-padded to a square


class BoxSegmenter:
    """``segmenter(img, xyxy) -> (masks (N, H, W) bool, scores (N,))``."""

    def __init__(self, encoder: str | Path, decoder: str | Path, device: str = "CPU",
                 backend: str | None = None) -> None:
        from .predictor import installed

        self.backend = backend or ("openvino" if installed("openvino") else "onnxruntime")
        self._local = threading.local()
        if self.backend == "openvino":
            import openvino as ov

            core = ov.Core()
            self._encoder = core.compile_model(str(encoder), device)
            self._decoder = core.compile_model(str(decoder), device)
        else:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            self._encoder = ort.InferenceSession(str(encoder), providers=providers)
            self._decoder = ort.InferenceSession(str(decoder), providers=providers)

    def _run(self, which: str, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        model = self._encoder if which == "encoder" else self._decoder
        if self.backend == "onnxruntime":
            return model.run(None, feed)
        # one infer request per thread, as the detector does
        requests = self._local.__dict__.setdefault("requests", {})
        if which not in requests:
            requests[which] = model.create_infer_request()
        out = requests[which].infer(feed)
        return [out[o] for o in model.outputs]

    @staticmethod
    def preprocess(img: np.ndarray) -> tuple[np.ndarray, float]:
        """BGR picture -> (1x3x1024x1024 RGB 0-255, scale from picture to model pixels)."""
        import cv2

        h, w = img.shape[:2]
        scale = SIZE / max(h, w)
        resized = cv2.resize(img, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((SIZE, SIZE, 3), np.float32)
        canvas[:resized.shape[0], :resized.shape[1]] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return canvas.transpose(2, 0, 1)[None], scale

    def __call__(self, img: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        if not len(xyxy):
            return np.zeros((0, h, w), bool), np.zeros(0, np.float32)
        tensor, scale = self.preprocess(img)
        embeddings = self._run("encoder", {"image": tensor})[0]
        masks, scores, _ = self._run("decoder", {
            "image_embeddings": embeddings,
            "point_coords": (xyxy.reshape(-1, 2, 2) * scale).astype(np.float32),
            "point_labels": np.tile(np.array([[2, 3]], np.float32), (len(xyxy), 1)),
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([h, w], np.float32),
        })
        return masks[:, 0] > 0.0, scores[:, 0].astype(np.float32)


def default_segmenter(device: str = "CPU", backend: str | None = None,
                      decoder: str | Path | None = None) -> BoxSegmenter:
    """The mirror's MobileSAM, downloaded once into the cache; ``decoder``, a
    mask decoder fine-tuned on your own masks, in place of MobileSAM's."""
    from .downloads import download_segmenter

    encoder, mirror_decoder = download_segmenter()
    return BoxSegmenter(encoder, decoder or mirror_decoder, device=device, backend=backend)
