# Apache-2.0
"""Masks for the boxes a detector found: MobileSAM, prompted with each box.

The detector says where and what; MobileSAM (Apache-2.0) outlines what is
inside each box, whatever the class — a model trained on your own boxes gets
masks without a single mask label. The image is encoded once (the heavy part:
about 150 ms on a 4-core CPU), then each box costs one small decoder pass
(about 26 ms). See tools/convert_sam.py for how the two ONNX files are built.

The decoder's own full-size output (one float mask the size of the picture per
box: 1 GB for 30 boxes on a 4K frame) is not used. Its 256 x 256 low-res
logits are, scaled up to the picture only around each box, with the same
geometry SAM's postprocessing has.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

SIZE = 1024  # MobileSAM's input: longest side 1024, zero-padded to a square
LOW_RES = 256  # the decoder's low-res masks cover the SIZE x SIZE frame


class BoxSegmenter:
    """``segmenter(img, xyxy) -> (masks (N, H, W) bool, scores (N,))``.

    ``box_pad``: each mask is cut to its box grown by this share of the box's
    width and height on each side (0.1 by default). A box prompt means "the
    thing in this box", so a mask spilling far outside it has leaked into the
    background or a neighbour; ``None`` keeps the whole mask. ``fill_holes``
    fills holes inside each mask and ``largest`` keeps only its largest
    connected part; both are off, as SAM's masks are often right to have
    holes (a cup's handle) and several parts (a partly hidden object).
    """

    def __init__(self, encoder: str | Path, decoder: str | Path, device: str = "CPU",
                 backend: str | None = None, box_pad: float | None = 0.1,
                 fill_holes: bool = False, largest: bool = False) -> None:
        from .predictor import installed

        self.backend = backend or ("openvino" if installed("openvino") else "onnxruntime")
        self.box_pad = box_pad
        self.fill_holes = fill_holes
        self.largest = largest
        self._local = threading.local()
        if self.backend == "openvino":
            from .predictor import ov_core

            core = ov_core()
            self._encoder = core.compile_model(str(encoder), device)
            self._decoder = core.compile_model(_low_res_only(core.read_model(str(decoder))),
                                               device)
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

    def _decode(self, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """``(low-res mask logits (N, 256, 256), predicted IoU (N,))``."""
        if self.backend == "onnxruntime":
            scores, low = self._decoder.run(["iou_predictions", "low_res_masks"], feed)
        else:
            scores, low = self._run("decoder", feed)  # outputs pruned to these two
        return low[:, 0], scores[:, 0]

    @staticmethod
    def preprocess(img: np.ndarray) -> tuple[np.ndarray, float]:
        """BGR picture -> (1x3x1024x1024 RGB 0-255, scale from picture to model pixels)."""
        import cv2

        h, w = img.shape[:2]
        scale = SIZE / max(h, w)
        # SAM shrinks with an antialiased resize; INTER_AREA is OpenCV's nearest
        # match, where INTER_LINEAR would alias fine detail on a big picture
        resized = cv2.resize(img, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
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
        low, scores = self._decode({
            "image_embeddings": embeddings,
            "point_coords": (xyxy.reshape(-1, 2, 2) * scale).astype(np.float32),
            "point_labels": np.tile(np.array([[2, 3]], np.float32), (len(xyxy), 1)),
            "mask_input": np.zeros((1, 1, LOW_RES, LOW_RES), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
            # the full-size output goes unused, so ask for it small
            "orig_im_size": np.array(_low_res_size(h, w), np.float32),
        })
        masks = np.zeros((len(xyxy), h, w), bool)
        for mask, logits, box in zip(masks, low, xyxy, strict=True):
            x1, y1, x2, y2 = self._region(box, h, w)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = upscale_logits(logits, (h, w), (x1, y1, x2, y2)) > 0
            mask[y1:y2, x1:x2] = self._clean(roi)
        return masks, scores.astype(np.float32)

    def _region(self, box: np.ndarray, h: int, w: int) -> tuple[int, int, int, int]:
        """The part of the picture a box's mask may cover: the box grown by
        ``box_pad`` (the whole picture when that is ``None``)."""
        if self.box_pad is None:
            return 0, 0, w, h
        x1, y1, x2, y2 = (float(v) for v in box)
        px, py = (x2 - x1) * self.box_pad, (y2 - y1) * self.box_pad
        return (max(int(np.floor(x1 - px)), 0), max(int(np.floor(y1 - py)), 0),
                min(int(np.ceil(x2 + px)), w), min(int(np.ceil(y2 + py)), h))

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        if not (self.fill_holes or self.largest) or not mask.any():
            return mask
        import cv2

        m = mask.astype(np.uint8)
        if self.largest:
            _, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        if self.fill_holes:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(m, contours, -1, 1, cv2.FILLED)
        return m.astype(bool)


def _low_res_size(h: int, w: int) -> tuple[int, int]:
    s = LOW_RES / max(h, w)
    return max(round(h * s), 1), max(round(w * s), 1)


def _low_res_only(model):
    """An OpenVINO model of the decoder without its full-size masks output,
    so that part of the graph never runs."""
    import openvino as ov

    keep = [o for o in model.outputs if o.get_names() & {"iou_predictions", "low_res_masks"}]
    if len(keep) != 2:
        keep = list(model.outputs)[1:3]  # SAM's order: masks, iou_predictions, low_res_masks
    keep.sort(key=lambda o: "low_res_masks" in o.get_names())  # scores first
    return ov.Model(keep, model.get_parameters(), "sam_decoder_low_res")


def upscale_logits(low: np.ndarray, image_size: tuple[int, int],
                   region: tuple[int, int, int, int]) -> np.ndarray:
    """The decoder's ``(256, 256)`` logits scaled up to the picture, for the
    picture's ``region`` (x1, y1, x2, y2) only.

    SAM's postprocessing, done one box at a time: bilinear up to the 1024
    frame, crop the padding off, bilinear to the picture's size (both with
    PyTorch's align_corners=False, which is OpenCV's pixel-centre convention
    and edge clamping). Each step warps only the pixels the next one reads,
    so nothing the size of the whole picture is ever made.
    """
    import cv2

    h, w = image_size
    x1, y1, x2, y2 = region
    s = SIZE / max(h, w)
    ph, pw = int(np.floor(h * s + 0.5)), int(np.floor(w * s + 0.5))  # the unpadded part
    sx, sy = pw / w, ph / h

    def span(lo, hi, f, size):  # the 1024-frame pixels that picture pixels lo..hi-1 read
        first = max(int(np.floor((lo + 0.5) * f - 0.5)), 0)
        last = min(int(np.floor((hi - 0.5) * f - 0.5)) + 1, size - 1)
        return first, max(last, first)

    u0, u1 = span(x1, x2, sx, pw)
    v0, v1 = span(y1, y2, sy, ph)
    k = LOW_RES / SIZE
    to_low = np.array([[k, 0, (u0 + 0.5) * k - 0.5], [0, k, (v0 + 0.5) * k - 0.5]], np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    frame = cv2.warpAffine(np.ascontiguousarray(low, np.float32), to_low,
                           (u1 - u0 + 1, v1 - v0 + 1), flags=flags,
                           borderMode=cv2.BORDER_REPLICATE)
    to_picture = np.array([[sx, 0, (x1 + 0.5) * sx - 0.5 - u0],
                           [0, sy, (y1 + 0.5) * sy - 0.5 - v0]], np.float32)
    return cv2.warpAffine(frame, to_picture, (x2 - x1, y2 - y1), flags=flags,
                          borderMode=cv2.BORDER_REPLICATE)


def default_segmenter(device: str = "CPU", backend: str | None = None,
                      decoder: str | Path | None = None) -> BoxSegmenter:
    """The mirror's MobileSAM, downloaded once into the cache; ``decoder``, a
    mask decoder fine-tuned on your own masks, in place of MobileSAM's."""
    from .downloads import download_segmenter

    encoder, mirror_decoder = download_segmenter()
    return BoxSegmenter(encoder, decoder or mirror_decoder, device=device, backend=backend)
