# Apache-2.0
"""``Results`` / ``Boxes`` — the objects every prediction call answers with.

One object holds the pixels, the boxes, and the ways to look at them::

    r = model("bus.jpg")[0]
    r.boxes.xyxy      # (N, 4) pixel coordinates
    r.boxes.conf      # (N,)
    r.boxes.cls       # (N,)
    r.names[int(r.boxes.cls[0])]
    r.plot(); r.save(); r.show()

Everything is plain numpy — no torch needed to look at a prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Boxes:
    """Detected boxes for one image.

    ``data`` is ``(N, 6)`` — ``x1 y1 x2 y2 conf cls`` — or ``(N, 7)`` when a
    tracker filled in ids: ``x1 y1 x2 y2 id conf cls``.
    """

    def __init__(self, data: np.ndarray, orig_shape: tuple[int, int]) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.size == 0:
            data = data.reshape(0, 6)
        if data.ndim != 2 or data.shape[1] not in (6, 7):
            raise ValueError(f"boxes data must be (N, 6) or (N, 7), got {data.shape}")
        self.data = data
        self.orig_shape = tuple(orig_shape)  # (h, w)
        self.is_track = data.shape[1] == 7

    # -- geometry -----------------------------------------------------------

    @property
    def xyxy(self) -> np.ndarray:
        """(N, 4) ``x1 y1 x2 y2`` in pixels."""
        return self.data[:, :4]

    @property
    def xywh(self) -> np.ndarray:
        """(N, 4) ``cx cy w h`` in pixels."""
        x = self.xyxy
        cx, cy = (x[:, 0] + x[:, 2]) / 2, (x[:, 1] + x[:, 3]) / 2
        return np.stack([cx, cy, x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]], axis=-1)

    @property
    def xyxyn(self) -> np.ndarray:
        """(N, 4) ``x1 y1 x2 y2`` normalized to 0..1."""
        h, w = self.orig_shape
        return self.xyxy / np.array([w, h, w, h], np.float32)

    @property
    def xywhn(self) -> np.ndarray:
        """(N, 4) ``cx cy w h`` normalized to 0..1."""
        h, w = self.orig_shape
        return self.xywh / np.array([w, h, w, h], np.float32)

    # -- scores -------------------------------------------------------------

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, -2]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, -1]

    @property
    def id(self) -> np.ndarray | None:
        """Track ids (``None`` unless the boxes came from :meth:`Detector.track`)."""
        return self.data[:, 4].astype(np.int32) if self.is_track else None

    # -- container niceties -------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: Any) -> Boxes:
        return Boxes(np.atleast_2d(self.data[i]), self.orig_shape)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def cpu(self) -> Boxes:  # torch-shaped no-ops, so ported snippets keep working
        return self

    def numpy(self) -> Boxes:
        return self

    def tolist(self) -> list[list[float]]:
        return self.data.tolist()

    def __repr__(self) -> str:
        return f"Boxes(shape={self.data.shape}, orig_shape={self.orig_shape})"


class Masks:
    """One mask per box (``task="segment"``): ``data`` is ``(N, H, W)`` bool."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, bool)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def xy(self) -> list[np.ndarray]:
        """Each mask's outline as ``(K, 2)`` pixel points (its largest part)."""
        import cv2

        outlines = []
        for mask in self.data:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            biggest = max(contours, key=cv2.contourArea) if contours else np.zeros((0, 1, 2))
            outlines.append(biggest.reshape(-1, 2).astype(np.float32))
        return outlines

    @property
    def area(self) -> np.ndarray:
        """Pixels inside each mask."""
        return self.data.reshape(len(self.data), -1).sum(1)

    def __repr__(self) -> str:
        return f"Masks(shape={self.data.shape})"


class Keypoints:
    """Keypoints per box (``task="pose"``): ``data`` is ``(N, K, 3)`` — x, y in
    pixels and a confidence in 0-1; boxes the keypoint model is not for have
    all-zero rows. ``spec`` says what the K are (``spec.names``, its skeleton):
    COCO's 17 body keypoints, or a model's own."""

    def __init__(self, data: np.ndarray, spec=None) -> None:
        from .pose import COCO

        self.data = np.asarray(data, np.float32)
        self.spec = spec or COCO

    @property
    def names(self) -> tuple[str, ...]:
        return self.spec.names

    def __len__(self) -> int:
        return len(self.data)

    @property
    def xy(self) -> np.ndarray:
        return self.data[..., :2]

    @property
    def conf(self) -> np.ndarray:
        return self.data[..., 2]

    def __repr__(self) -> str:
        return f"Keypoints(shape={self.data.shape})"


class Results:
    """One image's predictions, plus the pixels they came from."""

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str | Path = "",
        names: dict[int, str] | None = None,
        boxes: np.ndarray | None = None,
        speed: dict[str, float] | None = None,
        masks: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        keypoint_spec=None,
    ) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = str(path)
        self.names = names or {}
        self.boxes = Boxes(
            boxes if boxes is not None else np.zeros((0, 6), np.float32), self.orig_shape
        )
        self.speed = speed or {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        self.masks = Masks(masks) if masks is not None else None
        self.keypoints = Keypoints(keypoints, keypoint_spec) if keypoints is not None else None

    def __len__(self) -> int:
        return len(self.boxes)

    def name_of(self, index: int | float) -> str:
        return self.names.get(int(index), f"class_{int(index)}")

    # -- rendering ----------------------------------------------------------

    def plot(
        self, conf: bool = True, labels: bool = True, line_width: int | None = None
    ) -> np.ndarray:
        """Return a copy of the image with boxes (and masks, keypoints) drawn (BGR ndarray)."""
        from .plotting import draw_boxes, draw_keypoints, draw_masks

        image = self.orig_img
        if self.masks is not None and len(self.masks):
            image = draw_masks(image, self.masks.data, self.boxes.cls)
        image = draw_boxes(
            image,
            self.boxes,
            self.names,
            conf=conf,
            labels=labels,
            line_width=line_width,
        )
        if self.keypoints is not None and len(self.keypoints):
            image = draw_keypoints(image, self.keypoints.data, line_width=line_width,
                                   spec=self.keypoints.spec)
        return image

    def save(self, filename: str | Path | None = None) -> Path:
        """Write the annotated image; returns the path written."""
        import cv2

        if filename is None:
            stem = Path(self.path).stem or "image"
            filename = Path(f"{stem}.jpg")
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filename), self.plot())
        return filename

    def show(self, title: str | None = None) -> None:
        """Open a window with the annotated image (blocks until a key press)."""
        import cv2

        cv2.imshow(title or (Path(self.path).name or "easydetect"), self.plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -- text ---------------------------------------------------------------

    def verbose(self) -> str:
        """``"2 persons, 1 car, "`` — the middle of the per-image log line."""
        if not len(self.boxes):
            return "(no detections), "
        out = ""
        cls = self.boxes.cls.astype(int)
        for c in sorted(set(cls.tolist())):
            n = int((cls == c).sum())
            out += f"{n} {self.name_of(c)}{'s' * (n > 1)}, "
        return out

    def summary(self) -> list[dict[str, Any]]:
        """Detections as plain dicts — handy for JSON, CSV, or a quick print."""
        rows = []
        for i in range(len(self.boxes)):
            x1, y1, x2, y2 = (float(v) for v in self.boxes.xyxy[i])
            row: dict[str, Any] = {
                "name": self.name_of(self.boxes.cls[i]),
                "class": int(self.boxes.cls[i]),
                "confidence": round(float(self.boxes.conf[i]), 4),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
            if self.boxes.is_track:
                row["track_id"] = int(self.boxes.id[i])
            if self.masks is not None:
                row["mask"] = {"area": int(self.masks.area[i]),
                               "polygon": self.masks.xy[i].round(1).tolist()}
            if self.keypoints is not None and self.keypoints.conf[i].any():
                row["keypoints"] = {
                    name: {"x": round(float(x), 1), "y": round(float(y), 1),
                           "confidence": round(float(c), 3)}
                    for name, (x, y, c) in zip(self.keypoints.names, self.keypoints.data[i],
                                               strict=True)}
            rows.append(row)
        return rows

    def __str__(self) -> str:
        h, w = self.orig_shape
        return f"{h}x{w} {self.verbose()}".rstrip(", ")

    def __repr__(self) -> str:
        return f"Results({len(self.boxes)} detections, path={self.path!r})"
