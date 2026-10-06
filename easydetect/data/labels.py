# Apache-2.0
"""Where a label lives, and how one line of it reads — shared by everything.

The trainer reads labels, easydetect lab writes them, the exporter packs them.
If any two of those disagree on where ``a.jpg``'s boxes are, training runs on
nothing and nobody is told. So there is one rule, here, and no torch import:
the lab uses it at startup.
"""

from __future__ import annotations

from pathlib import Path, PurePath

import numpy as np


def label_path(image: str | PurePath) -> PurePath:
    """``…/images/train/a.jpg`` -> ``…/labels/train/a.txt``.

    The last folder called ``images`` becomes ``labels``; with none, the label
    sits beside the image. Works on relative paths too, which is how an export
    lays out a zip that the trainer will read after unpacking.
    """
    image = image if isinstance(image, PurePath) else Path(image)
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):  # folders only, never the file name
        if parts[i] == "images":
            parts[i] = "labels"
            return type(image)(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def label_row_to_box(values: list[str], kpt_shape=None) -> list[float] | None:
    """One label line -> ``[cls, cx, cy, w, h]``, or None for a line to skip.

    ``cls cx cy w h`` is a box (a trailing sixth value, a confidence, is
    ignored). ``cls x1 y1 x2 y2 x3 y3 ...`` is a segmentation polygon, common
    in exported datasets; reading its first four numbers as a box would train
    on nonsense without a single error, so it becomes its bounding box.

    With ``kpt_shape`` (a keypoint dataset's ``[K, 2 or 3]``) a line is
    ``cls cx cy w h`` and then the keypoints: told apart from a polygon by the
    data.yaml, not by counting — four keypoints of three numbers make 17
    values, which would otherwise read as a polygon.
    """
    if kpt_shape is not None and len(values) >= 5:
        return [float(v) for v in values[:5]]
    if len(values) >= 7 and len(values) % 2 == 1:
        xy = np.asarray(values[1:], np.float32).reshape(-1, 2)
        (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
        return [float(values[0]), float(x0 + x1) / 2, float(y0 + y1) / 2,
                float(x1 - x0), float(y1 - y0)]
    if len(values) >= 5:
        return [float(v) for v in values[:5]]
    return None


def label_row_to_keypoints(values: list[str], kpt_shape) -> tuple[list[float], np.ndarray] | None:
    """A keypoint-dataset line (``cls cx cy w h`` then ``kpt_shape[0]``
    keypoints of ``x y`` or ``x y visibility``, all 0-1) -> ``([cls, cx, cy,
    w, h], (K, 3) x, y, visibility)``; None for a line that does not fit.
    Without a visibility column a keypoint at (0, 0) is the unlabelled one."""
    k, dims = int(kpt_shape[0]), int(kpt_shape[1])
    if len(values) < 5 + k * dims:
        return None
    box = [float(v) for v in values[:5]]
    kpts = np.asarray(values[5:5 + k * dims], np.float32).reshape(k, dims)
    if dims == 2:
        seen = (kpts[:, 0] > 0) | (kpts[:, 1] > 0)
        kpts = np.concatenate([kpts, np.where(seen, 2.0, 0.0)[:, None]], 1).astype(np.float32)
    return box, kpts
