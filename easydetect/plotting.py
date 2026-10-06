# Apache-2.0
"""Box drawing. Deliberately small: OpenCV rectangles and labels, no font files."""

from __future__ import annotations

import numpy as np

#: A 20-colour palette (BGR), cycled by class index.
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def color_for(index: int) -> tuple[int, int, int]:
    return PALETTE[int(index) % len(PALETTE)]


def draw_boxes(
    img: np.ndarray,
    boxes,
    names: dict[int, str] | None = None,
    conf: bool = True,
    labels: bool = True,
    line_width: int | None = None,
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw ``boxes`` (a :class:`~easydetect.results.Boxes`) onto a copy of ``img``.

    ``color`` (BGR) overrides the per-class palette — what you want when two
    sets of boxes share a frame and the distinction is truth versus prediction
    rather than one class versus another.
    """
    import cv2

    out = np.ascontiguousarray(img.copy())
    names = names or {}
    h, w = out.shape[:2]
    lw = line_width or max(round((h + w) / 2 * 0.003), 2)
    font_scale = lw / 3.0

    ids = boxes.id if boxes.is_track else None
    placed: list[tuple[int, int, int, int]] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (int(round(v)) for v in boxes.xyxy[i])
        cls = int(boxes.cls[i])
        box_color = color or color_for(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, lw, cv2.LINE_AA)
        if not labels:
            continue
        text = names.get(cls, f"class_{cls}")
        if ids is not None:
            text = f"id:{int(ids[i])} {text}"
        if conf:
            text += f" {float(boxes.conf[i]):.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(lw - 1, 1))
        left, top = _label_position(x1, y1, y2, tw, th + 3, (h, w), placed)
        placed.append((left, top, left + tw, top + th + 3))
        cv2.rectangle(out, (left, top), (left + tw, top + th + 3), box_color, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            text,
            (left, top + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(lw - 1, 1),
            cv2.LINE_AA,
        )
    return out


def draw_masks(img: np.ndarray, masks: np.ndarray, classes, alpha: float = 0.45) -> np.ndarray:
    """Tint each mask with its class colour (a copy of ``img``)."""
    out = img.astype(np.float32, copy=True)
    for mask, cls in zip(masks, classes, strict=True):
        out[mask] = out[mask] * (1 - alpha) + np.array(color_for(int(cls)), np.float32) * alpha
    return out.astype(np.uint8)


#: limb colours (BGR): face, arms, body, legs
_LIMB_COLOURS = {"face": (255, 200, 80), "arm": (80, 220, 255), "body": (120, 255, 120),
                 "leg": (255, 120, 200)}


#: line colours (BGR) for a keypoint set of a model's own, in skeleton order
_EDGE_COLOURS = ((255, 200, 80), (80, 220, 255), (120, 255, 120), (255, 120, 200),
                 (80, 160, 255), (200, 120, 255), (255, 255, 120), (120, 200, 200))


def _limb_kind(a: int, b: int) -> str:
    pair = {a, b}
    if pair in ({5, 6}, {5, 11}, {6, 12}, {11, 12}):
        return "body"
    if pair <= set(range(5, 11)):
        return "arm"
    if pair <= set(range(11, 17)):
        return "leg"
    return "face"  # the face, and ears to shoulders


def draw_keypoints(img: np.ndarray, keypoints: np.ndarray, threshold: float = 0.3,
                   line_width: int | None = None, spec=None) -> np.ndarray:
    """Draw each box's keypoints (``(N, K, 3)`` x, y, conf) and the lines of
    ``spec``'s skeleton (COCO's body by default) onto a copy of ``img``,
    leaving out keypoints under ``threshold``."""
    import cv2

    from .pose import COCO

    spec = spec or COCO
    out = np.ascontiguousarray(img.copy())
    h, w = out.shape[:2]
    lw = line_width or max(round((h + w) / 2 * 0.003), 2)
    for person in keypoints:
        seen = person[:, 2] >= threshold
        for n, (a, b) in enumerate(spec.skeleton):
            if seen[a] and seen[b]:
                colour = (_LIMB_COLOURS[_limb_kind(a, b)] if spec.is_coco
                          else _EDGE_COLOURS[n % len(_EDGE_COLOURS)])
                cv2.line(out, (int(person[a, 0]), int(person[a, 1])),
                         (int(person[b, 0]), int(person[b, 1])), colour, lw, cv2.LINE_AA)
        for x, y, _ in person[seen]:
            cv2.circle(out, (int(x), int(y)), lw + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), lw + 1, (40, 40, 40), 1, cv2.LINE_AA)
    return out


def _label_position(x1, y1, y2, tw, th, shape, placed):
    """Where to put one label: inside the frame, and clear of its neighbours.

    Crowded scenes are the normal case for a detector, and labels that land on
    top of each other are unreadable ("personperson 0.87n 0.87"). Try above the
    box, then below it, then just inside it, and take the first free slot.
    """
    h, w = shape
    left = max(0, min(int(x1), w - tw))
    for top in (y1 - th, y2, y1, y1 + th):
        top = max(0, min(int(top), h - th))
        box = (left, top, left + tw, top + th)
        if not any(_overlaps(box, other) for other in placed):
            return left, top
    return left, max(0, min(int(y1 - th), h - th))


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
