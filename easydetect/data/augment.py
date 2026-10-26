# Apache-2.0
"""Training-time augmentation for boxes: the recipe ed trains with.

Photometric jitter, a random zoom-out onto a larger canvas, an IoU-constrained
random crop, and a horizontal flip — the steps of the original configs with the
probabilities and ranges they use (torchvision's defaults), written here in
NumPy/OpenCV. One difference: the zoom-out canvas is the picture's mean colour
rather than black, so the border is not a cue that says "zoomed".
Zoom-out makes objects small and crop makes them large, so a model sees each
class at many sizes and positions instead of only the framing the photos had.

Boxes are pixel ``xyxy`` float arrays in the original picture's coordinates;
the geometry moves a rectangle over the picture and renders it once, at the end.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

ZOOM_OUT_P, ZOOM_OUT_MAX = 0.5, 4.0          # canvas up to 4× the image, half the time
CROP_P = 0.8
CROP_MIN_IOUS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None)   # None: keep the whole image
CROP_SCALE, CROP_ASPECT, CROP_TRIES = (0.3, 1.0), (0.5, 2.0), 40
PHOTO_P = 0.5

DESCRIPTION = [
    f"photometric jitter (brightness, contrast, saturation, hue; p={PHOTO_P} each)",
    f"zoom-out onto a canvas up to {ZOOM_OUT_MAX:g}× (p={ZOOM_OUT_P})",
    f"IoU-constrained random crop, {CROP_SCALE[0]:g}–{CROP_SCALE[1]:g} of each side (p={CROP_P})",
    "horizontal flip (p=0.5)",
]


_LEVELS = np.arange(256, dtype=np.float32)


def photometric(img: np.ndarray) -> np.ndarray:
    """Brightness, contrast, saturation and hue, each on its own coin flip.

    Each is a per-value mapping, so each is a 256-entry table: brightness
    and contrast fold into one table for all three channels, saturation and
    hue into tables for the S and H planes.
    """
    brighten, stretch = random.random() < PHOTO_P, random.random() < PHOTO_P
    if brighten or stretch:
        table = _LEVELS * (random.uniform(0.875, 1.125) if brighten else 1.0)   # brightness
        if stretch:                                      # contrast ×0.5–1.5 about the mean grey
            grey = float(np.clip(table[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)], 0, 255).mean())
            table = grey + (np.clip(table, 0, 255) - grey) * random.uniform(0.5, 1.5)
        img = cv2.LUT(img, np.clip(table, 0, 255).astype(np.uint8))
    saturate, shift = random.random() < PHOTO_P, random.random() < PHOTO_P
    if saturate or shift:
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        if saturate:                                     # saturation ×0.5–1.5
            s = cv2.LUT(s, np.clip(_LEVELS * random.uniform(0.5, 1.5), 0, 255).astype(np.uint8))
        if shift:                                        # hue ±0.05 of the circle (OpenCV: 0–180)
            h = cv2.LUT(h, ((_LEVELS + random.uniform(-9, 9)) % 180).astype(np.uint8))
        img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    return img


def zoom_out(view: np.ndarray) -> np.ndarray:
    """Widen the view onto a canvas up to 4x the picture, the picture somewhere inside.

    Nothing is drawn here: ``view`` is the rectangle, in the picture's own
    pixel coordinates, that will become the output. The canvas outside the
    picture is filled when the view is finally rendered.
    """
    if random.random() >= ZOOM_OUT_P:
        return view
    w, h = view[2] - view[0], view[3] - view[1]
    s = random.uniform(1.0, ZOOM_OUT_MAX)
    cw, ch = int(w * s), int(h * s)
    left, top = random.randint(0, int(cw - w)), random.randint(0, int(ch - h))
    return np.array([view[0] - left, view[1] - top, view[0] - left + cw, view[1] - top + ch],
                    np.float32)


def _iou(region: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(region[0], boxes[:, 0])
    y1 = np.maximum(region[1], boxes[:, 1])
    x2 = np.minimum(region[2], boxes[:, 2])
    y2 = np.minimum(region[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    region_area = (region[2] - region[0]) * (region[3] - region[1])
    return inter / np.maximum(area + region_area - inter, 1e-9)


def iou_crop(view: np.ndarray, boxes: np.ndarray,
             classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Narrow the view to a region that overlaps some box by at least a random IoU.

    Boxes whose centre falls outside the region are dropped; the rest are
    clipped to it. A picture with no boxes (a background frame) takes any
    region. When no region qualifies, the view stays as it was.
    """
    if random.random() >= CROP_P:
        return view, boxes, classes
    w, h = view[2] - view[0], view[3] - view[1]
    min_iou = random.choice(CROP_MIN_IOUS)
    if min_iou is None:
        return view, boxes, classes
    for _ in range(CROP_TRIES):
        cw = w * random.uniform(*CROP_SCALE)
        ch = h * random.uniform(*CROP_SCALE)
        if not CROP_ASPECT[0] <= cw / ch <= CROP_ASPECT[1]:
            continue
        cw, ch = max(int(cw), 1), max(int(ch), 1)
        left = view[0] + random.randint(0, int(w - cw))
        top = view[1] + random.randint(0, int(h - ch))
        region = np.array([left, top, left + cw, top + ch], np.float32)
        if len(boxes):
            if _iou(region, boxes).max() < min_iou:
                continue
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            inside = (cx > region[0]) & (cx < region[2]) & (cy > region[1]) & (cy < region[3])
            if not inside.any():
                continue
            kept = boxes[inside].copy()
            kept[:, [0, 2]] = np.clip(kept[:, [0, 2]], region[0], region[2])
            kept[:, [1, 3]] = np.clip(kept[:, [1, 3]], region[1], region[3])
            boxes, classes = kept, classes[inside]
        return region, boxes, classes
    return view, boxes, classes


def render(img: np.ndarray, view: np.ndarray, size: tuple[int, int],
           flip: bool) -> np.ndarray:
    """Draw ``view`` (picture coordinates, may reach past the picture) at ``size``.

    One affine warp straight to the output, so a 4x zoom-out never builds a
    canvas 4x the photo. Outside the picture is its mean colour. Any
    shrink goes through an area resize first, so fine detail averages
    instead of aliasing (the original pictures are resized with antialiasing,
    which is what its COCO weights learned on).
    """
    h, w = img.shape[:2]
    out_w, out_h = size
    sx, sy = out_w / (view[2] - view[0]), out_h / (view[3] - view[1])
    inside = view[0] >= 0 and view[1] >= 0 and view[2] <= w and view[3] <= h
    shrink = max(sx, sy) < 1.0
    if inside and not flip and view[0] == 0 and view[1] == 0 and view[2] == w and view[3] == h:
        return cv2.resize(img, (out_w, out_h),
                          interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    src, fx, fy = img, 1.0, 1.0
    if shrink:                                  # average first, then place
        nw, nh = max(1, round(w * min(sx, 1.0))), max(1, round(h * min(sy, 1.0)))
        src = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        fx, fy = nw / w, nh / h
    ax, ay = sx / fx, sy / fy
    if flip:
        m = np.array([[-ax, 0, view[2] * sx], [0, ay, -view[1] * sy]], np.float32)
    else:
        m = np.array([[ax, 0, -view[0] * sx], [0, ay, -view[1] * sy]], np.float32)
    fill = cv2.mean(img)[:3]
    return cv2.warpAffine(src, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def apply(img: np.ndarray, labels: np.ndarray, strong: bool = True,
          size: int | tuple[int, int] | None = None,
          min_side: float = 1.0, zoom: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Augment one picture and its ``cls cx cy w h`` (normalised) labels.

    ``size`` renders the result square at that many pixels (what training
    wants, and the cheap way), or at ``(width, height)``; ``None`` keeps the
    view's own pixel size.
    ``strong=False`` keeps only the flip — what the last epochs train on, as
    in the original, so the model settles on pictures framed like the ones it will
    see. ``min_side`` drops boxes thinner than that many output pixels.
    ``zoom=False`` skips the zoom-out (a mosaic tile is small already).
    Colour jitter runs last, on the small output: it does not care where the
    pixels came from, and there are far fewer of them there.
    """
    h, w = img.shape[:2]
    classes = labels[:, 0].copy()
    boxes = np.empty((len(labels), 4), np.float32)
    boxes[:, 0] = (labels[:, 1] - labels[:, 3] / 2) * w
    boxes[:, 1] = (labels[:, 2] - labels[:, 4] / 2) * h
    boxes[:, 2] = (labels[:, 1] + labels[:, 3] / 2) * w
    boxes[:, 3] = (labels[:, 2] + labels[:, 4] / 2) * h

    view = np.array([0, 0, w, h], np.float32)
    if strong:
        if zoom:
            view = zoom_out(view)
        view, boxes, classes = iou_crop(view, boxes, classes)
    flip = random.random() < 0.5

    vw, vh = view[2] - view[0], view[3] - view[1]
    if isinstance(size, int):
        out_size = (size, size)
    elif size:
        out_size = (int(size[0]), int(size[1]))
    else:
        out_size = (max(1, round(float(vw))), max(1, round(float(vh))))
    img = render(img, view, out_size, flip)
    if strong:
        img = photometric(img)

    # boxes into the output: relative to the view, mirrored if flipped
    rel = boxes - np.array([view[0], view[1], view[0], view[1]], np.float32)
    if flip:
        rel[:, [0, 2]] = vw - rel[:, [2, 0]]
    rel /= np.array([vw, vh, vw, vh], np.float32)
    ow, oh = out_size
    keep = ((rel[:, 2] - rel[:, 0]) * ow >= min_side) & ((rel[:, 3] - rel[:, 1]) * oh >= min_side)
    rel, classes = rel[keep], classes[keep]
    out = np.empty((len(rel), 5), np.float32)
    out[:, 0] = classes
    out[:, 1] = (rel[:, 0] + rel[:, 2]) / 2
    out[:, 2] = (rel[:, 1] + rel[:, 3]) / 2
    out[:, 3] = rel[:, 2] - rel[:, 0]
    out[:, 4] = rel[:, 3] - rel[:, 1]
    return img, out


MOSAIC_SPLIT = (0.3, 0.7)  # where the four tiles meet, as a share of each side
MIXUP_RATIO = (0.4, 0.6)  # how much of the first picture shows through


def mosaic(pictures: list[tuple[np.ndarray, np.ndarray]], size: int,
           min_side: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Four pictures as the four tiles of one, meeting at a random point.

    Each tile is its picture cropped, flipped and jittered as usual and
    rendered straight at the tile's size, so objects come out smaller and in
    unusual company — more objects a step. No zoom-out inside a tile: a
    quarter-sized tile zoomed out 4× more would leave most objects a few
    pixels wide, too small to learn from. Labels are ``cls cx cy w h``
    normalised, in and out.
    """
    cx = round(size * random.uniform(*MOSAIC_SPLIT))
    cy = round(size * random.uniform(*MOSAIC_SPLIT))
    tiles = ((0, 0, cx, cy), (cx, 0, size, cy), (0, cy, cx, size), (cx, cy, size, size))
    canvas = np.empty((size, size, 3), np.uint8)
    out = []
    for (img, labels), (x0, y0, x1, y1) in zip(pictures, tiles, strict=True):
        tile, tile_labels = apply(img, labels, strong=True, size=(x1 - x0, y1 - y0),
                                  min_side=min_side, zoom=False)
        canvas[y0:y1, x0:x1] = tile
        if len(tile_labels):
            tile_labels = tile_labels.copy()
            tile_labels[:, [1, 3]] *= (x1 - x0) / size
            tile_labels[:, [2, 4]] *= (y1 - y0) / size
            tile_labels[:, 1] += x0 / size
            tile_labels[:, 2] += y0 / size
            out.append(tile_labels)
    labels = np.concatenate(out) if out else np.zeros((0, 5), np.float32)
    return canvas, labels.astype(np.float32)


def mixup(a: tuple[np.ndarray, np.ndarray],
          b: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Two same-sized pictures laid over each other, with both sets of boxes:
    the model learns to find an object through clutter that is not part of it."""
    ratio = random.uniform(*MIXUP_RATIO)
    img = cv2.addWeighted(a[0], ratio, b[0], 1.0 - ratio, 0.0)
    return img, np.concatenate([a[1], b[1]]).astype(np.float32)
