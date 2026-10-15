# Apache-2.0
"""Source resolution: everything a model can be pointed at.

Accepted::

    "bus.jpg"                 a single image
    "frames/*.jpg"            a glob
    "dataset/images"          a folder (recursed, images + videos)
    "list.txt"                a text file of paths, one per line
    "https://…/bus.jpg"       an image URL
    "https://…/clip.mp4"      a video URL / rtsp:// / rtmp:// stream
    "clip.mp4"                a video file
    0                         a webcam index
    ndarray | PIL.Image       pixels you already have
    [ …any of the above… ]    a list, processed in order

Pictures and video files are read a couple of frames ahead on a thread of
their own, so decoding the next frame overlaps running the model on this one.
A live source (a webcam, rtsp://…) is read on its own thread too, but only its
newest frame is kept: a model slower than the camera skips frames rather than
falling further and further behind it.
"""

from __future__ import annotations

import glob as _glob
import queue
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dng", ".pfm"}
VID_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv", ".webm", ".gif"}
_URL_RE = re.compile(r"^(https?|rtsp|rtmp|tcp|udp)://", re.IGNORECASE)
_LIVE_RE = re.compile(r"^(rtsp|rtmp|tcp|udp)://", re.IGNORECASE)
PREFETCH = 2  # frames read ahead of the model from pictures and video files


def is_url(value: Any) -> bool:
    return isinstance(value, (str, Path)) and bool(_URL_RE.match(str(value)))


class Frame:
    """One unit of work handed to the predictor."""

    __slots__ = ("img", "path", "index", "total", "kind", "frame", "frames", "fps")

    def __init__(self, img, path, index, total, kind, frame=0, frames=0, fps=0.0):
        self.img = img  # BGR ndarray
        self.path = str(path)
        self.index = index  # 1-based source index
        self.total = total  # number of sources, 0 when unbounded
        self.kind = kind  # "image" | "video" | "stream"
        self.frame = frame  # 1-based frame number within a video
        self.frames = frames  # total frames, 0 when unknown
        self.fps = fps  # frames per second after vid_stride, 0 when unknown

    def prefix(self) -> str:
        """``"image 1/3 bus.jpg: "`` — the head of the verbose log line."""
        total = self.total or 1
        if self.kind == "image":
            return f"image {self.index}/{total} {self.path}: "
        where = f"(frame {self.frame}/{self.frames})" if self.frames else f"(frame {self.frame})"
        return f"{self.kind} {self.index}/{total} {where} {self.path}: "


def _pil_to_bgr(img) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def _read_url_image(url: str) -> np.ndarray:
    import cv2

    from .downloads import read_url_bytes

    buf = np.frombuffer(read_url_bytes(url), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode an image from {url}")
    return img


def _expand(source: Any) -> list[tuple[str, Any]]:
    """Flatten one user-supplied source into ``(kind, payload)`` entries."""
    if isinstance(source, (list, tuple)):
        return [e for s in source for e in _expand(s)]
    if isinstance(source, np.ndarray):
        return [("array", source)]
    if hasattr(source, "convert") and hasattr(source, "size"):  # PIL.Image
        return [("array", _pil_to_bgr(source))]
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return [("stream", int(source))]
    if is_url(source):
        url = str(source)
        kind = "url-image" if Path(url.split("?")[0]).suffix.lower() in IMG_FORMATS else "stream"
        return [(kind, url)]

    p = Path(str(source))
    if any(ch in str(source) for ch in "*?[") and not p.exists():
        files = sorted(_glob.glob(str(source), recursive=True))
        if not files:
            raise FileNotFoundError(f"no files match {source!r}")
        return [e for f in files for e in _expand(f)]
    if p.is_dir():
        files = sorted(
            f for f in p.rglob("*") if f.suffix.lower() in IMG_FORMATS | VID_FORMATS
        )
        if not files:
            raise FileNotFoundError(f"no images or videos under {p}")
        return [e for f in files for e in _expand(f)]
    if not p.exists():
        raise FileNotFoundError(f"source not found: {p}")
    suffix = p.suffix.lower()
    if suffix in IMG_FORMATS:
        return [("image", p)]
    if suffix in VID_FORMATS:
        return [("video", p)]
    if suffix == ".txt":
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [e for ln in lines for e in _expand(ln if Path(ln).is_absolute() else p.parent / ln)]
    raise ValueError(f"unsupported source type: {p}")


def _is_live(payload: Any, cap) -> bool:
    """A webcam or a network stream, not a file: its frames come at the
    camera's pace whether or not anyone reads them."""
    import cv2

    if isinstance(payload, int) or _LIVE_RE.match(str(payload)):
        return True
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0  # e.g. an http MJPEG stream


class _Newest:
    """Reads a live capture on a thread of its own and keeps only the newest
    frame, so a slow model always gets the latest picture instead of one that
    waited in a queue. Frames skipped by ``stride`` are grabbed, not decoded.
    The thread releases the capture when it ends."""

    def __init__(self, cap, stride: int = 1) -> None:
        self.cap, self.stride = cap, stride
        self._cond = threading.Condition()
        self._img, self._count, self._done, self._stop = None, 0, False, False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            n = 0
            while not self._stop:
                if n % self.stride:
                    ok, img = self.cap.grab(), None
                else:
                    ok, img = self.cap.read()
                if not ok:
                    break
                n += 1
                if img is not None:
                    with self._cond:
                        self._img, self._count = img, self._count + 1
                        self._cond.notify_all()
        finally:
            self.cap.release()
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def next(self, after: int) -> tuple[int, np.ndarray] | None:
        """The newest frame once it is newer than frame ``after``: ``(its
        number, pixels)``, or ``None`` when the source has ended."""
        with self._cond:
            self._cond.wait_for(lambda: self._count > after or self._done)
            if self._count > after:
                return self._count, self._img
            return None

    def close(self) -> None:
        self._stop = True
        self._thread.join(timeout=2.0)


def _read_ahead(frames: Iterator[Frame], depth: int = PREFETCH) -> Iterator[Frame]:
    """``frames``, produced on a thread of its own up to ``depth`` ahead of
    whoever iterates; its errors are raised here, where they would have been."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def work() -> None:
        try:
            for frame in frames:
                if not put(("frame", frame)):
                    return
            put(("end", None))
        except BaseException as exc:  # handed over and raised by the reader
            put(("error", exc))
        finally:
            frames.close()  # on this thread, where the generator runs

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        while True:
            kind, item = q.get()
            if kind == "end":
                return
            if kind == "error":
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=2.0)


class SourceLoader:
    """Iterates a resolved source as :class:`Frame` objects."""

    def __init__(self, source: Any, vid_stride: int = 1) -> None:
        self.entries = _expand(source)
        self.vid_stride = max(int(vid_stride), 1)
        self.total = len(self.entries)
        self.is_stream = any(k == "stream" for k, _ in self.entries)

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Iterator[Frame]:
        # never read ahead of a stream: a live one must be read as it comes
        if self.is_stream:
            return self._frames()
        return _read_ahead(self._frames())

    def _frames(self) -> Iterator[Frame]:
        import cv2

        for i, (kind, payload) in enumerate(self.entries, start=1):
            if kind == "array":
                yield Frame(payload, "image.jpg", i, self.total, "image")
            elif kind == "url-image":
                yield Frame(_read_url_image(payload), payload, i, self.total, "image")
            elif kind == "image":
                img = cv2.imread(str(payload))
                if img is None:
                    raise ValueError(f"could not read image: {payload}")
                yield Frame(img, payload, i, self.total, "image")
            else:  # video file, webcam index, or network stream
                cap = cv2.VideoCapture(payload if isinstance(payload, int) else str(payload))
                if not cap.isOpened():
                    raise ValueError(f"could not open {payload!r}")
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if kind == "video" else 0
                frames = max(frames, 0) // self.vid_stride
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                # a stride of 2 keeps every other frame, so playback halves too
                fps = fps / self.vid_stride if fps > 0 else 0.0
                label = "video" if kind == "video" else "stream"
                path = f"webcam{payload}" if isinstance(payload, int) else payload
                if kind == "stream" and _is_live(payload, cap):
                    yield from self._live(cap, path, i, label, fps)
                    continue
                n = 0
                try:
                    while True:
                        # frames vid_stride skips are grabbed, never decoded
                        if n % self.vid_stride:
                            ok, img = cap.grab(), None
                        else:
                            ok, img = cap.read()
                        if not ok:
                            break
                        n += 1
                        if img is None:
                            continue
                        yield Frame(img, path, i, self.total, label,
                                    frame=(n - 1) // self.vid_stride + 1, frames=frames, fps=fps)
                finally:
                    cap.release()

    def _live(self, cap, path, i: int, label: str, fps: float) -> Iterator[Frame]:
        import cv2

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # where the backend allows: no stale backlog
        reader = _Newest(cap, self.vid_stride)
        count, shown = 0, 0
        try:
            while (got := reader.next(count)) is not None:
                count, img = got
                shown += 1
                yield Frame(img, path, i, self.total, label, frame=shown, fps=fps)
        finally:
            reader.close()
