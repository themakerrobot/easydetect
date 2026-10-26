# Apache-2.0
"""Pictures in and out by any path, Hangul included.

OpenCV's ``imread``/``imwrite`` take the path as bytes in the system code
page; on Windows a path with characters outside it (a folder called
``바탕 화면``, a OneDrive in Korean) reads as nothing. Reading the file's
bytes with Python and decoding them in memory works for every path.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags: int = cv2.IMREAD_COLOR):
    """``cv2.imread`` for any path: the picture, or ``None`` when it cannot be read."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, img, params=None) -> bool:
    """``cv2.imwrite`` for any path; the format follows the file's extension."""
    ext = Path(str(path)).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def cv2_path(path) -> str:
    """A path OpenCV's video reader and writer can open. On Windows a path
    outside the code page becomes its short (8.3) form, which is plain ASCII;
    elsewhere, or when it is ASCII already, the path as it is. A file that
    does not exist yet (a video about to be written) keeps its name and gets
    its folder's short form."""
    text = str(path)
    import os
    import sys

    if sys.platform != "win32" or text.isascii():
        return text
    import ctypes

    def short(p: str) -> str:
        size = ctypes.windll.kernel32.GetShortPathNameW(p, None, 0)
        if not size:
            return p
        buf = ctypes.create_unicode_buffer(size)
        ctypes.windll.kernel32.GetShortPathNameW(p, buf, size)
        return buf.value

    if os.path.exists(text):
        return short(text)
    folder, name = os.path.split(text)
    return os.path.join(short(folder), name)
