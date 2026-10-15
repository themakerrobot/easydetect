# Apache-2.0
"""How frames are read: files a little ahead of the model, live sources
newest-frame-only, and frames vid_stride skips grabbed but never decoded."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from easydetect.sources import SourceLoader


class _FakeCapture:
    """A cv2.VideoCapture stand-in: ``frames`` frames whose pixels hold
    their number, ``live`` ones coming every ``interval`` seconds."""

    made: list[_FakeCapture] = []

    def __init__(self, source, frames=12, live=False, interval=0.0):
        self.source, self.frames, self.live, self.interval = source, frames, live, interval
        self.n, self.reads, self.grabs, self.released = 0, 0, 0, False
        self.props = {}
        _FakeCapture.made.append(self)

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 0 if self.live else self.frames
        return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def grab(self):
        if self.n >= self.frames:
            return False
        time.sleep(self.interval)
        self.n += 1
        self.grabs += 1
        return True

    def read(self):
        if not self.grab():
            return False, None
        self.grabs -= 1
        self.reads += 1
        return True, np.full((4, 4, 3), self.n, np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    _FakeCapture.made.clear()
    settings = {}
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: _FakeCapture(src, **settings))
    return settings


def test_frames_vid_stride_skips_are_grabbed_not_decoded(tmp_path, fake_capture):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    frames = list(SourceLoader(clip, vid_stride=3))
    cap = _FakeCapture.made[0]
    assert [int(f.img[0, 0, 0]) for f in frames] == [1, 4, 7, 10]
    assert [f.frame for f in frames] == [1, 2, 3, 4] and frames[0].frames == 4
    assert cap.reads == 4 and cap.grabs == 8 and cap.released


def test_a_live_source_hands_over_its_newest_frame(fake_capture):
    fake_capture.update(frames=40, live=True, interval=0.005)
    seen = []
    for frame in SourceLoader(0):
        seen.append(int(frame.img[0, 0, 0]))
        time.sleep(0.03)  # a model slower than the camera
    cap = _FakeCapture.made[0]
    assert cap.props.get(cv2.CAP_PROP_BUFFERSIZE) == 1 and cap.released
    assert seen == sorted(set(seen)) and len(seen) < 25   # frames skipped, never repeated
    assert seen[-1] == 40                                 # and the last one not lost
    assert frame.kind == "stream" and frame.path == "webcam0" and frame.frame == len(seen)


def test_an_rtsp_stream_is_live_even_with_a_frame_count(fake_capture):
    fake_capture.update(frames=5)
    frames = list(SourceLoader("rtsp://cam/live"))
    assert _FakeCapture.made[0].props.get(cv2.CAP_PROP_BUFFERSIZE) == 1
    assert [int(f.img[0, 0, 0]) for f in frames][-1] == 5


def test_a_live_reader_stops_when_the_loop_does(fake_capture):
    fake_capture.update(frames=10_000, live=True, interval=0.001)
    before = threading.active_count()
    for _ in SourceLoader(0):
        break
    deadline = time.time() + 3
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() == before and _FakeCapture.made[0].released


@pytest.fixture
def pictures(tmp_path):
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((6, 6, 3), i * 10, np.uint8))
    return tmp_path


def test_pictures_are_read_ahead_of_the_model(pictures, monkeypatch):
    read = []
    real = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda p, *a: read.append(p) or real(p, *a))
    frames = iter(SourceLoader(pictures))
    first = next(frames)
    deadline = time.time() + 3
    while len(read) < 3 and time.time() < deadline:   # the reader runs while we "infer"
        time.sleep(0.01)
    time.sleep(0.2)
    # two queued, one more in hand waiting for room; not the whole folder
    assert int(first.img[0, 0, 0]) == 0 and 3 <= len(read) <= 4
    assert [int(f.img[0, 0, 0]) for f in frames] == [10, 20, 30, 40]


def test_a_bad_picture_fails_where_it_comes_in_the_order(pictures):
    (pictures / "2.png").write_bytes(b"not a picture")
    frames = iter(SourceLoader(pictures))
    assert [int(next(frames).img[0, 0, 0]) for _ in range(2)] == [0, 10]
    with pytest.raises(ValueError, match="could not read image"):
        next(frames)


def test_stopping_early_stops_the_reader(pictures):
    before = threading.active_count()
    for _ in SourceLoader(pictures):
        break
    deadline = time.time() + 3
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() == before
