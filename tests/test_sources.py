# Apache-2.0
"""Everything a source can be, and what the loader turns it into."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from easydetect.sources import SourceLoader, _expand, is_url


@pytest.fixture
def folder(tmp_path):
    for name in ("a.jpg", "b.png", "notes.md"):
        if name.endswith(".md"):
            (tmp_path / name).write_text("ignore me")
        else:
            cv2.imwrite(str(tmp_path / name), np.zeros((10, 10, 3), np.uint8))
    return tmp_path


def test_a_single_image_is_one_entry(folder):
    assert _expand(folder / "a.jpg") == [("image", folder / "a.jpg")]


def test_a_folder_collects_images_and_skips_everything_else(folder):
    kinds = _expand(folder)
    assert [k for k, _ in kinds] == ["image", "image"]
    assert [p.name for _, p in kinds] == ["a.jpg", "b.png"]


def test_a_glob_expands_in_sorted_order(folder):
    assert [p.name for _, p in _expand(str(folder / "*.jpg"))] == ["a.jpg"]


def test_a_txt_file_is_a_list_of_paths(folder):
    listing = folder / "list.txt"
    listing.write_text("a.jpg\nb.png\n")
    assert [p.name for _, p in _expand(listing)] == ["a.jpg", "b.png"]


def test_a_list_source_keeps_its_order(folder):
    entries = _expand([folder / "b.png", folder / "a.jpg"])
    assert [p.name for _, p in entries] == ["b.png", "a.jpg"]


def test_arrays_and_pil_images_go_straight_through():
    array = np.zeros((4, 4, 3), np.uint8)
    assert _expand(array)[0][0] == "array"

    pil = pytest.importorskip("PIL.Image")
    entries = _expand(pil.new("RGB", (4, 4), (255, 0, 0)))
    assert entries[0][0] == "array"
    assert entries[0][1].shape == (4, 4, 3)
    assert entries[0][1][0, 0].tolist() == [0, 0, 255]  # RGB red -> BGR


def test_webcam_indices_and_streams_are_recognised_without_opening_them():
    assert _expand(0) == [("stream", 0)]
    assert _expand("0") == [("stream", 0)]
    assert _expand("rtsp://cam/live") == [("stream", "rtsp://cam/live")]
    assert _expand("https://example.com/bus.jpg") == [("url-image", "https://example.com/bus.jpg")]
    assert _expand("https://example.com/clip.mp4")[0][0] == "stream"
    assert is_url("http://x/y.jpg") and not is_url("y.jpg")


def test_a_missing_source_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        _expand(tmp_path / "nope.jpg")
    with pytest.raises(FileNotFoundError, match="no files match"):
        _expand(str(tmp_path / "*.tif"))


def test_an_unsupported_file_type_is_rejected(tmp_path):
    odd = tmp_path / "weights.bin"
    odd.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="unsupported source"):
        _expand(odd)


def test_iterating_a_folder_yields_frames_with_numbered_prefixes(folder):
    loader = SourceLoader(folder)
    frames = list(loader)
    assert len(loader) == 2 and not loader.is_stream
    assert frames[0].prefix().startswith("image 1/2 ")
    assert frames[0].img.shape == (10, 10, 3)


def test_video_frames_carry_their_position_and_honour_vid_stride(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    for _ in range(6):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()

    frames = list(SourceLoader(path))
    assert len(frames) == 6
    assert frames[1].kind == "video" and frames[1].frame == 2
    assert "(frame 2/6)" in frames[1].prefix()
    assert len(list(SourceLoader(path, vid_stride=2))) == 3


def test_a_video_file_keeps_its_own_path_and_frame_rate(tmp_path):
    """A file is not a webcam: its name is what the log and the saved file use."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    for _ in range(4):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()

    frame = next(iter(SourceLoader(path)))
    assert frame.path == str(path)
    assert Path(frame.path).stem == "clip"
    assert frame.fps == pytest.approx(5.0, abs=0.1)
    # every other frame kept means playback at half the rate
    strided = next(iter(SourceLoader(path, vid_stride=2)))
    assert strided.fps == pytest.approx(2.5, abs=0.1)

