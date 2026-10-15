# Apache-2.0
"""The Detector facade: what it accepts, what it refuses, what predict returns."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from easydetect import Detector, ModelNotFoundError
from easydetect.model import _torch_device, increment_path

from .conftest import needs_ov, needs_torch


def test_a_pretrained_name_is_accepted_in_either_spelling():
    assert Detector("dfine_s").model_name == "dfine-s"
    assert Detector("D-FINE-L").variant == "l"


def test_an_unknown_name_lists_the_ones_that_exist():
    with pytest.raises(ModelNotFoundError, match="dfine-s, dfine-m, dfine-l"):
        Detector("easydetect-r101")


def test_a_missing_file_is_a_file_error_not_an_import_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        Detector(tmp_path / "best.pt")
    with pytest.raises(FileNotFoundError, match="model not found"):
        Detector(tmp_path / "model.xml")


def test_predict_without_a_source_says_what_is_missing():
    with pytest.raises(ValueError, match="needs a source"):
        Detector("dfine-s").predict(None)


def test_a_typo_in_a_predict_keyword_is_not_swallowed():
    with pytest.raises(TypeError, match="confidence"):
        Detector("dfine-s").predict("bus.jpg", confidence=0.5)


def test_run_directories_increment_instead_of_overwriting(tmp_path):
    first = increment_path(tmp_path / "predict")
    first.mkdir()
    second = increment_path(tmp_path / "predict")
    assert second.name == "predict2"
    second.mkdir()
    assert increment_path(tmp_path / "predict").name == "predict3"


@pytest.mark.parametrize(
    ("given", "expected"), [(0, "cuda:0"), ("0", "cuda:0"), ("cpu", "cpu"), (None, None)]
)
def test_device_shorthands_map_onto_torch_devices(given, expected):
    assert _torch_device(given) == expected


def test_val_without_weights_explains_how_to_get_them():
    model = Detector("dfine-s")
    model.model_name = "not-a-mirror-name"  # simulate a local model with no torch side
    with pytest.raises(RuntimeError, match="no torch weights"):
        model.val(data="data.yaml")


def test_export_rejects_formats_it_cannot_write():
    with pytest.raises(ValueError, match="openvino"):
        Detector("dfine-s").export(format="tflite")


class TestStreamGuard:
    """A generator nobody iterates runs nothing — say so instead of exiting quietly."""

    def _stream(self, monkeypatch):
        from easydetect.model import _Stream

        ran = []

        def generator():
            ran.append(True)
            yield "frame"

        return _Stream(generator(), "predict"), ran

    def test_dropping_the_stream_unused_explains_why_nothing_happened(self, monkeypatch, capsys):
        import gc

        stream, ran = self._stream(monkeypatch)
        del stream
        gc.collect()
        assert not ran
        err = capsys.readouterr().err
        assert "never iterated" in err and "for r in model.predict(" in err

    def test_iterating_it_stays_silent_and_yields_results(self, monkeypatch, capsys):
        import gc

        stream, ran = self._stream(monkeypatch)
        assert list(stream) == ["frame"] and ran
        del stream
        gc.collect()
        assert capsys.readouterr().err == ""

    def test_it_still_behaves_like_an_iterator(self):
        from easydetect.model import _Stream

        def generator():
            yield 1
            yield 2

        stream = _Stream(generator(), "track")
        assert next(stream) == 1
        assert list(stream) == [2]


class TestShowingFrames:
    """show=True has to keep a stream moving; a single image still waits."""

    @staticmethod
    def _fake_cv2(key):
        import types

        seen = {"windows": [], "delays": [], "closed": 0}

        def wait(delay):
            seen["delays"].append(delay)
            return key

        fake = types.SimpleNamespace(
            imshow=lambda window, img: seen["windows"].append(window),
            waitKey=wait,
            getWindowProperty=lambda window, prop: 1.0,
            destroyAllWindows=lambda: seen.__setitem__("closed", seen["closed"] + 1),
            WND_PROP_VISIBLE=0,
        )
        return fake, seen

    def _frame(self, kind):
        from easydetect.sources import Frame

        return Frame(np.zeros((4, 4, 3), np.uint8), "clip.mp4", 1, 1, kind, frame=1, frames=10)

    def _result(self, monkeypatch):
        from easydetect.results import Results

        monkeypatch.setattr(
            Results, "plot", lambda self, **kwargs: np.zeros((4, 4, 3), np.uint8)
        )
        return Results(np.zeros((4, 4, 3), np.uint8))

    def test_video_frames_do_not_wait_for_a_key(self, monkeypatch):
        from easydetect import model as model_module

        fake, seen = self._fake_cv2(key=ord("x"))
        monkeypatch.setitem(sys.modules, "cv2", fake)
        assert model_module._display(self._result(monkeypatch), self._frame("video")) is True
        assert seen["delays"] == [1] and seen["windows"] == ["clip.mp4"]

    def test_a_single_image_still_waits_for_a_key(self, monkeypatch):
        from easydetect import model as model_module

        fake, seen = self._fake_cv2(key=ord("x"))
        monkeypatch.setitem(sys.modules, "cv2", fake)
        model_module._display(self._result(monkeypatch), self._frame("image"))
        assert seen["delays"] == [0]

    @pytest.mark.parametrize("key", [ord("q"), 27])
    def test_q_and_esc_stop_the_stream(self, monkeypatch, key):
        from easydetect import model as model_module

        fake, _ = self._fake_cv2(key=key)
        monkeypatch.setitem(sys.modules, "cv2", fake)
        assert model_module._display(self._result(monkeypatch), self._frame("video")) is False


@needs_torch
@needs_ov
class TestAgainstARealModel:
    """The user-facing checklist, run against an actual compiled IR."""

    def test_predict_answers_with_a_list_of_results(self, tiny_ir, image):
        model = Detector(str(tiny_ir), device="CPU")
        results = model(image, conf=0.0, max_det=3)
        assert isinstance(results, list) and len(results) == 1
        result = results[0]
        assert len(result.boxes) == 3
        assert result.names == {0: "can", 1: "bottle"}
        assert result.orig_shape == (80, 80)
        assert result.path.endswith("bus.jpg")

    def test_stream_true_gives_a_generator(self, tiny_ir, image):
        model = Detector(str(tiny_ir), device="CPU", verbose=False)
        stream = model.predict(image, conf=0.0, max_det=1, stream=True)
        assert not isinstance(stream, list)
        assert [len(r.boxes) for r in stream] == [1]

    def test_an_ndarray_source_needs_no_file_at_all(self, tiny_ir):
        model = Detector(str(tiny_ir), device="CPU", verbose=False)
        results = model(np.zeros((50, 70, 3), np.uint8), conf=0.0, max_det=2)
        assert results[0].orig_shape == (50, 70)

    def test_a_folder_is_processed_in_order(self, tiny_ir, tmp_path):
        import cv2

        for name in ("1.jpg", "2.jpg"):
            cv2.imwrite(str(tmp_path / name), np.zeros((40, 40, 3), np.uint8))
        model = Detector(str(tiny_ir), device="CPU", verbose=False)
        results = model.predict(tmp_path, conf=0.0, max_det=1)
        assert [Path(r.path).name for r in results] == ["1.jpg", "2.jpg"]

    def test_save_writes_into_runs_detect_predict(self, tiny_ir, image, tmp_path):
        model = Detector(str(tiny_ir), device="CPU", verbose=False)
        model.predict(image, conf=0.0, max_det=2, save=True, project=str(tmp_path / "runs"))
        saved = tmp_path / "runs" / "detect" / "predict" / "bus.jpg"
        assert saved.exists()
        model.predict(image, conf=0.0, max_det=2, save=True, project=str(tmp_path / "runs"))
        assert (tmp_path / "runs" / "detect" / "predict2" / "bus.jpg").exists()

    def test_the_verbose_line_names_the_image_size_and_timing(self, tiny_ir, image, capsys):
        Detector(str(tiny_ir), device="CPU")(image, conf=0.0, max_det=2)
        line = capsys.readouterr().out.strip()
        assert line.startswith("image 1/1 ")
        assert "64x64" in line and line.endswith("ms")

    def test_track_puts_an_id_on_every_box(self, tiny_ir, tmp_path):
        import cv2

        clip = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 48))
        for _ in range(3):
            writer.write(np.zeros((48, 48, 3), np.uint8))
        writer.release()

        model = Detector(str(tiny_ir), device="CPU", verbose=False)
        real = model._ensure_predictor()
        sure = np.array([[5, 5, 20, 20, 0.9, 0], [25, 25, 40, 40, 0.8, 0],
                         [30, 5, 45, 15, 0.3, 0]], np.float32)   # the last one is unsure
        fake = type("Fake", (), {k: getattr(real, k) for k in
                                 ("backend", "device", "precision", "imgsz", "names")})()
        fake.__class__.__call__ = lambda self, img, **kw: (sure.copy(), {})
        model.predictor = fake                    # an untrained net finds nothing
        results = model.track(clip, conf=0.5, max_det=3)
        assert len(results) == 3
        assert [len(r.boxes) for r in results] == [2, 2, 2]     # the unsure box is not shown
        assert results[0].boxes.id.tolist() == results[-1].boxes.id.tolist()
        assert sorted(results[0].boxes.id.tolist()) == [1, 2]

    def test_track_takes_iou_as_the_duplicate_filter_and_match_iou_for_the_tracker(
            self, tiny_ir, tmp_path, monkeypatch):
        """As in Ultralytics, track(iou=) means what predict(iou=) means."""
        import cv2

        from easydetect import tracker as tracker_module

        clip = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 48))
        writer.write(np.zeros((48, 48, 3), np.uint8))
        writer.release()
        made = {}
        real = tracker_module.IoUTracker

        def spy(**kw):
            made.update(kw)
            return real(**kw)

        monkeypatch.setattr(tracker_module, "IoUTracker", spy)
        model = Detector(str(tiny_ir), device="CPU", verbose=False)
        model.track(clip, conf=0.0, iou=0.5, match_iou=0.2)
        assert made == {"iou": 0.2, "max_age": 30}
        seen = []
        predictor = model._ensure_predictor()
        original = type(predictor).__call__
        def watch(self, img, **kw):
            seen.append(kw["iou"])
            return original(self, img, **kw)

        monkeypatch.setattr(type(predictor), "__call__", watch)
        model.track(clip, conf=0.0, iou=0.5)
        assert seen == [0.5]
