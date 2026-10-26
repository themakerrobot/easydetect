# Apache-2.0
"""ONNX Runtime as the second runtime: same boxes as OpenVINO, picked when OpenVINO is absent."""

from __future__ import annotations

import numpy as np
import pytest

from easydetect import Detector
from easydetect import predictor as predictor_module

from .conftest import draw, needs_torch

ort = pytest.importorskip("onnxruntime")


@pytest.fixture
def no_openvino(monkeypatch):
    """This machine, as a Raspberry Pi with only onnxruntime installed would see it."""
    monkeypatch.setattr(predictor_module, "installed", lambda backend: backend != "openvino")


def test_both_runtimes_give_the_same_raw_outputs(tiny_ir):
    from easydetect.predictor import ORTPredictor, OVPredictor

    ov = OVPredictor(tiny_ir, device="CPU", precision="f32")
    rt = ORTPredictor(tiny_ir.with_suffix(".onnx"))
    assert rt.imgsz == ov.imgsz == 64 and rt.names == ov.names == {0: "can", 1: "bottle"}
    tensor = rt.preprocess(draw())
    (ob, os_), (rb, rs) = ov.infer(tensor), rt.infer(tensor)
    np.testing.assert_allclose(rb, ob, atol=1e-4)
    np.testing.assert_allclose(rs, os_, atol=1e-4)


def test_a_detector_on_onnxruntime(tiny_ir):
    model = Detector(str(tiny_ir.with_suffix(".onnx")), backend="onnxruntime", verbose=False)
    r = model(draw(), conf=0.0, max_det=5)[0]
    assert model.predictor.backend == "onnxruntime"
    assert r.boxes.xyxy.shape[1] == 4 and len(r.boxes) <= 5 and r.names[0] == "can"


def test_without_openvino_an_xml_runs_from_the_onnx_beside_it(tiny_ir, no_openvino):
    model = Detector(str(tiny_ir), verbose=False)
    model(draw(), conf=0.0)
    assert model.predictor.backend == "onnxruntime"
    assert model.ir_path.suffix == ".onnx"


def test_without_openvino_and_without_the_onnx_the_message_says_what_to_install(
        tiny_ir, tmp_path, no_openvino):
    lone = tmp_path / "lone.xml"
    lone.write_bytes(tiny_ir.read_bytes())
    with pytest.raises(ImportError, match="pip install openvino"):
        Detector(str(lone), verbose=False)(draw())


@needs_torch
def test_a_checkpoint_exports_onnx_when_the_runtime_is_onnxruntime(tmp_path, no_openvino,
                                                                    monkeypatch):
    import torch

    from easydetect import downloads
    from easydetect.nn import EDNet

    monkeypatch.setattr(downloads, "cache_dir", lambda: tmp_path / "cache")
    net = EDNet("n", num_classes=2, pretrained_backbone=False)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": net.state_dict(), "variant": "n", "num_classes": 2,
                "names": {0: "can", 1: "bottle"}, "imgsz": 64}, ckpt)
    model = Detector(str(ckpt), verbose=False)
    model(draw(), conf=0.0)
    assert model.predictor.backend == "onnxruntime" and model.ir_path.suffix == ".onnx"
    assert model.predictor.imgsz == 64


def test_the_choice_of_runtime_is_explained_when_it_cannot_work(tiny_ir):
    from easydetect.predictor import pick_backend

    with pytest.raises(ValueError, match="cannot read an OpenVINO .xml"):
        pick_backend("onnxruntime", tiny_ir)
    with pytest.raises(ValueError, match="OpenVINO device"):
        pick_backend("onnxruntime", tiny_ir.with_suffix(".onnx"), device="NPU")
    with pytest.raises(ValueError, match="backend must be one of"):
        pick_backend("tensorrt")
    assert pick_backend("onnxruntime", tiny_ir.with_suffix(".onnx")) == "onnxruntime"


def test_the_environment_can_choose_the_runtime(tiny_ir, monkeypatch):
    from easydetect.predictor import pick_backend

    monkeypatch.setenv("EASYDETECT_BACKEND", "onnxruntime")
    assert pick_backend(None, tiny_ir.with_suffix(".onnx")) == "onnxruntime"


def test_a_named_model_downloads_the_onnx_for_onnxruntime(monkeypatch, tmp_path, no_openvino):
    from easydetect import downloads

    fetched = []

    def fake_asset(name, filename, required=True):
        fetched.append(filename)
        return tmp_path / filename

    monkeypatch.setattr(downloads, "_asset", fake_asset)
    assert downloads.download_onnx("ed-s") == tmp_path / "ed-s.onnx"
    assert fetched == ["ed-s.onnx", "labels.txt"]


def test_an_onnx_carries_its_class_names_so_it_works_alone(tiny_ir, tmp_path):
    """One file from a Hugging Face page: no labels.txt beside it, or someone else's."""
    import shutil

    lone = tmp_path / "downloads" / "best.onnx"
    lone.parent.mkdir()
    shutil.copy(tiny_ir.with_suffix(".onnx"), lone)
    (lone.parent / "labels.txt").write_text("person\ncar\nbus\n")   # another model's
    for backend in ("openvino", "onnxruntime"):
        model = Detector(str(lone), backend=backend, verbose=False)
        assert model(draw(), conf=0.0)[0].names == {0: "can", 1: "bottle"}, backend


def test_the_names_are_read_without_the_onnx_package(tiny_ir, tmp_path):
    onnx = pytest.importorskip("onnx")
    from easydetect.predictor import ONNX_NAMES_KEY, onnx_metadata, read_names

    path = tiny_ir.with_suffix(".onnx")
    expected = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert onnx_metadata(path) == expected and ONNX_NAMES_KEY in expected

    # not an ONNX file at all: no names, and no crash
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"\xff\xff\xff\xff\x0f not a model")
    assert read_names(junk) == {}
