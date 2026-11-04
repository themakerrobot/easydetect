# Apache-2.0
"""Inference without PyTorch: what the robot (ONNX Runtime) and edge-lab
(OpenVINO) run, on stand-in models, offline, in a fresh interpreter each time.

Each check runs in a subprocess so that it starts from a clean ``sys.modules``:
it then fails the moment anything on the inference path imports torch, with
or without torch installed. The CI job "inference without torch" also runs
them where torch is not installed at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("onnx")          # builds the stand-ins (in [dev])

from .tiny_models import build  # noqa: E402

BACKENDS = ("openvino", "onnxruntime")

#: run first in every script: torch must not be imported, by anything
PRELUDE = """
import sys
def no_torch(where):
    loaded = sorted(m for m in ("torch", "torchvision") if m in sys.modules)
    assert not loaded, f"{loaded} imported by {where}"
"""

#: makes ``import torch`` fail as it does where torch is not installed
BLOCK_TORCH = """
import importlib.abc
class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "torchvision"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
sys.meta_path.insert(0, _NoTorch())
"""


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict[str, Path]:
    found = build(tmp_path_factory.mktemp("tiny"))
    # MobileSAM where default_segmenter looks first: the cache, no download
    cache = found["onnx"].parent / "cache"
    (cache / "mobile_sam").mkdir(parents=True)
    for name in ("encoder", "decoder"):
        (cache / "mobile_sam" / f"{name}.onnx").write_bytes(found[name].read_bytes())
    found["cache"] = cache
    return found


def run(script: str, models: dict[str, Path], block_torch: bool = False) -> str:
    env = dict(os.environ, EASYDETECT_HOME=str(models["cache"]),
               EASYDETECT_ASSETS_URL="http://127.0.0.1:9")    # nothing answers: offline
    code = PRELUDE + (BLOCK_TORCH if block_torch else "") + textwrap.dedent(script)
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def test_importing_easydetect_imports_no_torch(models):
    run("""
        import easydetect
        from easydetect import Detector, Results, Boxes
        from easydetect import pose, segment, plotting, predictor, results
        no_torch("import easydetect")
    """, models)


@pytest.mark.parametrize("backend", BACKENDS)
def test_detect_pose_segment_and_drawing_run_without_torch(models, backend):
    model = models["xml"] if backend == "openvino" else models["onnx"]
    out = run(f"""
        import numpy as np
        from easydetect import Detector
        from easydetect.plotting import draw_boxes, draw_keypoints, draw_masks
        from easydetect.pose import KeypointEstimator
        from easydetect.predictor import installed, onnx_metadata
        from easydetect.segment import BoxSegmenter, default_segmenter

        img = np.full((96, 128, 3), 90, np.uint8)
        assert installed({backend!r})
        meta = onnx_metadata({str(models["onnx"])!r})
        assert meta["easydetect.conf"] == "0.5" and "person" in meta["easydetect.names"]
        no_torch("onnx_metadata")

        r = Detector({str(model)!r}, backend={backend!r}, verbose=False)(img, conf=0.5)[0]
        assert r.names[0] == "person" and len(r.boxes) == 1, r.boxes.xyxy
        x1, y1, x2, y2 = r.boxes.xyxy[0]
        assert abs((x1 + x2) / 2 - 64) < 2 and abs((y1 + y2) / 2 - 48) < 2
        assert r.boxes.cls[0] == 0 and r.boxes.conf[0] > 0.9
        r.plot(); r.summary()
        no_torch("detect")

        pose = Detector({str(model)!r}, backend={backend!r}, task="pose", verbose=False)
        pose.pose_model = {str(models["pose"])!r}
        k = pose(img, conf=0.5)[0].keypoints
        assert k is not None and k.data.shape == (1, 17, 3)
        est = KeypointEstimator({str(models["pose"])!r}, backend={backend!r})
        xy, conf = est(img, r.boxes.xyxy)
        assert xy.shape == (1, 17, 2) and conf.shape == (1, 17)
        no_torch("pose")

        seg = Detector({str(model)!r}, backend={backend!r}, task="segment", verbose=False)
        m = seg(img, conf=0.5)[0].masks
        assert m is not None and m.data.shape == (1, 96, 128) and m.data.any()
        for sam in (BoxSegmenter({str(models["encoder"])!r}, {str(models["decoder"])!r},
                                 backend={backend!r}),
                    default_segmenter(backend={backend!r})):
            masks, scores = sam(img, r.boxes.xyxy)
            assert masks.shape == (1, 96, 128) and masks.any() and scores.shape == (1,)
        no_torch("segment")

        draw_boxes(img, r.boxes, r.names)
        draw_masks(img, masks, r.boxes.cls)
        draw_keypoints(img, k.data)
        no_torch("drawing")
        print("ran", {backend!r})
    """, models)
    assert f"ran {backend}" in out


def test_what_needs_torch_says_what_to_install(models):
    out = run(f"""
        from easydetect import Detector, EasyDetectError, TorchRequiredError
        for call in (lambda: Detector({str(models["onnx"])!r}, verbose=False).train(data="x"),
                     lambda: Detector({str(models["onnx"])!r}, verbose=False).val(data="x"),
                     lambda: Detector({str(models["onnx"])!r}, verbose=False).export()):
            try:
                call()
            except TorchRequiredError as exc:
                assert isinstance(exc, EasyDetectError) and isinstance(exc, ImportError)
                assert 'pip install "easydetect[train]"' in str(exc), exc
            else:
                raise AssertionError("no TorchRequiredError")
        print("explained")
    """, models, block_torch=True)
    assert "explained" in out
