# Apache-2.0
"""Shared fixtures. Anything needing torch/openvino is built once per session."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the install extra
    HAS_TORCH = False

try:
    import openvino  # noqa: F401

    HAS_OV = True
except ImportError:  # pragma: no cover
    HAS_OV = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs the [train] extra")
needs_ov = pytest.mark.skipif(not HAS_OV, reason="needs openvino")


def draw(size: int = 80) -> np.ndarray:
    img = np.full((size, size, 3), 30, np.uint8)
    img[20:60, 20:50] = (40, 40, 200)
    return img


@pytest.fixture
def image(tmp_path: Path) -> Path:
    import cv2

    path = tmp_path / "bus.jpg"
    cv2.imwrite(str(path), draw())
    return path


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> Path:
    """A two-image dataset — enough to exercise train and val."""
    import cv2

    root = tmp_path_factory.mktemp("ds")
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), draw())
            (root / "labels" / split / f"{i}.txt").write_text("0 0.4375 0.5 0.375 0.5\n")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "box"},
            }
        )
    )
    return data


@pytest.fixture(scope="session")
def tiny_ir(tmp_path_factory) -> Path:
    """A real (untrained) 2-class IR at 64x64 — small enough to export in a test."""
    if not (HAS_TORCH and HAS_OV):
        pytest.skip("needs torch + openvino")
    from easydetect.exporter import export_openvino
    from easydetect.nn import EDNet

    out = tmp_path_factory.mktemp("ir")
    net = EDNet("n", num_classes=2, pretrained_backbone=False)
    return export_openvino(
        net, {0: "can", 1: "bottle"}, imgsz=64, out_dir=out, fname="tiny", verbose=False
    )
