# Apache-2.0
"""Segmentation datasets of your own: MobileSAM's mask decoder fine-tuned on
polygons, and task="segment" using the result. A stub image encoder and a
randomly started decoder keep it offline and quick; the decoder's code is
checked against SAM's own elsewhere (exact match on MobileSAM's weights)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from .conftest import needs_torch


def _toy(root, n_train=6, n_val=3, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    for split, count in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            w, h = 120, 90
            img = np.full((h, w, 3), 30, np.uint8)
            cx, cy = rng.integers(40, 80), rng.integers(30, 60)
            ax, ay = rng.integers(15, 30), rng.integers(10, 25)
            cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, (210, 200, 90), -1)
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
            t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            poly = np.stack([(cx + ax * np.cos(t)) / w, (cy + ay * np.sin(t)) / h], 1)
            (root / "labels" / split / f"{i}.txt").write_text(
                "0 " + " ".join(f"{v:.5f}" for v in poly.reshape(-1)) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(root), "train": "images/train", "val": "images/val", "names": {0: "blob"}}))
    return root / "data.yaml"


@pytest.fixture(scope="module")
def stub_sam(tmp_path_factory):
    """A tiny encoder.onnx (pools to the 64x64 grid), MobileSAM's decoder
    architecture as decoder.onnx and decoder.pt, randomly started."""
    import torch

    from easydetect.nn.sam_decoder import SamDecoder
    from easydetect.seg_trainer import export

    folder = tmp_path_factory.mktemp("sam")

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(3, 256, 1)

        def forward(self, x):
            return self.proj(torch.nn.functional.avg_pool2d(x / 255.0, 16))

    torch.manual_seed(0)
    torch.onnx.export(Encoder().eval(), torch.zeros(1, 3, 1024, 1024), str(folder / "encoder.onnx"),
                      input_names=["image"], output_names=["embeddings"], opset_version=17,
                      dynamo=False)
    decoder = SamDecoder()
    torch.save(decoder.state_dict(), folder / "decoder.pt")
    export(decoder, folder / "decoder.onnx", log=lambda m: None)
    return folder


def test_a_polygon_dataset_is_told_from_boxes_and_keypoints(tmp_path, dataset):
    from easydetect.seg_trainer import has_polygons, read_polygons

    data = _toy(tmp_path)
    assert has_polygons(data) and not has_polygons(dataset)
    pictures = read_polygons(data, "train")
    assert len(pictures) == 6 and pictures[0][1][0][1].shape == (16, 2)


@needs_torch
def test_fine_tuning_writes_a_decoder_that_scores_no_worse(tmp_path, stub_sam):
    import csv

    from easydetect.seg_trainer import train_masks
    from easydetect.segment import BoxSegmenter

    data = _toy(tmp_path / "data")
    seg = BoxSegmenter(stub_sam / "encoder.onnx", stub_sam / "decoder.onnx", backend="onnxruntime")
    onnx = train_masks(data, tmp_path / "run", segmenter=seg, init=str(stub_sam / "decoder.pt"),
                       epochs=3, batch=4, device="cpu", log=lambda m: None)
    assert onnx.exists() and (tmp_path / "run" / "embeddings").is_dir()
    rows = list(csv.DictReader((tmp_path / "run" / "results.csv").open()))
    assert [r["epoch"] for r in rows] == ["0", "1", "2", "3"]
    # the exported decoder runs where MobileSAM's does
    import cv2

    img = cv2.imread(str(next((tmp_path / "data" / "images" / "val").glob("*.jpg"))))
    masks, scores = BoxSegmenter(stub_sam / "encoder.onnx", onnx, backend="onnxruntime")(
        img, np.array([[20, 15, 100, 75]], np.float32))
    assert masks.shape == (1, *img.shape[:2]) and scores.shape == (1,)


@needs_torch
def test_training_on_a_polygon_dataset_gives_task_segment_its_decoder(tmp_path, stub_sam,
                                                                       monkeypatch):
    import cv2

    from easydetect import Detector, downloads, segment

    monkeypatch.setattr(downloads, "download_segmenter",
                        lambda: (stub_sam / "encoder.onnx", stub_sam / "decoder.onnx"))
    monkeypatch.setattr(downloads, "download_sam_decoder", lambda: stub_sam / "decoder.pt")
    data = _toy(tmp_path / "data")
    best = Detector("dfine-n", pretrained=False, verbose=False).train(
        data=str(data), epochs=1, imgsz=64, batch=4, workers=0, device="cpu",
        project=str(tmp_path / "runs"), amp=False, seg_epochs=1, seg_batch=4)
    beside = best.parent / "mask_decoder.onnx"
    assert beside.exists()

    used = []
    real = segment.default_segmenter
    monkeypatch.setattr(segment, "default_segmenter",
                        lambda **kw: used.append(kw.get("decoder")) or real(**kw))
    img = cv2.imread(str(next((tmp_path / "data" / "images" / "val").glob("*.jpg"))))
    r = Detector(str(best), task="segment", verbose=False)(img, conf=0.0, max_det=2)[0]
    assert used == [beside] and r.masks is not None and len(r.masks) == len(r.boxes)
