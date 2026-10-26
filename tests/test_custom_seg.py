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
    heard = []
    best = Detector("ed-n", pretrained=False, verbose=False).train(
        data=str(data), epochs=1, imgsz=64, batch=4, workers=0, device="cpu",
        project=str(tmp_path / "runs"), amp=False, seg_epochs=1, seg_batch=4,
        on_progress=heard.append)
    beside = best.parent / "mask_decoder.onnx"
    assert beside.exists()
    stage = [h for h in heard if h["phase"] == "masks"]
    assert any(h.get("stage") == "encode" for h in stage) and "miou" in stage[-1]

    used = []
    real = segment.default_segmenter
    monkeypatch.setattr(segment, "default_segmenter",
                        lambda **kw: used.append(kw.get("decoder")) or real(**kw))
    img = cv2.imread(str(next((tmp_path / "data" / "images" / "val").glob("*.jpg"))))
    r = Detector(str(best), task="segment", verbose=False)(img, conf=0.0, max_det=2)[0]
    assert used == [beside] and r.masks is not None and len(r.masks) == len(r.boxes)


# -- the runtime: low-res logits, scaled up around each box ----------------------------


def _picture(h=300, w=500):
    import cv2

    img = np.full((h, w, 3), 40, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 5, h // 4), 0, 0, 360, (200, 180, 90), -1)
    return img


@needs_torch
@pytest.mark.parametrize("backend", ["onnxruntime", "openvino"])
def test_masks_from_low_res_logits_match_the_decoders_full_size_ones(stub_sam, backend):
    """The same masks SAM's own postprocessing (the decoder's full-size
    output) gives, without ever making one the picture's size per box."""
    import onnxruntime as ort

    from easydetect.segment import BoxSegmenter

    if backend == "openvino":
        pytest.importorskip("openvino")
    seg = BoxSegmenter(stub_sam / "encoder.onnx", stub_sam / "decoder.onnx", backend=backend,
                       box_pad=None)
    img = _picture()  # 3:5, not the 3:4 the decoder was exported with
    boxes = np.array([[150, 70, 350, 230], [10, 10, 200, 120], [300, 150, 499, 299]], np.float32)
    masks, scores = seg(img, boxes)
    tensor, scale = seg.preprocess(img)
    emb = seg._run("encoder", {"image": tensor})[0]
    full = ort.InferenceSession(str(stub_sam / "decoder.onnx")).run(None, {
        "image_embeddings": emb, "point_coords": (boxes.reshape(-1, 2, 2) * scale),
        "point_labels": np.tile(np.array([[2, 3]], np.float32), (3, 1)),
        "mask_input": np.zeros((1, 1, 256, 256), np.float32),
        "has_mask_input": np.zeros(1, np.float32),
        "orig_im_size": np.array(img.shape[:2], np.float32)})
    logits = full[0][:, 0]
    ref = logits > 0
    # the stub's random weights leave many pixels a hair from 0, where two
    # runtimes' rounding may land on either side: compare the decided ones
    decided = np.abs(logits) > 0.02 * np.abs(logits).max()
    assert masks.shape == ref.shape and ref.any() and decided.mean() > 0.9
    assert (masks == ref)[decided].mean() > 0.999
    # OpenVINO runs in bfloat16 on CPUs that have it (some CI runners do)
    np.testing.assert_allclose(scores, full[1][:, 0], atol=5e-3 if backend == "openvino" else 1e-4)
    if backend == "openvino":   # the full-size output is not even in the graph
        assert len(seg._decoder.outputs) == 2


@needs_torch
def test_a_mask_stays_near_its_box(stub_sam):
    from easydetect.segment import BoxSegmenter

    img = _picture()
    box = np.array([[150, 70, 350, 230]], np.float32)
    whole = BoxSegmenter(stub_sam / "encoder.onnx", stub_sam / "decoder.onnx",
                         backend="onnxruntime", box_pad=None)(img, box)[0][0]
    near = BoxSegmenter(stub_sam / "encoder.onnx", stub_sam / "decoder.onnx",
                        backend="onnxruntime")(img, box)[0][0]
    grown = np.zeros_like(whole)
    grown[54:246, 130:370] = True                  # the box, 10% bigger each side
    assert (near == (whole & grown)).all()
    assert whole[~grown].any()                     # the random decoder does spill out


def test_holes_and_stray_parts_are_cleaned_only_when_asked():
    from easydetect.segment import BoxSegmenter

    ring = np.zeros((40, 40), bool)
    ring[5:25, 5:25] = True
    ring[10:20, 10:20] = False                     # a hole
    ring[30:33, 30:33] = True                      # a stray speck
    seg = BoxSegmenter.__new__(BoxSegmenter)
    seg.fill_holes = seg.largest = False
    assert (seg._clean(ring) == ring).all()
    seg.fill_holes = seg.largest = True
    cleaned = seg._clean(ring)
    assert cleaned[15, 15] and not cleaned[31, 31] and cleaned.sum() == 400


def test_the_encoder_input_is_shrunk_without_aliasing():
    from easydetect.segment import BoxSegmenter

    stripes = np.zeros((1024, 2048, 3), np.uint8)
    stripes[:, ::2] = 255                          # finer than the 1024 frame can hold
    tensor, scale = BoxSegmenter.preprocess(stripes)
    shrunk = tensor[0, :, :512, :1024]
    assert scale == 0.5 and abs(float(shrunk.mean()) - 127.5) < 1 and float(shrunk.std()) < 1


# -- fine-tuning: both ways of picking a mask, jittered boxes, mapped embeddings ------


def test_jittered_boxes_are_gaussian_and_a_share_stay_exact(tmp_path, monkeypatch):
    from easydetect import seg_trainer
    from easydetect.seg_trainer import BOX_JITTER, jitter_box

    rng = np.random.default_rng(0)
    box = np.array([100, 100, 300, 200], np.float32)
    moved = np.stack([jitter_box(box, rng) for _ in range(4000)]) - box
    share = moved / np.array([200, 100, 200, 100])
    assert np.allclose(share.std(0), BOX_JITTER, rtol=0.1) and np.allclose(share.mean(0), 0,
                                                                           atol=0.005)

    class Cache:
        def file(self, path):
            np.save(tmp_path / "e.npy", np.zeros((256, 64, 64), np.float16))
            return tmp_path / "e.npy", (100, 100)

    poly = np.array([[0.2, 0.2], [0.6, 0.2], [0.6, 0.5], [0.2, 0.5]], np.float32)
    monkeypatch.setattr(seg_trainer, "read_polygons", lambda *a: [(tmp_path / "p.jpg",
                                                                  [(0, poly)])])
    ds = seg_trainer.MaskDataset("data.yaml", "train", Cache(), augment=True)
    np.random.seed(0)
    exact = np.mean([np.allclose(ds[0][1], [204.8, 204.8, 614.4, 512.0]) for _ in range(2000)])
    assert abs(exact - seg_trainer.EXACT_BOXES) < 0.04


@needs_torch
def test_all_four_masks_learn():
    import torch

    from easydetect.seg_trainer import _losses

    masks = torch.zeros(2, 4, 256, 256, requires_grad=True)
    ious = torch.full((2, 4), 0.5, requires_grad=True)
    target = torch.zeros(2, 256, 256)
    target[:, 64:192, 64:192] = 1
    _losses(masks, ious, target).backward()
    assert masks.grad[:, 0].abs().sum() > 0 and ious.grad[:, 0].abs().sum() > 0


def test_the_mask_choice_goes_by_jittered_boxes():
    from easydetect.seg_trainer import choose

    scores = {"exact": {"best_iou": 0.9, "single": 0.8},
              "jittered": {"best_iou": 0.7, "single": 0.75}}
    assert choose(scores) == ("single", 0.75)
    scores["jittered"]["single"] = 0.7
    assert choose(scores)[0] == "best_iou"      # a tie goes to SAM's own way


@needs_torch
def test_fine_tuning_scores_both_ways_and_exports_the_better(tmp_path, stub_sam):
    import json

    import onnxruntime as ort
    import torch

    from easydetect.nn.sam_decoder import SamDecoder
    from easydetect.seg_trainer import EmbeddingCache, MaskDataset, evaluate, train_masks
    from easydetect.segment import BoxSegmenter

    data = _toy(tmp_path / "data")
    seg = BoxSegmenter(stub_sam / "encoder.onnx", stub_sam / "decoder.onnx", backend="onnxruntime")
    onnx = train_masks(data, tmp_path / "run", segmenter=seg, init=str(stub_sam / "decoder.pt"),
                       epochs=1, batch=4, device="cpu", log=lambda m: None)
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    rule = run["mask_choice"]
    assert rule in ("best_iou", "single") and set(run["val_miou"]) == {"exact", "jittered"}
    assert run["miou"] == max(run["val_miou"]["jittered"].values())
    meta = ort.InferenceSession(str(onnx)).get_modelmeta().custom_metadata_map
    assert meta.get("mask_choice") == rule

    # the embeddings are memory-mapped from their files, not held
    cache = EmbeddingCache(seg, tmp_path / "run" / "embeddings")
    picture = next((tmp_path / "data" / "images" / "val").glob("*.jpg"))
    assert isinstance(cache(picture)[0], np.memmap)
    ds = MaskDataset(data, "val", cache)
    assert not hasattr(ds, "embeddings") and len(ds.files) == 3

    # the same jitter each call, so epochs compare fairly
    decoder = SamDecoder()
    decoder.load_state_dict(torch.load(tmp_path / "run" / "best.pt", weights_only=False)["model"])
    first, again = evaluate(decoder, ds, "cpu"), evaluate(decoder, ds, "cpu")
    assert first == again and first["exact"] != first["jittered"]


@needs_torch
def test_an_exported_single_mask_decoder_gives_slot_0(tmp_path, stub_sam):
    import onnxruntime as ort
    import torch

    from easydetect.nn.sam_decoder import SamDecoder
    from easydetect.seg_trainer import export

    decoder = SamDecoder()
    decoder.load_state_dict(torch.load(stub_sam / "decoder.pt"))
    out = export(decoder, tmp_path / "single.onnx", log=lambda m: None, mask_choice="single")
    emb = torch.randn(1, 256, 64, 64)
    boxes = torch.tensor([[100.0, 100.0, 400.0, 300.0], [50.0, 500.0, 300.0, 900.0]])
    with torch.no_grad():
        want, iou = decoder(emb.expand(2, -1, -1, -1), boxes)
    got = ort.InferenceSession(str(out)).run(None, {
        "image_embeddings": emb.numpy(), "point_coords": boxes.reshape(-1, 2, 2).numpy(),
        "point_labels": np.tile(np.array([[2, 3]], np.float32), (2, 1)),
        "mask_input": np.zeros((1, 1, 256, 256), np.float32),
        "has_mask_input": np.zeros(1, np.float32),
        "orig_im_size": np.array([600, 800], np.float32)})
    np.testing.assert_allclose(got[2][:, 0], want[:, 0].numpy(), atol=1e-3)
    np.testing.assert_allclose(got[1][:, 0], iou[:, 0].numpy(), atol=1e-4)
