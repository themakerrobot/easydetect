# Apache-2.0
"""Training and inference edge cases: bad data, odd numbers, unsafe files."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import yaml

from .conftest import needs_torch


class Payload:
    """Stands for anything a pickle could build — and so run."""


def _copy(dataset, tmp_path):
    root = tmp_path / "ds"
    shutil.copytree(dataset.parent, root)
    cfg = yaml.safe_load(dataset.read_text())
    (root / "data.yaml").write_text(yaml.safe_dump(dict(cfg, path=str(root))))
    return root, root / "data.yaml"


@needs_torch
def test_pictures_leave_the_loader_as_uint8(dataset):
    import torch

    from easydetect.data.dataset import DetDataset

    ds = DetDataset(dataset, "val", 64, augment=False)
    img, _ = ds[0]
    assert img.dtype == torch.uint8
    as_float = DetDataset.as_float(img[None])
    assert as_float.dtype == torch.float32 and float(as_float.max()) <= 1.0


@needs_torch
def test_an_unknown_class_id_is_dropped_with_a_warning(dataset, tmp_path, capsys):
    from easydetect.data.dataset import DetDataset

    root, data = _copy(dataset, tmp_path)
    (root / "labels" / "train" / "0.txt").write_text("0 .5 .5 .2 .2\n7 .5 .5 .2 .2\n")
    ds = DetDataset(data, "train", 64, augment=False)
    labels = ds._load_labels(ds.files[0])
    assert labels[:, 0].tolist() == [0]
    assert "class" in capsys.readouterr().out


@needs_torch
def test_an_unreadable_picture_is_skipped(dataset, tmp_path, capsys):
    from easydetect.data.dataset import DetDataset

    root, data = _copy(dataset, tmp_path)
    (root / "images" / "train" / "0.jpg").write_bytes(b"broken")
    ds = DetDataset(data, "train", 64, augment=False)
    for i in range(len(ds)):
        img, _ = ds[i]
        assert img.shape[-2:] == (64, 64)
    assert "cannot read" in capsys.readouterr().out


def test_val_boxes_are_clipped_into_the_picture():
    from easydetect.data.dataset import _clean

    labels = np.array([[0, 0.95, 0.5, 0.3, 0.2],    # hangs off the right edge
                       [0, 1.5, 0.5, 0.2, 0.2]],    # wholly outside
                      np.float32)
    out = _clean(labels)
    assert len(out) == 1
    x1, x2 = out[0, 1] - out[0, 3] / 2, out[0, 1] + out[0, 3] / 2
    assert x1 == pytest.approx(0.8) and x2 == pytest.approx(1.0)


@needs_torch
@pytest.mark.parametrize("steps, epochs, expected", [
    (10, 100, 5.0),      # 1000 steps in all: 5% of them, 50 steps = 5 epochs
    (100, 100, 5.0),     # 10000 steps: 500 steps = 5 epochs
    (1000, 100, 1.0),    # a big dataset: the configured epoch is longer than 500 steps
])
def test_warmup_scales_with_the_run(steps, epochs, expected):
    from easydetect.trainer import Trainer

    fake = type("T", (), {"epochs": epochs, "warmup_epochs": 1.0})()
    assert Trainer.warmup_for(fake, steps) == pytest.approx(expected)


@needs_torch
def test_top_pairs_can_give_one_query_two_classes():
    import torch

    from easydetect.validator import top_pairs

    logits = torch.full((3, 2), -9.0)
    logits[1, 0], logits[1, 1] = 3.0, 2.0      # query 1 is sure of both classes
    boxes = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b, s, c = top_pairs(logits, boxes, k=2)
    assert c.tolist() == [0, 1]
    assert torch.equal(b[0], boxes[1]) and torch.equal(b[1], boxes[1])
    assert s[0] > s[1]


@needs_torch
def test_float32_casts_nested_outputs():
    import torch

    from easydetect.trainer import _float32

    out = _float32({"a": torch.ones(2, dtype=torch.float16),
                    "b": [torch.ones(1, dtype=torch.bfloat16), torch.ones(1, dtype=torch.long)]})
    assert out["a"].dtype == torch.float32 and out["b"][0].dtype == torch.float32
    assert out["b"][1].dtype == torch.long


@needs_torch
def test_ema_moves_toward_the_weights():
    import torch

    from easydetect.trainer import ModelEMA

    net = torch.nn.BatchNorm1d(3)
    ema = ModelEMA(net, decay=0.5, warmups=1e-9)
    with torch.no_grad():
        net.weight.fill_(3.0)
    net.num_batches_tracked.fill_(7)
    ema.update(net)
    assert torch.allclose(ema.module.weight, torch.full((3,), 2.0))
    assert int(ema.module.num_batches_tracked) == 7


@pytest.mark.parametrize("device, available, expected", [
    ("AUTO", ["CPUExecutionProvider"], ["CPUExecutionProvider"]),
    ("AUTO", ["CUDAExecutionProvider", "CPUExecutionProvider"],
     ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("AUTO", ["DmlExecutionProvider", "CPUExecutionProvider"],
     ["DmlExecutionProvider", "CPUExecutionProvider"]),
    ("CPU", ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]),
    ("TENSORRT", ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
     ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]),
])
def test_onnx_runtime_providers(device, available, expected):
    from easydetect.predictor import ort_providers

    assert ort_providers(device, available) == expected


def test_a_class_filter_scores_each_query_by_the_wanted_classes():
    from .test_predictor import decode

    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], np.float32)
    scores = np.array([[0.6, 0.7]], np.float32)       # sure of 0, surer of 1
    out = decode(boxes, scores, conf=0.5, classes=[0])
    assert len(out) == 1 and out[0, 5] == 0 and out[0, 4] == pytest.approx(0.6)


@needs_torch
def test_a_checkpoint_with_code_in_it_is_refused(tmp_path, monkeypatch):
    import torch

    from easydetect.utils.checkpoint import load_checkpoint

    path = tmp_path / "evil.pt"
    torch.save({"x": Payload()}, path)
    monkeypatch.delenv("EASYDETECT_TRUST_CHECKPOINTS", raising=False)
    with pytest.raises(ValueError):
        load_checkpoint(path)
    monkeypatch.setenv("EASYDETECT_TRUST_CHECKPOINTS", "1")
    assert isinstance(load_checkpoint(path)["x"], Payload)
    torch.save({"w": torch.ones(2)}, path)
    monkeypatch.delenv("EASYDETECT_TRUST_CHECKPOINTS")
    assert torch.equal(load_checkpoint(path)["w"], torch.ones(2))


@needs_torch
def test_data_warnings_name_the_usual_mistakes(dataset):
    from easydetect.trainer import data_warnings

    data = {"train": {"images": 30, "boxes": 120, "per_class": [118, 2, 0]},
            "val": {"images": 2}}
    files = ["a.jpg", "b.jpg"]
    out = " | ".join(data_warnings(data, ["cat", "dog", "fox"], files, files))
    assert "only 30 training pictures" in out
    assert "fox" in out and "dog" in out and "uneven" in out
    assert "same pictures as train" in out
    assert any("no val split" in w for w in data_warnings({"train": {"images": 500}}, []))


@needs_torch
def test_the_plain_epochs_restart_from_the_best_weights(tmp_path):
    import torch

    from easydetect.trainer import EMA_RESTART_STEP, ModelEMA, Trainer

    (tmp_path / "weights").mkdir()
    best, now = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    torch.save({"model": best.state_dict(), "raw": best.state_dict(), "ema_updates": 5,
                "epoch": 3, "best": 0.5}, tmp_path / "weights" / "best.pt")
    stub = type("T", (), {"save_dir": tmp_path, "clean_from": 9, "ema": ModelEMA(now)})()
    Trainer._restart_from_best(stub, now)
    assert torch.equal(now.weight, best.weight)
    assert torch.equal(stub.ema.module.weight, best.weight)
    assert stub.ema.updates == 5 and stub.ema.decay == pytest.approx(0.9999 - EMA_RESTART_STEP)

    stub.clean_from = 4                  # the best was the last epoch: nothing reloaded
    other = torch.nn.Linear(2, 2)
    before = other.weight.clone()
    Trainer._restart_from_best(stub, other)
    assert torch.equal(other.weight, before)


@needs_torch
def test_matching_all_layers_at_once_gives_the_same_loss():
    import torch

    from easydetect.nn import EDNet
    from easydetect.trainer import _float32
    from easydetect.utils.loss import DetectionLoss

    torch.manual_seed(0)
    net = EDNet("n", 3, pretrained_backbone=False).train()
    targets = [{"labels": torch.tensor([0, 2]),
                "boxes": torch.tensor([[.3, .3, .2, .2], [.6, .6, .3, .2]])},
               {"labels": torch.tensor([1]), "boxes": torch.tensor([[.5, .5, .4, .4]])}]
    out = _float32(net(torch.rand(2, 3, 128, 128), targets))
    crit = DetectionLoss(3, reg_max=net.decoder.reg_max)
    together, _ = crit(out, targets)
    matcher, calls = crit.criterion.matcher, []

    def one_by_one(layers, t):
        calls.append(len(layers))
        return [matcher(o, t)["indices"] for o in layers]

    matcher.match_many = one_by_one
    apart, _ = crit(out, targets)
    assert calls and calls[0] > 2                    # the last layer, the others, the encoder
    assert torch.allclose(together, apart)


def test_pictures_in_a_hangul_folder_read_and_write(tmp_path):
    """Windows' OpenCV cannot open such a path itself (바탕 화면, a Korean OneDrive)."""
    import numpy as np

    from easydetect.utils.imageio import cv2_path, imread, imwrite

    folder = tmp_path / "바탕 화면" / "사진"
    folder.mkdir(parents=True)
    img = np.full((20, 30, 3), 120, np.uint8)
    assert imwrite(folder / "고양이.png", img)
    back = imread(folder / "고양이.png")
    assert back is not None and back.shape == img.shape and (back == img).all()
    assert imread(folder / "없음.jpg") is None
    assert imread(folder / "고양이.png").dtype == np.uint8
    assert cv2_path(folder / "a.mp4").endswith("a.mp4")


@needs_torch
def test_a_dataset_in_a_hangul_folder_trains(dataset, tmp_path):
    import shutil

    import yaml

    from easydetect.data.dataset import DetDataset

    root = tmp_path / "바탕 화면" / "데이터"
    shutil.copytree(dataset.parent, root)
    cfg = yaml.safe_load(dataset.read_text())
    (root / "data.yaml").write_text(yaml.safe_dump(dict(cfg, path=str(root))), encoding="utf-8")
    ds = DetDataset(root / "data.yaml", "train", 64, augment=True)
    img, labels = ds[0]
    assert img.shape[-2:] == (64, 64)
