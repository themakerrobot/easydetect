# Apache-2.0
"""task="pose": the crop geometry, the decoding, the OKS score, the dataset, the
network's training step and export, and the plumbing into Detector."""

from __future__ import annotations

import json

import numpy as np
import pytest

from easydetect import Detector
from easydetect.pose import (
    FLIP,
    INPUT,
    KEYPOINT_NAMES,
    KeypointAP,
    apply,
    box_to_crop,
    crop_matrix,
    decode,
    invert,
    oks,
    person_rows,
)
from easydetect.results import Results

from . import pose_toy
from .conftest import draw, needs_torch


def test_a_box_becomes_a_padded_crop_of_the_network_shape():
    center, size = box_to_crop(np.array([[100, 50, 160, 250]]))
    np.testing.assert_allclose(center, [[130, 150]])
    assert size[0, 0] / size[0, 1] == pytest.approx(INPUT[1] / INPUT[0])
    assert size[0, 1] == pytest.approx(200 * 1.25)  # tall box: the height decides
    wide = box_to_crop(np.array([[0, 0, 300, 30]]))[1][0]
    assert wide[0] == pytest.approx(300 * 1.25)


@pytest.mark.parametrize("rotation", [0.0, 30.0, -75.0])
def test_the_crop_map_goes_there_and_back(rotation):
    m = crop_matrix(np.array([130.0, 150.0]), np.array([187.5, 250.0]), rotation)
    np.testing.assert_allclose(apply(m, np.array([130.0, 150.0])), [INPUT[1] / 2, INPUT[0] / 2])
    pts = np.random.default_rng(0).uniform(0, 300, (17, 2))
    np.testing.assert_allclose(apply(invert(m), apply(m, pts)), pts, atol=1e-9)


def test_decoding_reads_the_peak_and_how_sure_it_is():
    h, w = INPUT
    x = np.full((1, 2, w * 2), -10.0, np.float32)
    y = np.full((1, 2, h * 2), -10.0, np.float32)
    x[0, 0, 100], y[0, 0, 300] = 10.0, 10.0  # sharp
    x[0, 1] = 0.0
    y[0, 1] = 0.0  # flat: anywhere at all
    xy, conf = decode(x, y)
    np.testing.assert_allclose(xy[0, 0], [50.0, 150.0])
    assert conf[0, 0] == 1.0 and conf[0, 1] < 0.05


def test_flip_pairs_are_left_and_right():
    for i, j in enumerate(FLIP):
        assert FLIP[j] == i
        a, b = KEYPOINT_NAMES[i], KEYPOINT_NAMES[j]
        assert a == b or a.replace("left", "right") == b or a.replace("right", "left") == b


def test_oks_and_ap_follow_the_coco_definition():
    gt = np.concatenate([pose_toy.TEMPLATE * [60, 200] + [100, 50], np.full((17, 1), 2.0)], 1)
    assert oks(gt[:, :2], gt, 6000) == pytest.approx(1.0)
    assert 0.0 < oks(gt[:, :2] + 8, gt, 6000) < oks(gt[:, :2] + 2, gt, 6000) < 1.0

    perfect = KeypointAP()
    perfect.add(gt[None, :, :2], np.array([0.9]), gt[None], np.array([6000.0]))
    assert perfect.compute() == {"ap": 1.0, "ap50": 1.0, "ap75": 1.0}

    sloppy = KeypointAP()
    sloppy.add(gt[None, :, :2] + 6, np.array([0.9]), gt[None], np.array([6000.0]))
    result = sloppy.compute()
    assert result["ap50"] == 1.0 and result["ap"] < 1.0

    # a missed person halves recall; a person with no labelled keypoints is ignored
    unlabelled = gt.copy()
    unlabelled[:, 2] = 0
    half = KeypointAP()
    half.add(gt[None, :, :2], np.array([0.9]), np.stack([gt, gt + [300, 0, 0]]),
             np.array([6000.0, 6000.0]))
    assert half.compute()["ap"] == pytest.approx(0.5, abs=0.01)
    ignored = KeypointAP()
    ignored.add(gt[None, :, :2], np.array([0.9]), np.stack([gt, unlabelled + [300, 0, 0]]),
                np.array([6000.0, 6000.0]))
    assert ignored.compute()["ap"] == 1.0


def test_people_are_the_boxes_named_person():
    cls = np.array([0, 2, 0], np.float32)
    assert person_rows({0: "person", 2: "car"}, cls).tolist() == [True, False, True]
    assert person_rows({0: "worker"}, cls).all()  # no "person": every box


def test_a_result_draws_and_lists_keypoints():
    img = np.full((240, 320, 3), 60, np.uint8)
    kp = np.zeros((2, 17, 3), np.float32)
    kp[0, :, :2] = pose_toy.TEMPLATE * [60, 200] + [100, 20]
    kp[0, :, 2] = 0.9
    boxes = np.array([[100, 20, 160, 220, 0.9, 0], [200, 50, 300, 120, 0.8, 1]], np.float32)
    r = Results(img, names={0: "person", 1: "dog"}, boxes=boxes, keypoints=kp)
    drawn = r.plot()
    nose = kp[0, 0, :2].astype(int)
    assert (drawn[nose[1], nose[0]] != img[nose[1], nose[0]]).any()
    person, dog = r.summary()
    assert set(person["keypoints"]) == set(KEYPOINT_NAMES) and "keypoints" not in dog
    assert person["keypoints"]["nose"]["confidence"] == pytest.approx(0.9)
    assert Results(img, boxes=boxes).keypoints is None


class _FakeEstimator:
    def __init__(self):
        from easydetect.pose import COCO

        self.calls = []
        self.spec = COCO

    def __call__(self, img, xyxy):
        self.calls.append(np.asarray(xyxy).copy())
        n = len(xyxy)
        centers = (np.asarray(xyxy)[:, :2] + np.asarray(xyxy)[:, 2:]) / 2
        return np.repeat(centers[:, None], 17, 1).astype(np.float32), np.full((n, 17), 0.8,
                                                                            np.float32)


def test_pose_puts_keypoints_on_people_only(tiny_ir):
    model = Detector(str(tiny_ir), task="pose", verbose=False)
    model.pose = _FakeEstimator()
    r = model(draw(), conf=0.0, max_det=4)[0]
    assert len(r.keypoints) == len(r.boxes) == 4 and "pose" in r.speed
    people = person_rows(r.names, r.boxes.cls)
    assert (r.keypoints.conf[people] == 0.8).all() and (r.keypoints.conf[~people] == 0).all()
    if people.any():
        np.testing.assert_allclose(model.pose.calls[0], r.boxes.xyxy[people])


def test_the_command_line_takes_task_pose(tiny_ir, tmp_path, monkeypatch):
    import cv2

    from easydetect import cli, pose

    made = []
    monkeypatch.setattr(pose, "default_estimator",
                        lambda **kw: made.append(_FakeEstimator()) or made[-1])
    picture = tmp_path / "p.jpg"
    cv2.imwrite(str(picture), draw())
    code = cli.main(["predict", f"model={tiny_ir}", "task=pose", f"source={picture}",
                     "conf=0.0", "save=false", f"project={tmp_path}"])
    assert code == 0


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return pose_toy.make(tmp_path_factory.mktemp("pose"), train=12, val=4, empty=2)


def test_a_crop_carries_its_keypoints(toy):
    from easydetect.data.keypoints import KeypointDataset

    ds = KeypointDataset(toy, "train", augment=False)
    crop, xy, weight, i = ds[0]
    assert crop.shape == (3, *INPUT) and xy.shape == (17, 2) and weight.all()
    kpts = np.asarray(ds.items[0][1]["keypoints"]).reshape(17, 3)
    np.testing.assert_allclose(apply(invert(ds.matrix(0)), xy), kpts[:, :2], atol=1e-3)
    # the drawn joint is where the keypoint says (a white-ish dot on the figure)
    x, y = xy[9].round().astype(int)
    assert crop[:, y - 1:y + 2, x - 1:x + 2].max() > 150

    aug = KeypointDataset(toy, "train", augment=True)
    for k in range(10):
        c, p, wt, _ = aug[k % len(aug)]
        assert c.shape == (3, *INPUT)
        inside = (p[:, 0] >= 0) & (p[:, 0] < INPUT[1]) & (p[:, 1] >= 0) & (p[:, 1] < INPUT[0])
        assert not (wt.astype(bool) & ~inside).any()


@needs_torch
def test_a_training_step_lowers_the_loss_and_the_export_matches(toy, tmp_path, tiny_ir):
    import torch

    from easydetect.data.keypoints import KeypointDataset
    from easydetect.nn.posenet import PoseNet, simcc_loss

    torch.manual_seed(0)
    ds = KeypointDataset(toy, "train", augment=False)
    batch = [ds[i] for i in range(4)]
    crops = torch.from_numpy(np.stack([b[0] for b in batch]))
    xy = torch.from_numpy(np.stack([b[1] for b in batch]))
    weight = torch.from_numpy(np.stack([b[2] for b in batch]))
    net = PoseNet("s")
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(8):
        loss = simcc_loss(*net(crops), xy, weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train_pose", Path(__file__).resolve().parent.parent / "tools" / "train_pose.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ckpt = tmp_path / "best.pt"
    torch.save({"kind": "pose", "size": "s", "model": net.eval().state_dict()}, ckpt)
    onnx_path = tool.export(ckpt)
    assert onnx_path.name == "pose-s.onnx"

    from easydetect.pose import KeypointEstimator

    img = np.zeros((240, 320, 3), np.uint8)
    for backend in ("onnxruntime", "openvino"):
        est = KeypointEstimator(onnx_path, backend=backend)
        kp, conf = est(img, np.array([[10, 10, 100, 200], [150, 20, 300, 230]], np.float32))
        assert kp.shape == (2, 17, 2) and conf.shape == (2, 17)
        empty = est(img, np.zeros((0, 4), np.float32))
        assert empty[0].shape == (0, 17, 2)

    # the whole pipeline scores: detector boxes, then keypoints in each — on
    # every picture, the ones with nobody in them too (the first CPU run's
    # scoring failed on those)
    scores = tool.evaluate_pipeline(str(toy), onnx_path, str(tiny_ir), "CPU")
    assert set(scores) == {"ap", "ap50", "ap75"} and 0.0 <= scores["ap"] <= 1.0


@needs_torch
def test_frozen_stages_keep_their_weights_and_statistics():
    import torch

    from easydetect.nn.posenet import PoseNet

    net = PoseNet("s")
    net.freeze(2)
    net.train()
    stem_bn = next(m for m in net.backbone.stem.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = stem_bn.running_mean.clone()
    frozen_w = next(net.backbone.stages[1].parameters()).clone()
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    x, y = net(torch.rand(2, 3, *INPUT) * 255)
    (x.sum() + y.sum()).backward()
    opt.step()
    assert not stem_bn.training and torch.equal(stem_bn.running_mean, before)
    assert torch.equal(next(net.backbone.stages[1].parameters()), frozen_w)
    assert net.backbone.stages[2].training and net.lateral4.training


def test_half_body_skips_halves_too_small_to_see():
    from easydetect.data.keypoints import HALF_BODY_MIN_SIZE, KeypointDataset

    def person(spread):
        k = np.zeros((17, 3))
        k[:, 0] = 100 + np.linspace(0, spread, 17)
        k[:, 1] = 200 + np.linspace(0, spread, 17)
        k[:, 2] = 2
        return k

    # every labelled keypoint within a few pixels: blown up 150x, a blur
    assert KeypointDataset._half_body(None, person(6.0)) is None
    box = KeypointDataset._half_body(None, person(4 * HALF_BODY_MIN_SIZE))
    assert box is not None and (box[2:] - box[:2]).max() >= HALF_BODY_MIN_SIZE


def _tool():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train_pose", Path(__file__).resolve().parent.parent / "tools" / "train_pose.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_results_csv_from_an_older_run_gains_the_new_columns(tmp_path):
    import csv

    tool = _tool()
    path = tmp_path / "results.csv"
    path.write_text("epoch,loss,lr,seconds,ap,ap50,ap75\n1,3.4,0.001,3313.5,,,\n"
                    "5,2.5,0.0009,3205.9,0.4915,0.818,0.5198\n")
    tool.append_row(path, {"epoch": 6, "loss": 2.4, "lr": 0.0009, "seconds": 3000.0,
                           "ap": 0.53, "ap50": 0.84, "ap75": 0.58, "ap_ema": 0.51,
                           "ap_net": 0.53, "weights": "net"})
    rows = list(csv.DictReader(path.open()))
    assert [r["epoch"] for r in rows] == ["1", "5", "6"]
    assert rows[1]["ap"] == "0.4915" and rows[1]["ap_net"] == ""
    assert rows[2]["weights"] == "net" and rows[2]["ap_ema"] == "0.51"


@needs_torch
def test_training_scores_the_ema_and_the_network_and_keeps_the_better(toy, tmp_path):
    import csv

    import torch

    tool = _tool()
    out = tmp_path / "runs"
    assert tool.main(["--coco", str(toy), "--size", "s", "--init", "none", "--epochs", "2",
                      "--batch", "4", "--workers", "0", "--val-every", "1", "--warmup", "2",
                      "--freeze", "2", "--out", str(out), "--device", "cpu"]) == 0
    rows = list(csv.DictReader((out / "s" / "results.csv").open()))
    assert len(rows) == 2
    for r in rows:
        assert r["weights"] in ("ema", "net")
        assert float(r["ap"]) == max(float(r["ap_ema"]), float(r["ap_net"]))
    best = torch.load(out / "s" / "best.pt", map_location="cpu", weights_only=False)
    assert best["weights"] in ("ema", "net")
    assert json.loads((out / "s" / "run.json").read_text())["clip"] == 3.0
    assert (out / "s" / "pose-s.onnx").exists() and (out / "s" / "finished").exists()


def _peaked(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose softmax is the training target around crop positions ``xy``."""
    from easydetect.pose import SIGMA, SPLIT

    bx = np.arange(INPUT[1] * SPLIT)
    by = np.arange(INPUT[0] * SPLIT)
    x = -((bx - xy[..., :1] * SPLIT) ** 2) / (2 * SIGMA[1] ** 2)
    y = -((by - xy[..., 1:] * SPLIT) ** 2) / (2 * SIGMA[0] ** 2)
    return x.astype(np.float32), y.astype(np.float32)


def test_subpixel_decoding_lands_between_bins():
    from easydetect.pose import decode

    rng = np.random.default_rng(0)
    xy = rng.uniform(20, 170, (3, 17, 2))
    sub, _ = decode(*_peaked(xy))
    peak, _ = decode(*_peaked(xy), subpixel=False)
    assert np.abs(sub - xy).max() < 0.02  # the bins alone are off by up to 0.25 px
    assert np.abs(peak - xy).max() > 0.1


def test_a_mirrored_crop_reads_back_to_the_same_keypoints():
    """What the network says for a mirrored crop: keypoint k there is the
    mirrored position of keypoint FLIP[k] here (as training flips them)."""
    from easydetect.pose import FLIP, decode, unflip

    rng = np.random.default_rng(1)
    xy = rng.uniform(10, 180, (2, 17, 2))
    mirrored = xy[:, list(FLIP)].copy()
    mirrored[..., 0] = INPUT[1] - 1 - mirrored[..., 0]
    back, _ = decode(*unflip(*_peaked(mirrored)))
    assert np.abs(back - xy).max() < 0.02


@needs_torch
def test_the_estimator_can_average_each_crop_with_its_mirror(toy, tmp_path):
    import torch

    from easydetect.nn.posenet import PoseNet
    from easydetect.pose import KeypointEstimator

    torch.manual_seed(0)
    ckpt = tmp_path / "best.pt"
    torch.save({"kind": "pose", "size": "s", "model": PoseNet("s").eval().state_dict()}, ckpt)
    onnx_path = _tool().export(ckpt)
    img = np.zeros((240, 320, 3), np.uint8)
    img[40:200, 120:200] = 200
    box = np.array([[100, 30, 220, 210]], np.float32)
    plain = KeypointEstimator(onnx_path, backend="onnxruntime", flip=False)(img, box)
    both = KeypointEstimator(onnx_path, backend="onnxruntime", flip=True)(img, box)
    assert both[0].shape == plain[0].shape == (1, 17, 2)
    assert not np.allclose(both[0], plain[0])  # the mirror changed the reading

    scores = _tool().evaluate_pipeline(str(toy), onnx_path, "gt", "CPU", flip=True)
    assert set(scores) == {"ap", "ap50", "ap75"}


@needs_torch
def test_training_can_carry_on_from_a_keypoint_model(tmp_path, capsys):
    import torch

    from easydetect.nn.posenet import PoseNet

    torch.manual_seed(0)
    start = PoseNet("s").eval()
    ckpt = tmp_path / "best.pt"
    torch.save({"kind": "pose", "size": "s", "model": start.state_dict(), "epoch": 40,
                "ap": 0.59}, ckpt)
    net = _tool().build("s", str(ckpt))
    said = capsys.readouterr().out
    assert f"{len(start.state_dict())} of {len(start.state_dict())} weights from" in said
    for (k, a), b in zip(start.state_dict().items(), net.state_dict().values(), strict=True):
        assert torch.equal(a, b), k


def test_pose_flip_reaches_the_estimator(tiny_ir, monkeypatch):
    from easydetect import pose

    asked = []
    monkeypatch.setattr(pose, "default_estimator",
                        lambda **kw: asked.append(kw["flip"]) or _FakeEstimator())
    for flip in (True, False):
        model = Detector(str(tiny_ir), task="pose", verbose=False)
        assert model.pose_flip is True
        model.pose_flip = flip
        model(draw(), conf=0.0)
    assert asked == [True, False]


def test_pose_model_picks_the_size_or_a_file(tiny_ir, tmp_path, monkeypatch):
    from easydetect import pose

    sizes = []
    monkeypatch.setattr(pose, "default_estimator",
                        lambda **kw: sizes.append(kw["size"]) or _FakeEstimator())
    for size in ("s", "m"):
        model = Detector(str(tiny_ir), task="pose", verbose=False)
        model.pose_model = size
        model(draw(), conf=0.0)
    assert sizes == ["s", "m"]

    made = []
    monkeypatch.setattr(pose, "KeypointEstimator",
                        lambda path, **kw: made.append((path, kw["flip"])) or _FakeEstimator())
    own = tmp_path / "mine.onnx"
    own.write_bytes(b"onnx")
    model = Detector(str(tiny_ir), task="pose", verbose=False)
    model.pose_model, model.pose_flip = str(own), False
    model(draw(), conf=0.0)
    assert made == [(str(own), False)]

    model = Detector(str(tiny_ir), task="pose", verbose=False)
    model.pose_model = "xl"
    with pytest.raises(ValueError, match="pose_model"):
        model(draw(), conf=0.0)


def test_the_command_line_takes_pose_model_and_pose_flip(tiny_ir, tmp_path, monkeypatch):
    import cv2

    from easydetect import cli, pose

    asked = []
    monkeypatch.setattr(pose, "default_estimator",
                        lambda **kw: asked.append((kw["size"], kw["flip"])) or _FakeEstimator())
    picture = tmp_path / "p.jpg"
    cv2.imwrite(str(picture), draw())
    assert cli.main(["predict", f"model={tiny_ir}", "task=pose", f"source={picture}",
                     "pose_model=m", "pose_flip=false", "conf=0.0", "save=false",
                     f"project={tmp_path}"]) == 0
    assert asked == [("m", False)]


@needs_torch
@pytest.mark.parametrize("size, detector", [("s", "s"), ("m", "m"), ("l", "l")])
def test_each_keypoint_size_takes_its_detectors_whole_backbone(size, detector):
    """dfine-l's backbone has no lab layers where s and m have them: every
    backbone tensor of the detector of that size must carry over."""
    from easydetect.nn import DFINENet
    from easydetect.nn.posenet import SIZE_CFG, PoseNet
    from easydetect.pose import SIZES

    assert tuple(SIZE_CFG) == SIZES
    det = {k: v for k, v in DFINENet(detector, 80, pretrained_backbone=False).state_dict().items()
           if k.startswith("backbone.")}
    net = PoseNet(size)
    assert net.load_detector_backbone(det) == len(det)
    # all of it: what the detector lacks is only BatchNorm's step counters
    # (dfine-l freezes its norms, which keeps none)
    missing = set(net.backbone.state_dict()) - {k[len("backbone."):] for k in det}
    assert all(k.endswith("num_batches_tracked") for k in missing)
