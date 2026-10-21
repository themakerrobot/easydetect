# Apache-2.0
"""Training, validating and exporting for real — tiny, but the whole pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from easydetect import Detector
from easydetect.metrics import DetMetrics

from .conftest import needs_ov, needs_torch

pytestmark = [needs_torch]

TRAIN_KWARGS = dict(epochs=1, imgsz=64, batch=2, workers=0, device="cpu", val=False, amp=False)


@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    """One real (one-epoch, 64px) training run, reused by the tests below."""
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    runs = str(tmp_path_factory.mktemp("runs"))
    best = model.train(data=str(dataset), project=runs, **TRAIN_KWARGS)
    return model, best


def _blank_net(nc):
    from easydetect.nn import DFINENet

    return DFINENet("n", nc, pretrained_backbone=False)


def test_training_writes_last_and_best_with_everything_needed_to_resume(trained):
    import torch

    _, best = trained
    weights = best.parent
    assert best.exists() and (weights / "last.pt").exists()
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert ckpt["variant"] == "n" and ckpt["num_classes"] == 1
    assert ckpt["names"] == {0: "box"} and ckpt["imgsz"] == 64
    assert "optimizer" in ckpt and ckpt["epoch"] == 0


def test_the_model_reloads_its_own_checkpoint(trained):
    _, best = trained
    reloaded = Detector(str(best), verbose=False)
    assert reloaded.variant == "n" and reloaded.names == {0: "box"}
    assert "1 classes" in reloaded.info()


def test_val_reports_map_through_a_metrics_object(trained, dataset):
    model, _ = trained
    metrics = model.val(data=str(dataset), imgsz=64, batch=2, device="cpu", workers=0)
    assert isinstance(metrics, DetMetrics)
    assert 0.0 <= metrics.box.map50 <= 1.0 and 0.0 <= metrics.box.map <= 1.0
    assert metrics["map50"] == metrics.box.map50  # dict-style access still works


def test_resuming_continues_the_same_run_directory(dataset, tmp_path):
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    first = model.train(data=str(dataset), project=str(tmp_path), **TRAIN_KWARGS)

    resumed = Detector(str(first), verbose=False)
    again = resumed.train(
        data=str(dataset), project=str(tmp_path), resume=True, **{**TRAIN_KWARGS, "epochs": 2}
    )
    assert again.parent == first.parent  # same runs/train dir, not train2

    import torch

    assert torch.load(again, map_location="cpu", weights_only=False)["epoch"] == 1


def test_a_dataset_with_different_classes_re_heads_the_network(trained):
    from easydetect.model import transfer_weights

    model, _ = trained
    fresh = _blank_net(7)
    moved = transfer_weights(fresh, model.net.state_dict())
    assert moved > 100  # backbone/encoder came across
    assert fresh.decoder.dec_score_head[0].out_features == 7


def test_early_stopping_gives_up_after_patience_epochs(dataset, tmp_path, capsys):
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    model.train(
        data=str(dataset),
        project=str(tmp_path),
        epochs=6,
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        amp=False,
        patience=1,
        val=True,
    )
    assert "early stop" in capsys.readouterr().out


def test_training_falls_back_to_an_imagenet_start_when_the_mirror_is_unreachable(
    dataset, tmp_path, monkeypatch
):
    """No mirror, no crash: the run just starts from a fresh backbone."""
    from easydetect import downloads
    from easydetect.errors import ModelNotFoundError as NotFound

    monkeypatch.setattr(
        downloads,
        "download_checkpoint",
        lambda name: (_ for _ in ()).throw(NotFound("offline")),
    )
    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)
    model = Detector("dfine-n", verbose=False)
    best = model.train(data=str(dataset), project=str(tmp_path), **TRAIN_KWARGS)
    assert best.exists() and model.net.num_classes == 1


def test_export_writes_onnx_next_to_the_checkpoint(trained):
    model, best = trained
    onnx = model.export(format="onnx", imgsz=64)
    assert onnx.exists() and onnx.parent == best.parent
    assert (best.parent / "labels.txt").read_text().splitlines() == ["box"]


@needs_ov
def test_export_writes_an_ir_that_predicts(trained, tmp_path, image):
    model, _ = trained
    xml = model.export(format="openvino", imgsz=64, half=True, out_dir=tmp_path)
    assert xml.exists() and xml.with_suffix(".bin").exists()
    assert (tmp_path / "labels.txt").read_text().splitlines() == ["box"]

    results = Detector(str(xml), device="CPU", verbose=False)(image, conf=0.0, max_det=2)
    assert results[0].names == {0: "box"}
    assert results[0].boxes.xyxy.shape == (2, 4)


@needs_ov
def test_predicting_from_a_pt_exports_an_ir_behind_the_scenes(
    trained, image, tmp_path, monkeypatch
):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path / "cache"))
    model = Detector(str(trained[1]), device="CPU", verbose=False)
    results = model(image, conf=0.0, max_det=1)
    assert model.ir_path is not None and model.ir_path.exists()
    assert results[0].names == {0: "box"}


@needs_ov
def test_a_trained_model_still_decodes_its_own_probabilities_once(trained, tmp_path):
    """End to end: the IR emits probabilities, so conf filtering must be honest."""
    model, _ = trained
    xml = model.export(format="openvino", imgsz=64, out_dir=tmp_path / "ir")
    from easydetect.predictor import OVPredictor

    predictor = OVPredictor(xml, device="CPU")
    _, scores = predictor.infer(predictor.preprocess(np.zeros((64, 64, 3), np.uint8)))
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    det, _ = predictor(np.zeros((64, 64, 3), np.uint8), conf=float(scores.max()) + 1e-6)
    assert len(det) == 0


@needs_torch
def test_freezing_the_backbone_leaves_it_out_of_the_optimiser():
    from easydetect.nn import DFINENet
    from easydetect.trainer import freeze_modules

    net = DFINENet("n", num_classes=2, pretrained_backbone=False)
    frozen = freeze_modules(net, "backbone")
    assert frozen == [net.backbone]
    assert not any(p.requires_grad for p in net.backbone.parameters())
    # up and reg_scale are fixed by design; everything else in the decoder learns
    assert all(p.requires_grad for n, p in net.decoder.named_parameters()
               if not n.endswith(("up", "reg_scale")))


@needs_torch
def test_freezing_something_that_does_not_exist_is_an_error():
    from easydetect.nn import DFINENet
    from easydetect.trainer import freeze_modules

    net = DFINENet("n", num_classes=2, pretrained_backbone=False)
    with pytest.raises(ValueError, match="nothing called"):
        freeze_modules(net, "neck")


@needs_torch
def test_training_reports_every_epoch_to_a_callback_and_a_csv(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)
    rows = []
    model = Detector("dfine-n", verbose=False, pretrained=False)
    best = model.train(
        data=str(dataset), project=str(tmp_path), epochs=2, imgsz=64, batch=2, workers=0,
        device="cpu", val=False, amp=False, freeze="backbone", on_epoch_end=rows.append,
    )
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[0]["epochs"] == 2 and "loss" in rows[0]

    run = best.parent.parent
    lines = (run / "results.csv").read_text().splitlines()
    assert lines[0].startswith("epoch,loss") and len(lines) == 3

    import json

    summary = json.loads((run / "summary.json").read_text())
    assert summary["epochs_run"] == 2 and summary["names"] == {"0": "box"}

    # the record a model card is written from: how it was set up, and how it went
    setup = json.loads((run / "run.json").read_text())
    assert setup["variant"] == "n" and setup["start"] == {"kind": "scratch"}
    assert setup["freeze"] == "backbone" and 0 < setup["trainable_params"] < setup["params"]
    assert setup["optimizer"]["name"] == "AdamW" and setup["amp"] is False
    # no lr0: the rate came from the batch of 2, and the record says so
    assert setup["optimizer"]["lr"] == pytest.approx(1e-4 * (2 / 4) ** 0.5)
    assert setup["optimizer"]["lr_from"].startswith("batch 2:")
    train = setup["data"]["train"]
    assert train["images"] > 0 and sum(train["per_class"]) == train["boxes"] > 0
    assert setup["versions"]["torch"] and setup["device"] == "cpu" and setup["gpu"] is None
    assert summary["run"]["variant"] == "n" and summary["stopped_early"] is False
    assert summary["train_seconds"] > 0
    assert set(summary["final"]) == {"loss", "vfl", "l1", "giou", "fgl", "ddf"}


@needs_torch
def test_a_callback_that_stops_the_run_keeps_the_epoch_it_just_saw(dataset, tmp_path, monkeypatch):
    """Cancelling from on_epoch_end must not throw that epoch's weights away."""
    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)

    class Stop(Exception):
        pass

    seen = {}

    def stop_after_the_first(row):
        seen["epoch"] = row["epoch"]
        seen["saved"] = (Path(row["save_dir"]) / "weights" / "last.pt").exists()
        raise Stop()

    model = Detector("dfine-n", verbose=False, pretrained=False)
    with pytest.raises(Stop):
        model.train(
            data=str(dataset), project=str(tmp_path), epochs=3, imgsz=64, batch=2, workers=0,
            device="cpu", val=False, amp=False, freeze="backbone",
            on_epoch_end=stop_after_the_first,
        )
    assert seen == {"epoch": 1, "saved": True}


@needs_torch
def test_a_long_epoch_reports_where_it_is(dataset, tmp_path, monkeypatch):
    """Progress inside an epoch, then a word before validation — with loader
    workers running, which is how a GPU box trains."""
    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)
    heard = []
    model = Detector("dfine-n", verbose=False, pretrained=False)
    model.train(
        data=str(dataset), project=str(tmp_path), epochs=2, imgsz=64, batch=2, workers=2,
        device="cpu", amp=False, freeze="backbone", on_progress=heard.append,
    )
    train = [h for h in heard if h["phase"] == "train"]
    val = [h for h in heard if h["phase"] == "val"]
    assert {h["epoch"] for h in train} == {1, 2} and [h["epoch"] for h in val] == [1, 2]
    last = [h for h in train if h["epoch"] == 1][-1]
    assert last["step"] == last["steps"] and last["epochs"] == 2 and last["seconds"] >= 0


@needs_torch
def test_a_resumed_optimizer_lives_on_the_training_device(dataset, tmp_path, monkeypatch):
    """Resuming on a GPU failed on its first step with "cuda:0 and cpu": the
    optimizer state was loaded while the network was still on the CPU. The
    "meta" device stands in for the GPU this machine may not have."""
    from easydetect.nn import DFINENet
    from easydetect.trainer import Trainer

    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)
    Detector("dfine-n", verbose=False, pretrained=False).train(
        data=str(dataset), project=str(tmp_path), name="run", epochs=1, imgsz=64, batch=2,
        workers=0, device="cpu", val=False, amp=False,
    )
    net = DFINENet("n", num_classes=1, pretrained_backbone=False)
    trainer = Trainer(net, str(dataset), epochs=2, imgsz=64, batch=2, device="meta",
                      project=str(tmp_path), name="run", resume=True, amp=False, workers=0)
    assert trainer.start_epoch == 1
    for group in trainer.opt.param_groups:
        for param in group["params"]:
            assert param.device.type == "meta"
            for key, value in trainer.opt.state.get(param, {}).items():
                if key != "step" and hasattr(value, "device"):
                    assert value.device == param.device, key


@needs_torch
def test_pretrained_false_never_reaches_for_the_mirror(dataset, tmp_path, monkeypatch):
    from easydetect import downloads

    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)
    monkeypatch.setattr(
        downloads, "download_checkpoint", lambda name: pytest.fail("mirror should not be touched")
    )
    model = Detector("dfine-n", verbose=False, pretrained=False)
    assert model.train(
        data=str(dataset), project=str(tmp_path), epochs=1, imgsz=64, batch=2, workers=0,
        device="cpu", val=False, amp=False,
    ).exists()


def test_the_learning_rate_follows_the_batch_unless_it_is_given():
    from easydetect.trainer import auto_lr

    assert auto_lr(4) == pytest.approx(1e-4)
    assert auto_lr(16) == pytest.approx(2e-4)
    assert auto_lr(32) == pytest.approx(2.83e-4, rel=1e-3)
    assert auto_lr(0) == auto_lr(1)


@needs_torch
def test_a_learning_rate_that_is_given_is_used_as_it_is(dataset, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)
    model = Detector("dfine-n", verbose=False, pretrained=False)
    best = model.train(data=str(dataset), project=str(tmp_path), epochs=1, imgsz=64, batch=2,
                       workers=0, device="cpu", val=False, amp=False, lr0=3e-4,
                       warmup_epochs=0, weight_decay=0.0, lr_backbone_mult=0.25)
    opt = json.loads((best.parent.parent / "run.json").read_text())["optimizer"]
    assert opt["lr"] == pytest.approx(3e-4) and opt["lr_from"] == "set"
    assert opt["lr_backbone"] == pytest.approx(0.75e-4)
    assert opt["warmup_epochs"] == 0 and opt["weight_decay"] == 0.0


def test_a_lighter_decoder_exports_and_predicts(tmp_path, image):
    """layers / queries: fewer decoder layers and candidate boxes, same outputs otherwise."""
    import onnxruntime as ort

    from easydetect.exporter import export_onnx
    from easydetect.nn import DFINENet

    net = DFINENet("n", 2, pretrained_backbone=False)
    names = {0: "a", 1: "b"}
    full = export_onnx(net, names, imgsz=256, out_dir=tmp_path, fname="full", verbose=False)
    light = export_onnx(net, names, imgsz=256, out_dir=tmp_path, fname="light", verbose=False,
                        layers=1, queries=50)
    shapes = {}            # at 256 px dfine-n has 16x16 + 8x8 places to start a box: over 300
    for path in (full, light):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outs = session.run(None, {"images": np.zeros((1, 3, 256, 256), np.float32)})
        shapes[path.stem] = [o.shape for o in outs]
    assert shapes["full"] == [(1, 300, 4), (1, 300, 2)]
    assert shapes["light"] == [(1, 50, 4), (1, 50, 2)]
    assert net.decoder.num_queries == 300            # the caller's network is untouched

    with pytest.raises(ValueError, match="layers must be 1..3"):
        DFINENet("n", 2, pretrained_backbone=False).trim(layers=4)
    with pytest.raises(RuntimeError, match="before deploy"):
        DFINENet("n", 2, pretrained_backbone=False).deploy().trim(queries=10)


@needs_ov
def test_an_int8_export_calibrates_on_the_dataset_and_predicts(trained, dataset, tmp_path, image):
    pytest.importorskip("nncf")
    model, _ = trained
    xml = model.export(format="openvino", imgsz=64, out_dir=tmp_path, int8=True,
                       data=str(dataset), calib=4, verbose=False)
    assert xml.name.endswith("_int8.xml") and xml.with_suffix(".bin").exists()
    import openvino as ov

    ops = {op.get_type_name() for op in ov.Core().read_model(str(xml)).get_ops()}
    assert "FakeQuantize" in ops                        # 8-bit, not a float IR renamed
    results = Detector(str(xml), device="CPU", verbose=False)(image, conf=0.0, max_det=2)
    assert results[0].names == {0: "box"} and results[0].boxes.xyxy.shape == (2, 4)

    with pytest.raises(ValueError, match="calibrate"):
        model.export(format="openvino", imgsz=64, out_dir=tmp_path, int8=True)
    with pytest.raises(ValueError, match="OpenVINO export"):
        model.export(format="onnx", imgsz=64, out_dir=tmp_path, int8=True, data=str(dataset))


def test_loader_workers_are_not_forked_from_a_process_that_ran_openvino(dataset, monkeypatch):
    """A fork copies only one thread: after an OpenVINO model has run (its pool
    lives on), forked workers crashed the process — the lab serves predictions
    and trains in one process. Such a process starts them from a forkserver;
    any other forks, as a script without a __main__ guard needs."""
    import multiprocessing

    import torch

    from easydetect.data.dataset import DetDataset, worker_context
    from easydetect.trainer import Trainer

    class Stub:
        _loader = Trainer._loader
        batch, workers, device = 2, 2, torch.device("cpu")

    has_forkserver = "forkserver" in multiprocessing.get_all_start_methods()
    monkeypatch.setitem(sys.modules, "openvino", object())
    loader = Stub()._loader(DetDataset(str(dataset), "train", 64, augment=False))
    expected = "forkserver" if has_forkserver else multiprocessing.get_start_method()
    assert loader.multiprocessing_context.get_start_method() == expected

    monkeypatch.delitem(sys.modules, "openvino")
    assert worker_context() is None  # the platform's own default


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX start methods")
@pytest.mark.parametrize("how", ["piped", "unguarded file"])
def test_scripts_that_cannot_be_rerun_still_get_working_workers(dataset, tmp_path, how):
    """A forkserver worker re-runs the main script: one piped in has no file,
    one without a __main__ guard would train again in every worker. Neither
    loads OpenVINO here, so their workers are forked and work."""
    import signal

    # Earlier tests leave PyTorch's DataLoader SIGCHLD handler in this process,
    # and inside pytest (only there: a script that predicts with OpenVINO,
    # trains, and polls a subprocess every 0.2 s runs clean) the child's exit
    # signal then crashes the interpreter. The handler is not what is tested.
    previous = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    try:
        _run_script(dataset, tmp_path, how)
    finally:
        signal.signal(signal.SIGCHLD, previous)


def _run_script(dataset, tmp_path, how):
    import subprocess

    script = (
        "from easydetect.data.dataset import DetDataset, worker_context\n"
        "from torch.utils.data import DataLoader\n"
        f"ds = DetDataset({str(dataset)!r}, 'train', 64, augment=False)\n"
        "dl = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=DetDataset.collate,\n"
        "                multiprocessing_context=worker_context())\n"
        "print(worker_context(), sum(1 for _ in dl))\n"
    )
    if how == "piped":
        done = subprocess.run([sys.executable, "-"], input=script, capture_output=True,
                              text=True, timeout=300)
    else:
        path = tmp_path / "train_it.py"
        path.write_text(script)
        done = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                              timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ["None", "2"]


@needs_torch
def test_multiscale_mosaic_and_mixup_train_and_are_recorded(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)
    import torch.nn.functional as F

    sizes = []
    real = F.interpolate

    def spy(x, *args, **kwargs):
        sizes.append(kwargs.get("size"))
        return real(x, *args, **kwargs)

    monkeypatch.setattr(F, "interpolate", spy)
    monkeypatch.setattr("easydetect.trainer.random.choice", lambda seq: seq[0])  # the smallest
    model = Detector("dfine-n", verbose=False, pretrained=False)
    best = model.train(
        data=str(dataset), project=str(tmp_path), epochs=2, imgsz=64, batch=2, workers=0,
        device="cpu", val=False, amp=False, multiscale=True, mosaic=1.0, mixup=1.0,
    )
    import json

    setup = json.loads((best.parent.parent / "run.json").read_text())
    assert any("mosaic" in line for line in setup["augment"])
    assert any("mixup" in line for line in setup["augment"])
    assert any("random size, 32–64" in line for line in setup["augment"])
    assert (32, 32) in sizes  # the batches went through at the drawn size


def test_fewer_queries_train_and_stay_with_the_model(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, name, url: None)
    model = Detector("dfine-n", verbose=False, pretrained=False)
    best = model.train(data=str(dataset), project=str(tmp_path), epochs=1, imgsz=64, batch=2,
                       workers=0, device="cpu", val=False, amp=False, queries=40, denoising=10)
    again = Detector(str(best), verbose=False)
    again._ensure_net()
    assert again.net.decoder.num_queries == 40 and again.net.decoder.num_denoising == 10
    # at 160 px the encoder has more than 300 candidates: only the trim limits it
    onnx = again.export(format="onnx", imgsz=160, out_dir=tmp_path / "x", verbose=False)
    import onnxruntime as ort

    out = ort.InferenceSession(str(onnx)).run(None, {"images": np.zeros((1, 3, 160, 160),
                                                                       np.float32)})
    assert any(o.shape[1] == 40 for o in out if o.ndim == 3)


def test_plain_picture_epochs_keep_the_batch_norm_statistics(dataset, tmp_path):
    """The last tenth trains on plain pictures with the augmented epochs'
    batch-norm statistics: re-estimating them there lost most of the score."""
    import torch

    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    seen = []

    def stats(row):
        bn = [m for m in model.net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        seen.append(torch.cat([m.running_mean.clone() for m in bn]))

    model.train(data=str(dataset), project=str(tmp_path), on_epoch_end=stats,
                **{**TRAIN_KWARGS, "epochs": 4})
    assert not torch.equal(seen[1], seen[2])   # augmented epochs: they adapt
    assert torch.equal(seen[2], seen[3])       # the plain last epoch: kept


def test_with_the_backbone_frozen_the_plain_epochs_adapt_the_batch_norms(dataset, tmp_path):
    """Frozen backbone: the encoder and decoder batch norms follow the plain
    pictures in the last epochs (it scored higher that way, three seeds)."""
    import torch

    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    seen = []

    def stats(row):
        bn = [m for n, m in model.net.named_modules()
              if isinstance(m, torch.nn.BatchNorm2d) and not n.startswith("backbone")]
        seen.append(torch.cat([m.running_mean.clone() for m in bn]))

    model.train(data=str(dataset), project=str(tmp_path), on_epoch_end=stats, freeze="backbone",
                **{**TRAIN_KWARGS, "epochs": 4})
    assert not torch.equal(seen[2], seen[3])   # the plain last epoch: they adapt
