# Apache-2.0
"""Resuming, patience and re-heading: the cases that went wrong quietly."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
import yaml

from easydetect import Detector
from easydetect.trainer import Trainer

from .conftest import needs_torch

pytestmark = [needs_torch]

FAST = dict(imgsz=64, batch=2, workers=0, device="cpu", amp=False)


def _blank_net(nc):
    from easydetect.nn import DFINENet

    return DFINENet("n", nc, pretrained_backbone=False)


def _scripted_val(monkeypatch, scores):
    """Validation that answers ``scores`` in turn (then the last one forever)."""
    calls = []

    def fake(net, data, **kw):
        calls.append(1)
        return {"map": scores[min(len(calls), len(scores)) - 1]}

    monkeypatch.setattr("easydetect.validator.validate_torch", fake)
    return calls


def _epochs_in(run: Path) -> int:
    with (run / "results.csv").open(newline="", encoding="utf-8") as f:
        return len(list(csv.DictReader(f)))


def test_the_plain_epochs_get_their_full_patience_after_the_restart(
        dataset, tmp_path, monkeypatch, capsys):
    """20 epochs, the last 2 plain. The best was epoch 1, so by epoch 18 the
    count stands at 17 of 18: the restart has to reset it, or the run stops
    one epoch into the phase it restarted for."""
    _scripted_val(monkeypatch, [0.5, 0.1])
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    best = model.train(data=str(dataset), project=str(tmp_path), epochs=20, patience=18,
                       val=True, **FAST)
    out = capsys.readouterr().out
    assert "restarting from the best epoch" in out
    assert "early stop" not in out
    assert _epochs_in(best.parent.parent) == 20


def test_resuming_with_more_epochs_starts_the_patience_again(dataset, tmp_path, monkeypatch):
    """A run that stopped early, resumed with more epochs, gets ``patience``
    epochs to improve — not one, as its saved count would give it."""
    _scripted_val(monkeypatch, [0.5, 0.1])
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    best = model.train(data=str(dataset), project=str(tmp_path), epochs=10, patience=2,
                       val=True, **FAST)
    run = best.parent.parent
    assert _epochs_in(run) == 3                       # 0.5, then two without

    Detector(str(best), verbose=False).train(data=str(dataset), project=str(tmp_path),
                                              epochs=13, patience=2, val=True, resume=True,
                                              **FAST)
    assert _epochs_in(run) == 5                       # two more, not one


def test_resuming_keeps_the_run_s_own_queries_and_frozen_parts(dataset, tmp_path, monkeypatch):
    """Resumed from a plain ``Detector("dfine-n")`` without repeating them: the
    optimizer state only fits the same frozen parts, and the trimmed queries
    are part of the run."""
    import torch

    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, n, u: None)
    first = Detector("dfine-n", verbose=False, pretrained=False)
    best = first.train(data=str(dataset), project=str(tmp_path), epochs=1, val=False,
                       queries=40, denoising=10, freeze="backbone", **FAST)

    again = Detector("dfine-n", verbose=False, pretrained=False)
    best2 = again.train(data=str(dataset), project=str(tmp_path), epochs=2, val=False,
                        resume=True, **FAST)
    assert best2.parent == best.parent
    setup = json.loads((best.parent.parent / "run.json").read_text(encoding="utf-8"))
    assert setup["queries"] == 40 and setup["denoising"] == 10
    assert setup["freeze"] == "backbone"
    ckpt = torch.load(best2, map_location="cpu", weights_only=False)
    assert ckpt["queries"] == 40 and ckpt["epoch"] == 1


def test_resume_picks_the_run_written_last_and_only_numbered_ones(tmp_path):
    for i, name in enumerate(["train", "train9", "train10", "train_backup", "trainx"]):
        last = tmp_path / name / "weights" / "last.pt"
        last.parent.mkdir(parents=True)
        last.write_bytes(b"")
        os.utime(last, (1000 + i, 1000 + i))
    # train_backup and trainx are newer, but not this run's
    assert Trainer._resume_dir(tmp_path / "train").name == "train10"
    os.utime(tmp_path / "train9" / "weights" / "last.pt", (5000, 5000))
    assert Trainer._resume_dir(tmp_path / "train").name == "train9"
    assert Trainer._resume_dir(tmp_path / "other") == tmp_path / "other"


def _model_with(names, queries=None):
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(len(names))
    if queries:
        model.net.trim(queries=queries)
    model.ckpt = {"names": dict(names)}
    return model


def test_the_same_classes_in_another_order_keep_their_own_rows():
    """[cat, dog] fine-tuned on [dog, cat]: same count, so the old heads fit
    as they are — each class must still get its own row."""
    import torch

    model = _model_with({0: "cat", 1: "dog"})
    assert model._names_moved({0: "dog", 1: "cat"})
    assert not model._names_moved({0: "cat", 1: "dog"})
    old = model.net.state_dict()
    fresh = model._reheaded_net(2, {0: "dog", 1: "cat"})
    new = fresh.state_dict()
    for k in ("decoder.enc_score_head.weight", "decoder.dec_score_head.0.weight",
              "decoder.denoising_class_embed.weight"):
        assert torch.equal(new[k][0], old[k][1]) and torch.equal(new[k][1], old[k][0]), k
    # everything else came across untouched, the backbone's 8-channel stem too
    assert torch.equal(new["backbone.stem.stem2a.conv.weight"],
                       old["backbone.stem.stem2a.conv.weight"])
    assert model.class_rows == {"dog": "dog", "cat": "cat"}


def test_with_name_matching_off_the_heads_start_fresh():
    import torch

    model = _model_with({0: "cat", 1: "dog"})
    model._match_names = False
    old = model.net.state_dict()
    fresh = model._reheaded_net(2, {0: "cat", 1: "dog"})
    k = "decoder.dec_score_head.0.weight"
    assert not torch.equal(fresh.state_dict()[k], old[k])
    assert model.class_rows == {}


def test_a_new_class_with_the_same_count_does_not_inherit_another_s_row():
    import torch

    model = _model_with({0: "cat", 1: "dog"})
    old = model.net.state_dict()
    fresh = model._reheaded_net(2, {0: "cat", 1: "bird"})
    k = "decoder.dec_score_head.0.weight"
    assert torch.equal(fresh.state_dict()[k][0], old[k][0])       # cat kept
    assert not torch.equal(fresh.state_dict()[k][1], old[k][1])   # bird is not dog


def test_re_heading_keeps_the_checkpoint_s_queries():
    model = _model_with({0: "cat"}, queries=40)
    model.net.decoder.num_denoising = 10
    fresh = model._reheaded_net(3, {0: "cat", 1: "dog", 2: "bird"})
    assert fresh.decoder.num_queries == 40 and fresh.decoder.num_denoising == 10


def test_a_data_yaml_without_val_trains_and_says_so(dataset, tmp_path, capsys):
    cfg = yaml.safe_load(dataset.read_text())
    cfg.pop("val")
    data = tmp_path / "noval.yaml"
    data.write_text(yaml.safe_dump(cfg))
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    best = model.train(data=str(data), project=str(tmp_path), epochs=1, **FAST)
    assert best.exists()
    assert "no val split" in capsys.readouterr().err


def test_a_misspelt_clean_bn_is_an_error(dataset, tmp_path):
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    with pytest.raises(ValueError, match="clean_bn"):
        model.train(data=str(dataset), project=str(tmp_path), epochs=1, val=False,
                    clean_bn="kep", **FAST)


def test_a_numeric_run_name_works(dataset, tmp_path):
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    best = model.train(data=str(dataset), project=str(tmp_path), name=3, epochs=1,
                       val=False, **FAST)
    assert best.parent.parent.name == "3"


def test_run_json_says_which_classes_started_from_a_known_row(dataset, tmp_path):
    cfg = yaml.safe_load(dataset.read_text())
    cfg["names"] = {0: "강아지"}
    data = tmp_path / "dog.yaml"
    data.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    model = _model_with({0: "cat", 1: "dog"})
    best = model.train(data=str(data), project=str(tmp_path), epochs=1, val=False, **FAST)
    setup = json.loads((best.parent.parent / "run.json").read_text(encoding="utf-8"))
    assert setup["start"]["class_rows"] == {"강아지": "dog"}


def test_words_that_name_other_things_as_often_are_not_matched():
    from easydetect.class_names import match_classes

    coco = {0: "scissors", 1: "dining table", 2: "boat", 3: "potted plant", 4: "kite"}
    assert match_classes(coco, {0: "가위", 1: "table", 2: "ship", 3: "plant", 4: "연"}) == {}
    assert match_classes(coco, {0: "scissors", 1: "식탁", 2: "보트"}) == {0: 0, 1: 1, 2: 2}


def test_validation_while_training_uses_no_more_workers_than_training(
        dataset, tmp_path, monkeypatch):
    """workers=0 (Windows' default on a CPU) must hold for the val loader too:
    a spawned val worker re-runs an unguarded script."""
    import torch.utils.data as tud

    seen = []
    real = tud.DataLoader.__init__

    def spy(self, *a, **kw):
        seen.append(kw.get("num_workers", 0))
        real(self, *a, **kw)

    monkeypatch.setattr(tud.DataLoader, "__init__", spy)
    model = Detector("dfine-n", verbose=False)
    model.net = _blank_net(1)
    model.train(data=str(dataset), project=str(tmp_path), epochs=1, val=True, **FAST)
    assert seen and set(seen) == {0}


def test_a_last_batch_of_two_is_trained_on_and_a_last_batch_of_one_is_not():
    from easydetect.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.batch, t.workers = 2, 0
    t.device = __import__("torch").device("cpu")

    class Sized(list):
        pass

    assert t._loader(Sized(range(4))).drop_last is False
    assert t._loader(Sized(range(5))).drop_last is True
    t.batch = 4
    assert t._loader(Sized(range(6))).drop_last is False
