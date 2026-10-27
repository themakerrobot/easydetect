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
    from easydetect.nn import EDNet

    return EDNet("n", nc, pretrained_backbone=False)


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
    model = Detector("ed-n", verbose=False)
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
    model = Detector("ed-n", verbose=False)
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
    """Resumed from a plain ``Detector("ed-n")`` without repeating them: the
    optimizer state only fits the same frozen parts, and the trimmed queries
    are part of the run."""
    import torch

    monkeypatch.setattr("easydetect.nn.hgnetv2.HGNetv2.load_imagenet", lambda self, n, u: None)
    first = Detector("ed-n", verbose=False, pretrained=False)
    best = first.train(data=str(dataset), project=str(tmp_path), epochs=1, val=False,
                       queries=40, denoising=10, freeze="backbone", **FAST)

    again = Detector("ed-n", verbose=False, pretrained=False)
    best2 = again.train(data=str(dataset), project=str(tmp_path), epochs=2, val=False,
                        resume=True, **FAST)
    assert best2.parent == best.parent
    setup = json.loads((best.parent.parent / "run.json").read_text(encoding="utf-8"))
    assert setup["queries"] == 40 and setup["denoising"] == 10
    assert setup["freeze"] == "backbone"
    ckpt = torch.load(best2, map_location="cpu", weights_only=False)
    assert ckpt["queries"] == 40 and ckpt["epoch"] == 1


def test_resume_carries_on_the_named_run_and_never_another_with_more_digits(tmp_path):
    def run(name, mtime):
        last = tmp_path / name / "weights" / "last.pt"
        last.parent.mkdir(parents=True)
        last.write_bytes(b"")
        os.utime(last, (mtime, mtime))

    run("job5", 1000)
    run("job57", 5000)                                   # another lab job, newer
    assert Trainer._resume_dir(tmp_path / "job5").name == "job5"
    # train has no last.pt (killed before its first epoch went by): the newest of
    # train2, train3 ... as they were made, and not train_backup or trainx
    (tmp_path / "train").mkdir()
    run("train2", 1000)
    run("train3", 3000)
    run("train_backup", 9000)
    run("trainx", 9000)
    assert Trainer._resume_dir(tmp_path / "train").name == "train3"
    os.utime(tmp_path / "train2" / "weights" / "last.pt", (8000, 8000))
    assert Trainer._resume_dir(tmp_path / "train").name == "train2"
    assert Trainer._resume_dir(tmp_path / "other") == tmp_path / "other"


def _model_with(names, queries=None):
    model = Detector("ed-n", verbose=False)
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
    model = Detector("ed-n", verbose=False)
    model.net = _blank_net(1)
    best = model.train(data=str(data), project=str(tmp_path), epochs=1, **FAST)
    assert best.exists()
    assert "no val split" in capsys.readouterr().err


def test_a_misspelt_clean_bn_is_an_error(dataset, tmp_path):
    model = Detector("ed-n", verbose=False)
    model.net = _blank_net(1)
    with pytest.raises(ValueError, match="clean_bn"):
        model.train(data=str(dataset), project=str(tmp_path), epochs=1, val=False,
                    clean_bn="kep", **FAST)


def test_a_numeric_run_name_works(dataset, tmp_path):
    model = Detector("ed-n", verbose=False)
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
    model = Detector("ed-n", verbose=False)
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


def _fake_coco(tmp_path, monkeypatch, at):
    """A COCO-like 80-class checkpoint whose classes ``at`` sit at given rows."""
    import torch

    names = {i: f"c{i}" for i in range(80)}
    names.update(at)
    net = _blank_net(80)
    path = tmp_path / "coco.pt"
    torch.save({"variant": "n", "num_classes": 80, "names": names,
                "model": net.state_dict()}, path)
    monkeypatch.setattr("easydetect.downloads.download_checkpoint", lambda name: path)
    return net.state_dict()


def _data_named(dataset, tmp_path, names):
    cfg = yaml.safe_load(dataset.read_text())
    cfg["names"] = names
    data = tmp_path / "named.yaml"
    data.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return data


@pytest.mark.parametrize("names", [{0: "cat", 1: "dog"}, {0: "bicycle", 1: "person"}])
def test_a_coco_start_gives_each_named_class_its_coco_row(dataset, tmp_path, monkeypatch,
                                                         names):
    """The default start (the mirror's COCO weights): re-headed once, each class
    from its own COCO row — it crashed (index past the class count) or swapped
    rows when train() re-headed a second time from COCO's names."""
    import torch

    coco = _fake_coco(tmp_path, monkeypatch, {15: "cat", 16: "dog", 1: "bicycle", 0: "person"})
    rows = {"cat": 15, "dog": 16, "bicycle": 1, "person": 0}
    model = Detector("ed-n", verbose=False)
    captured = {}
    real = Detector._reheaded_net

    def spy(self, nc, wanted=None):
        net = real(self, nc, wanted)
        captured.setdefault("head", net.decoder.enc_score_head.weight.detach().clone())
        captured["calls"] = captured.get("calls", 0) + 1
        return net

    monkeypatch.setattr(Detector, "_reheaded_net", spy)
    best = model.train(data=str(_data_named(dataset, tmp_path, names)), project=str(tmp_path),
                       epochs=1, val=False, **FAST)
    assert captured["calls"] == 1
    for j, name in names.items():
        assert torch.equal(captured["head"][j], coco["decoder.enc_score_head.weight"][rows[name]])
    setup = json.loads((best.parent.parent / "run.json").read_text(encoding="utf-8"))
    assert setup["start"]["kind"] == "coco"
    assert setup["start"]["class_rows"] == {n: n for n in names.values()}


def test_a_coco_start_with_80_classes_in_another_order_is_re_headed(dataset, tmp_path,
                                                                   monkeypatch):

    coco = _fake_coco(tmp_path, monkeypatch, {0: "person", 1: "bicycle"})
    names = {i: f"c{i}" for i in range(80)}
    names[0], names[1] = "bicycle", "person"
    model = Detector("ed-n", verbose=False)
    model.train(data=str(_data_named(dataset, tmp_path, names)), project=str(tmp_path),
                epochs=1, val=False, **FAST)
    # trained one epoch since: compare where the rows went, not their values
    head = model.net.decoder.enc_score_head.weight.detach()
    old = coco["decoder.enc_score_head.weight"]
    assert (head[0] - old[1]).abs().max() < (head[0] - old[0]).abs().max()


def test_an_older_run_without_its_plan_in_last_pt_gets_its_patience_back(
        dataset, tmp_path, monkeypatch):
    """A last.pt from 0.7.6 has no "epochs": run.json says what was planned."""
    import torch

    _scripted_val(monkeypatch, [0.5, 0.1])
    model = Detector("ed-n", verbose=False)
    model.net = _blank_net(1)
    best = model.train(data=str(dataset), project=str(tmp_path), epochs=10, patience=2,
                       val=True, **FAST)
    run = best.parent.parent
    last = run / "weights" / "last.pt"
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    del ckpt["epochs"]
    torch.save(ckpt, last)
    Detector(str(best), verbose=False).train(data=str(dataset), project=str(tmp_path),
                                              epochs=13, patience=2, val=True, resume=True,
                                              **FAST)
    assert _epochs_in(run) == 5


def test_with_new_rows_mean_a_class_coco_does_not_know_starts_from_the_mean_row(monkeypatch):
    import torch

    monkeypatch.setenv("EASYDETECT_NEW_ROWS", "mean")
    model = _model_with({0: "cat", 1: "dog"})
    old = model.net.state_dict()
    fresh = model._reheaded_net(2, {0: "cat", 1: "pill"})
    k = "decoder.enc_score_head.weight"
    assert torch.equal(fresh.state_dict()[k][0], old[k][0])                  # cat kept
    assert torch.allclose(fresh.state_dict()[k][1], old[k][:2].mean(0))      # pill: the mean
    pad = "decoder.denoising_class_embed.weight"                             # padding row apart
    assert torch.allclose(fresh.state_dict()[pad][1], old[pad][:2].mean(0))


def test_by_default_all_new_classes_start_from_the_mean_row(monkeypatch):
    """None of the dataset's classes known: each starts from the mean row."""
    import torch

    monkeypatch.delenv("EASYDETECT_NEW_ROWS", raising=False)
    model = _model_with({0: "cat", 1: "dog"})
    old = model.net.state_dict()
    fresh = model._reheaded_net(2, {0: "pill", 1: "capsule"})
    k = "decoder.enc_score_head.weight"
    for j in (0, 1):
        assert torch.allclose(fresh.state_dict()[k][j], old[k][:2].mean(0))
    assert model.new_rows == "mean"


@pytest.mark.parametrize("env, names", [(None, {0: "cat", 1: "pill"}),
                                        ("fresh", {0: "pill", 1: "capsule"})])
def test_new_classes_beside_known_ones_or_asked_fresh_start_fresh(monkeypatch, env, names):
    """A known class among them (african-wildlife: elephant, zebra), or
    EASYDETECT_NEW_ROWS=fresh: the new rows are not the mean."""
    import torch

    if env:
        monkeypatch.setenv("EASYDETECT_NEW_ROWS", env)
    else:
        monkeypatch.delenv("EASYDETECT_NEW_ROWS", raising=False)
    model = _model_with({0: "cat", 1: "dog"})
    old = model.net.state_dict()
    fresh = model._reheaded_net(2, names)
    k = "decoder.enc_score_head.weight"
    assert not torch.allclose(fresh.state_dict()[k][1], old[k][:2].mean(0))
    assert model.new_rows == "fresh"


def test_the_run_records_how_new_classes_started(dataset, tmp_path, monkeypatch):
    coco = _fake_coco(tmp_path, monkeypatch, {})
    del coco
    best = Detector("ed-n", verbose=False).train(
        data=str(dataset), project=str(tmp_path), epochs=1, val=False, **FAST)
    setup = json.loads((best.parent.parent / "run.json").read_text(encoding="utf-8"))
    assert setup["start"]["kind"] == "coco" and setup["start"]["new_rows"] == "mean"


@pytest.mark.parametrize("names, frozen", [({0: "pill", 1: "capsule"}, None),
                                           ({0: "cat", 1: "pill"}, "backbone")])
def test_freeze_auto_keeps_the_backbone_only_when_a_class_is_known(
        dataset, tmp_path, monkeypatch, names, frozen):
    _fake_coco(tmp_path, monkeypatch, {15: "cat"})
    best = Detector("ed-n", verbose=False).train(
        data=str(_data_named(dataset, tmp_path, names)), project=str(tmp_path),
        epochs=1, val=False, freeze="auto", **FAST)
    setup = json.loads((best.parent.parent / "run.json").read_text(encoding="utf-8"))
    assert setup["freeze"] == frozen


def test_a_small_run_with_few_queries_starts_from_the_sized_coco_fine_tune(
        dataset, tmp_path, monkeypatch):
    """train(imgsz=320, queries=100) starts from ed-n-320.pt when the mirror
    has it; the run records which start it was."""
    import torch

    from easydetect import downloads

    net = _blank_net(80)
    sized = tmp_path / "ed-n-64.pt"
    torch.save({"variant": "n", "num_classes": 80, "names": {i: f"c{i}" for i in range(80)},
                "model": net.state_dict(), "queries": 100}, sized)
    asked = []
    monkeypatch.setattr(downloads, "SIZED_STARTS", {64: 100})
    monkeypatch.setattr(downloads, "_asset",
                        lambda name, f, required=True: asked.append(f) or sized)
    monkeypatch.setattr(downloads, "download_checkpoint",
                        lambda name: pytest.fail("the 640 checkpoint is not the start"))
    best = Detector("ed-n", verbose=False).train(
        data=str(dataset), project=str(tmp_path), epochs=1, val=False, queries=100, **FAST)
    assert asked == ["ed-n-64.pt"]
    setup = json.loads((best.parent.parent / "run.json").read_text(encoding="utf-8"))
    assert setup["start"]["weights"] == "ed-n-64"


@pytest.mark.parametrize("imgsz, queries", [(640, 100), (320, None), (320, 300)])
def test_other_runs_keep_the_640_start(imgsz, queries, monkeypatch):
    from easydetect import downloads

    monkeypatch.setattr(downloads, "_asset", lambda *a, **k: pytest.fail("mirror touched"))
    assert downloads.download_sized_checkpoint("ed-n", imgsz, queries) is None


def test_a_sized_start_missing_from_the_mirror_falls_back(monkeypatch):
    from easydetect import downloads

    monkeypatch.setattr(downloads, "_asset", lambda name, f, required=True: None)
    assert downloads.download_sized_checkpoint("ed-n", 320, 100) is None
