"""Which device training picks: NVIDIA (CUDA), Intel (XPU) or the CPU."""

import pytest
import torch

from easydetect import devices
from easydetect.model import _torch_device


def _machine(monkeypatch, cuda: bool, xpu: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(devices, "xpu_available", lambda: xpu)


@pytest.mark.parametrize("cuda, xpu, picked", [(True, True, "cuda"), (False, True, "xpu"),
                                               (False, False, "cpu")])
def test_the_default_device_prefers_cuda_then_an_intel_gpu(monkeypatch, cuda, xpu, picked):
    _machine(monkeypatch, cuda, xpu)
    assert devices.default_device() == picked


@pytest.mark.parametrize("cuda, xpu, picked", [(True, False, "cuda:0"), (False, True, "xpu:0"),
                                               (False, False, "cuda:0")])
def test_a_device_number_names_the_gpu_this_pc_has(monkeypatch, cuda, xpu, picked):
    _machine(monkeypatch, cuda, xpu)
    assert _torch_device(0) == picked and _torch_device("0") == picked
    assert _torch_device("xpu") == "xpu" and _torch_device("cpu") == "cpu"
    assert _torch_device(None) is None and _torch_device("") is None


def test_bfloat16_on_an_intel_gpu_follows_what_it_says(monkeypatch):
    monkeypatch.setattr(torch, "xpu", type("X", (), {"is_bf16_supported": staticmethod(
        lambda: False)})(), raising=False)
    assert not devices.bf16_supported(torch.device("xpu"))
    assert not devices.bf16_supported(torch.device("cpu"))


def test_asking_for_an_intel_gpu_this_torch_cannot_reach_says_what_to_install(
        dataset, tmp_path, monkeypatch):
    from easydetect.nn import EDNet
    from easydetect.trainer import Trainer

    monkeypatch.setattr("easydetect.trainer.xpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="whl/xpu"):
        Trainer(EDNet("n", 1, pretrained_backbone=False), str(dataset), epochs=1, imgsz=64,
                batch=2, device="xpu", project=str(tmp_path), name="t", workers=0)
