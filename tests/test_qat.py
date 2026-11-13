# Apache-2.0
"""Quantization-aware training: folding, rounding, training and the QDQ export."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from easydetect import Detector

from .conftest import needs_torch

pytestmark = [needs_torch]


def _net(variant="s", nc=1):
    from easydetect.nn import EDNet

    net = EDNet(variant, nc, pretrained_backbone=False)
    net.trim(queries=40)
    return net.eval()


def _pictures(n=4, size=64):
    import torch

    torch.manual_seed(0)
    return [torch.rand(2, 3, size, size) for _ in range(n)]


def test_folding_the_batchnorms_changes_nothing():
    import torch

    from easydetect.qat import fold_batchnorm

    net = _net()
    for m in net.modules():                   # BatchNorms that do something
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        before = net(x)["pred_logits"]
        after = fold_batchnorm(copy.deepcopy(net)).eval()(x)["pred_logits"]
    assert torch.allclose(before, after, atol=1e-3), float((before - after).abs().max())
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in fold_batchnorm(net).modules())


def test_the_export_runs_in_8_bits_as_trained(tmp_path):
    import onnx
    import onnxruntime as ort
    import torch

    from easydetect.qat import QuantConv2d, calibrate, export_qat_onnx, prepare_qat

    net = prepare_qat(_net())
    assert not isinstance(net.backbone.stem.stem1.conv, QuantConv2d)     # the stem stays float
    calibrate(net, _pictures())
    out = export_qat_onnx(net, {0: "thing"}, 64, tmp_path / "q.onnx")
    model = onnx.load(str(out))
    ops = [n.op_type for n in model.graph.node]
    assert ops.count("QuantizeLinear") > 20 and ops.count("DequantizeLinear") > 20
    int8 = [i for i in model.graph.initializer if i.data_type == onnx.TensorProto.INT8]
    assert len(int8) >= sum(isinstance(m, QuantConv2d) for m in net.modules())
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        logits = net.eval()(x)["pred_logits"].sigmoid().numpy()
    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert "thing" in s.get_modelmeta().custom_metadata_map["easydetect.names"]
    _, scores = s.run(None, {"images": x.numpy()})
    assert np.abs(scores - logits).max() < 0.05


def test_a_quantized_network_trains(dataset, tmp_path):
    from easydetect.qat import calibrate, prepare_qat, quant_convs

    model = Detector("ed-s", verbose=False)
    model.net = prepare_qat(_net())
    calibrate(model.net, _pictures())
    before = quant_convs(model.net)[5].weight.detach().clone()
    best = model.train(data=str(dataset), project=str(tmp_path), epochs=3, imgsz=64, batch=2,
                       workers=0, device="cpu", amp=False, lr0=1e-3, val=False)
    assert best is not None                  # (epoch 1 of a tiny set is all warm-up)
    after = quant_convs(model.net)            # reloaded from best.pt: still quantizing
    assert after and not after[5].weight.detach().equal(before)


@pytest.mark.parametrize("size", [64])
def test_calibration_sets_ranges(size):
    from easydetect.qat import calibrate, prepare_qat, quant_convs

    net = prepare_qat(_net())
    calibrate(net, _pictures(size=size))
    c = quant_convs(net)[0]
    assert float(c.hi) > float(c.lo) and float(c.out_hi) > float(c.out_lo)
