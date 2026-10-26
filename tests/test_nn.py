# Apache-2.0
"""The network itself: sizes, train/eval outputs, and the export path for deformable attention."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_ov, needs_torch

pytestmark = [needs_torch]


@pytest.mark.parametrize(
    ("size", "decoder_layers", "millions"),
    [("n", 3, 3.8), ("s", 3, 10.3), ("m", 4, 19.6), ("l", 6, 31.2), ("x", 6, 62.6)],
)
def test_each_size_matches_the_published_shape(size, decoder_layers, millions):
    from easydetect.nn import EDNet

    net = EDNet(size, num_classes=80, pretrained_backbone=False)
    assert len(net.decoder.decoder.layers) == decoder_layers
    assert len(net.decoder.dec_score_head) == decoder_layers
    params = sum(p.numel() for p in net.parameters()) / 1e6
    assert params == pytest.approx(millions, abs=0.2)


def _targets(torch, n=1):
    return [{"labels": torch.tensor([0, 1]), "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2],
                                                                    [0.7, 0.6, 0.3, 0.4]])}
            for _ in range(n)]


def test_eval_returns_the_last_layer_and_training_returns_every_layer():
    import torch

    from easydetect.nn import EDNet

    net = EDNet("n", num_classes=5, pretrained_backbone=False)
    x = torch.zeros(1, 3, 320, 320)

    net.eval()
    with torch.no_grad():
        out = net(x)
    assert out["pred_logits"].shape == (1, 300, 5) and out["pred_boxes"].shape == (1, 300, 4)
    assert "aux_outputs" not in out

    net.train()
    out = net(x, _targets(torch))
    # the intermediate decoder layers, the first layer's plain head, the
    # encoder's top-k proposals, and the denoising queries
    assert len(out["aux_outputs"]) == 2
    assert out["pre_outputs"]["pred_logits"].shape == (1, 300, 5)
    assert len(out["enc_aux_outputs"]) == 1
    assert "pred_corners" in out and "dn_outputs" in out


def test_an_input_too_small_for_300_queries_still_runs():
    """4x4 + 2x2 tokens at 64 px is fewer than the query count — must not blow up."""
    import torch

    from easydetect.nn import EDNet

    net = EDNet("n", num_classes=2, pretrained_backbone=False)
    with torch.no_grad():
        out = net.eval()(torch.zeros(1, 3, 64, 64))
    assert out["pred_logits"].shape == (1, 20, 2)
    out = net.train()(torch.zeros(2, 3, 64, 64), _targets(torch, 2))
    assert out["pred_logits"].shape == (2, 20, 2)


def test_unknown_sizes_are_rejected():
    from easydetect.nn import EDNet

    with pytest.raises(ValueError, match="n.*s.*m.*l.*x"):
        EDNet("r18", num_classes=80, pretrained_backbone=False)


def test_the_deploy_form_gives_the_same_answer():
    """Fused convolutions, dropped training heads — same boxes."""
    import copy

    import torch

    from easydetect.nn import EDNet

    torch.manual_seed(0)
    net = EDNet("n", num_classes=3, pretrained_backbone=False).eval()
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        before = net(x)
        after = copy.deepcopy(net).deploy()(x)
    # an untrained net scores its queries almost equally, so fused maths can
    # reorder near-ties: compare as sets, not position by position
    a, b = before["pred_boxes"][0], after["pred_boxes"][0]
    assert torch.cdist(a, b).min(1).values.max() < 1e-3
    scores = [out["pred_logits"].sort(1).values for out in (before, after)]
    assert torch.allclose(*scores, atol=1e-3)


@needs_ov
def test_deformable_attention_survives_the_trip_to_openvino(tmp_path):
    """grid_sample is the one op the export path has to get right."""
    import openvino as ov
    import torch

    from easydetect.nn.decoder import MSDeformableAttention

    attn = MSDeformableAttention(embed_dim=64, num_heads=4, num_levels=3,
                                 num_points=[3, 6, 3]).eval()
    shapes = [(8, 8), (4, 4), (2, 2)]
    q = torch.randn(1, 20, 64)
    ref = torch.rand(1, 20, 1, 4)
    value = torch.randn(1, sum(h * w for h, w in shapes), 64)
    # the decoder hands the attention its values already split per level
    values = value.reshape(1, value.shape[1], 4, 16).permute(0, 2, 3, 1).split(
        [h * w for h, w in shapes], dim=-1)

    class Wrapper(torch.nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.attn = attn

        def forward(self, q, ref, value):
            v = value.reshape(1, value.shape[1], 4, 16).permute(0, 2, 3, 1).split(
                [h * w for h, w in shapes], dim=-1)
            return self.attn(q, ref, v, shapes)

    with torch.no_grad():
        expected = attn(q, ref, values, shapes)
    onnx_path = tmp_path / "attn.onnx"
    torch.onnx.export(
        Wrapper(attn), (q, ref, value), str(onnx_path), opset_version=17,
        input_names=["q", "ref", "value"], output_names=["out"], dynamo=False,
    )
    compiled = ov.Core().compile_model(
        ov.convert_model(str(onnx_path)), "CPU", {"INFERENCE_PRECISION_HINT": "f32"}
    )
    got = compiled([q.numpy(), ref.numpy(), value.numpy()])[compiled.output(0)]
    assert np.abs(got - expected.numpy()).max() < 1e-4
