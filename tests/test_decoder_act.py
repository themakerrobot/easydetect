"""The decoder's activation travels with the weights: ReLU (the original
COCO weights), SiLU (DEIM's, the same network trained a better way)."""

import pytest
import torch
from torch import nn

from easydetect import Detector
from easydetect.nn import EDNet
from easydetect.nn.decoder import MLP, TransformerDecoderLayer


def _acts(net):
    kinds = {type(m.act) for m in net.decoder.modules() if isinstance(m, MLP)}
    kinds |= {type(m.activation) for m in net.decoder.modules()
              if isinstance(m, TransformerDecoderLayer)}
    return kinds


def test_every_decoder_activation_follows_decoder_act():
    assert _acts(EDNet("n", 2, pretrained_backbone=False)) == {nn.ReLU}
    assert _acts(EDNet("n", 2, pretrained_backbone=False, decoder_act="silu")) == {nn.SiLU}
    with pytest.raises(ValueError, match="decoder_act"):
        EDNet("n", 2, pretrained_backbone=False, decoder_act="gelu")


def test_a_checkpoint_keeps_its_activation_and_an_old_one_reads_as_relu(tmp_path):
    net = EDNet("n", 80, pretrained_backbone=False, decoder_act="silu")
    new, old = tmp_path / "deim.pt", tmp_path / "old.pt"
    base = {"variant": "n", "num_classes": 80, "model": net.state_dict()}
    torch.save(dict(base, decoder_act="silu"), new)
    torch.save(base, old)                         # before 0.7.13: no decoder_act
    assert _acts(Detector(str(new), verbose=False)._ensure_net()) == {nn.SiLU}
    assert _acts(Detector(str(old), verbose=False)._ensure_net()) == {nn.ReLU}


def test_new_class_heads_keep_the_activation_and_training_saves_it(dataset, tmp_path):
    net = EDNet("n", 80, pretrained_backbone=False, decoder_act="silu")
    start = tmp_path / "start.pt"
    torch.save({"variant": "n", "num_classes": 80, "decoder_act": "silu",
                "names": {i: f"c{i}" for i in range(80)}, "model": net.state_dict()}, start)
    model = Detector(str(start), verbose=False)
    best = model.train(data=str(dataset), project=str(tmp_path), epochs=1, imgsz=64, batch=2,
                       workers=0, device="cpu", val=False, amp=False)
    assert model.net.num_classes != 80 and _acts(model.net) == {nn.SiLU}
    assert torch.load(best, map_location="cpu", weights_only=False)["decoder_act"] == "silu"
