"""Classes the COCO weights already know keep their head rows when re-headed."""

import torch

from easydetect.class_names import match_classes
from easydetect.model import copy_class_rows
from easydetect.nn import DFINENet

COCO = {0: "person", 2: "car", 39: "bottle", 41: "cup", 47: "apple", 67: "cell phone"}


def test_names_match_by_spelling_plural_and_korean():
    got = match_classes(COCO, {0: "사과", 1: "Bottles", 2: "rock", 3: "mobile_phone",
                               4: "머그컵", 5: "People"})
    assert got == {0: 47, 1: 39, 3: 67, 4: 41, 5: 0}


def test_each_known_class_lends_its_rows_once():
    assert match_classes(COCO, {0: "apple", 1: "사과"}) == {0: 47}


def test_rows_are_copied_and_the_padding_row_stays_empty():
    old = DFINENet("n", 80, pretrained_backbone=False)
    new = DFINENet("n", 3, pretrained_backbone=False)
    changed = copy_class_rows(new, old.state_dict(), 80, {1: 47})
    a, b = new.state_dict(), old.state_dict()
    assert changed == 1 + 2 * (1 + len(new.decoder.dec_score_head))  # embed + heads' w, b
    assert torch.equal(a["decoder.enc_score_head.weight"][1], b["decoder.enc_score_head.weight"][47])
    assert torch.equal(a["decoder.dec_score_head.0.bias"][1], b["decoder.dec_score_head.0.bias"][47])
    assert torch.equal(a["decoder.denoising_class_embed.weight"][1],
                       b["decoder.denoising_class_embed.weight"][47])
    assert a["decoder.denoising_class_embed.weight"][3].abs().sum() == 0  # padding
