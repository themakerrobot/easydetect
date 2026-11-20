# Apache-2.0
"""A crowded dataset (hundreds of boxes a picture) trains in the memory of a
sparse one: the loss and the matcher never build a (boxes x boxes) matrix."""

from __future__ import annotations

import torch

from easydetect.nn.ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_iou_aligned,
    generalized_box_iou,
    generalized_box_iou_aligned,
)
from easydetect.utils.loss import HungarianMatcher


def test_paired_iou_is_the_diagonal_of_the_matrix():
    torch.manual_seed(0)
    a = box_cxcywh_to_xyxy(torch.rand(300, 4) * 0.5 + 0.1)
    b = box_cxcywh_to_xyxy(torch.rand(300, 4) * 0.5 + 0.1)
    assert torch.allclose(torch.diag(box_iou(a, b)[0]), box_iou_aligned(a, b)[0], atol=1e-6)
    assert torch.allclose(torch.diag(generalized_box_iou(a, b)), generalized_box_iou_aligned(a, b),
                          atol=1e-6)


def test_a_big_batch_is_matched_a_picture_at_a_time_the_same_way():
    torch.manual_seed(0)
    matcher = HungarianMatcher({"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
                               use_focal_loss=True)
    out = {"pred_logits": torch.randn(4, 300, 7), "pred_boxes": torch.rand(4, 300, 4) * 0.5 + 0.2}
    targets = [{"labels": torch.randint(0, 7, (n,)), "boxes": torch.rand(n, 4) * 0.4 + 0.2}
               for n in (30, 0, 120, 55)]
    whole = matcher(out, targets)["indices"]
    matcher.PER_PICTURE = 0
    apart = matcher(out, targets)["indices"]
    for (i1, j1), (i2, j2) in zip(whole, apart, strict=True):
        assert torch.equal(i1, i2) and torch.equal(j1, j2)
