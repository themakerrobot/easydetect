# Apache-2.0
"""Datasets where not every object is boxed (Open Images): a picture's
``<stem>.unknown`` lists the classes neither boxed nor known to be absent
there, which are not taught as absent — nor counted wrong when scoring."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from .conftest import draw, needs_torch

pytestmark = [needs_torch]


def _dataset(root, unknown_classes=True):
    import cv2

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), draw())
            (root / "labels" / split / f"{i}.txt").write_text("1 0.4375 0.5 0.375 0.5\n")
        (root / "labels" / split / "0.unknown").write_text("1 2\n")   # 1 is boxed anyway
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train": "images/train",
                                    "val": "images/val", "names": {0: "a", 1: "b", 2: "c"},
                                    "unknown_classes": unknown_classes}))
    return data


def test_a_picture_knows_all_but_its_unknown_classes_and_always_its_boxed_ones(tmp_path):
    from easydetect.data.dataset import DetDataset

    ds = DetDataset(_dataset(tmp_path), "train", 64, augment=False)
    by_name = {f.name: ds[i][1]["known"].tolist() for i, f in enumerate(ds.files)}
    assert by_name["0.jpg"] == [True, True, False]       # 2 unknown; 1 boxed, so known
    assert by_name["1.jpg"] == [True, True, True]        # no .unknown: every class known
    plain = DetDataset(_dataset(tmp_path / "plain", unknown_classes=False), "train", 64,
                       augment=False)
    assert "known" not in plain[0][1]
    with pytest.raises(ValueError, match="mosaic"):
        DetDataset(_dataset(tmp_path), "train", 64, mosaic=0.5)


@pytest.mark.parametrize("cls_loss", ["vfl", "mal"])
def test_an_unchecked_class_is_not_taught_absent(cls_loss):
    import torch

    from easydetect.utils.loss import LOSS_WEIGHTS, MATCH_WEIGHTS, EDCriterion, HungarianMatcher

    torch.manual_seed(0)
    logits = torch.randn(1, 6, 3, requires_grad=True)
    outputs = {"pred_logits": logits, "pred_boxes": torch.rand(1, 6, 4) * 0.5 + 0.25}
    target = {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
    crit = EDCriterion(HungarianMatcher(MATCH_WEIGHTS, use_focal_loss=True), dict(LOSS_WEIGHTS),
                       [cls_loss], num_classes=3)
    loss_of = getattr(crit, f"loss_labels_{cls_loss}")
    indices = crit.matcher(outputs, [target])["indices"]

    every = loss_of(outputs, [dict(target, known=torch.ones(3, dtype=torch.bool))], indices, 1)
    plain = loss_of(outputs, [target], indices, 1)
    assert torch.allclose(every[f"loss_{cls_loss}"], plain[f"loss_{cls_loss}"])

    known = torch.tensor([True, True, False])               # class 2 unknown here
    loss_of(outputs, [dict(target, known=known)], indices, 1)[f"loss_{cls_loss}"].backward()
    assert logits.grad[..., 2].abs().sum() == 0
    assert logits.grad[..., :2].abs().sum() > 0


def test_scoring_ignores_boxes_of_classes_not_checked(tmp_path, monkeypatch):
    """A confident box of a class unknown in the picture does not cost precision."""
    import torch

    from easydetect import validator

    data = _dataset(tmp_path)
    seen = []
    real_add = validator.Evaluator.add

    def add(self, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls):
        seen.append(sorted(set(np.asarray(pred_cls).tolist())))
        return real_add(self, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls)

    monkeypatch.setattr(validator.Evaluator, "add", add)

    class Net(torch.nn.Module):     # every query scores all three classes high
        num_classes = 3

        def forward(self, x):
            b = x.shape[0]
            return {"pred_logits": torch.full((b, 4, 3), 3.0),
                    "pred_boxes": torch.tensor([[0.44, 0.5, 0.38, 0.5]]).repeat(b, 4, 1)}

    validator.validate_torch(Net(), str(data), imgsz=64, batch=2, device="cpu", workers=0)
    assert 2 not in seen[0] and 2 in seen[1]    # 2 is unknown in 0.jpg, known in 1.jpg
