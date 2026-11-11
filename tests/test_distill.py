# Apache-2.0
"""A larger network teaching a smaller one through train(extra_loss=)."""

from __future__ import annotations

import pytest

from easydetect import Detector

from .conftest import needs_torch

pytestmark = [needs_torch]


def _net(variant, nc=1):
    from easydetect.nn import EDNet

    net = EDNet(variant, nc, pretrained_backbone=False)
    net.trim(queries=40)
    return net


def test_a_student_trains_with_the_teacher_s_scores_and_boxes(dataset, tmp_path, capsys):
    import torch

    from easydetect.distill import Distiller

    teacher = _net("s")
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    model = Detector("ed-n", verbose=False)
    model.net = _net("n")
    kd = Distiller(teacher, model.net)
    model.train(data=str(dataset), project=str(tmp_path), epochs=2, imgsz=64, batch=2,
                workers=0, device="cpu", amp=False, val=False, extra_loss=kd)
    out = capsys.readouterr().out
    assert "kd_enc" in out and "kd_dec" in out
    # the teacher only teaches
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())


def test_the_distillation_loss_is_zero_when_the_student_is_the_teacher():
    """Same network, same pictures: nothing to learn, up to the focal floor."""
    import copy

    import torch

    from easydetect.distill import Distiller

    teacher = _net("n").eval()
    student = copy.deepcopy(teacher).eval()
    kd = Distiller(teacher, student, enc_weight=1.0, dec_weight=1.0)
    imgs = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = student(imgs)
        loss, logs = kd(imgs, out)
    assert logs["kd_enc"] < 1e-6 and logs["kd_dec"] < 1e-4, logs


def test_teacher_and_student_must_know_the_same_classes():
    from easydetect.distill import Distiller

    with pytest.raises(ValueError, match="same classes"):
        Distiller(_net("s", 2), _net("n", 1))


def test_boxes_without_area_do_not_break_the_matching():
    """bfloat16 rounds a tiny box's width to 0: GIoU made 0 / 0, and the matching
    stopped COCO training with "matrix contains invalid numeric entries"."""
    import torch

    from easydetect.distill import Distiller, _giou

    flat = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.3, 0.3]])
    assert torch.isfinite(_giou(flat, flat)).all()
    teacher, student = _net("s").eval(), _net("n").eval()
    kd = Distiller(teacher, student)
    imgs = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = student(imgs)
    out = dict(out)
    out["pred_boxes"] = out["pred_boxes"].clone()
    out["pred_boxes"][:, :10, 2:] = 0.0                      # no width, no height
    out["pred_boxes"][0, 10] = float("nan")                  # and one broken outright
    loss, logs = kd(imgs, out)
    assert logs["kd_dec"] == logs["kd_dec"]                  # not NaN


def test_a_small_input_learns_from_the_teacher_on_a_large_one(dataset, tmp_path, capsys):
    """student_scale: the teacher sees the pictures as loaded, the student
    (in training and in validation) the same pictures at half the size."""
    from easydetect.distill import Distiller

    teacher = _net("s")
    model = Detector("ed-s", verbose=False)
    model.net = _net("s")
    seen = []
    model.net.backbone.register_forward_pre_hook(lambda m, a: seen.append(a[0].shape[-1]))
    teacher.backbone.register_forward_pre_hook(lambda m, a: seen.append(-a[0].shape[-1]))
    kd = Distiller(teacher, model.net, student_scale=0.5)
    model.train(data=str(dataset), project=str(tmp_path), epochs=2, imgsz=128, batch=2,
                workers=0, device="cpu", amp=False, extra_loss=kd)
    out = capsys.readouterr().out
    assert "kd_enc" in out and "kd_dec" in out
    student, teacher_sizes = {s for s in seen if s > 0}, {-s for s in seen if s < 0}
    assert max(student) <= 96 and min(teacher_sizes) >= 96, (student, teacher_sizes)


def test_the_teacher_s_scores_are_pooled_to_the_student_s_grid():
    import torch

    from easydetect.distill import Distiller

    teacher, student = _net("s").eval(), _net("s").eval()
    kd = Distiller(teacher, student, student_scale=0.5)
    imgs = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out = student(imgs)                      # shrunk to 64 by the hook
        loss, logs = kd(imgs, out)
    assert kd._seen["student"].shape[1] == 8 * 8 + 4 * 4 + 2 * 2
    assert kd._seen["teacher"].shape[1] == 16 * 16 + 8 * 8 + 4 * 4
    assert torch.isfinite(loss) and logs["kd_enc"] > 0
