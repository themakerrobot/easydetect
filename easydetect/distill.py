# Apache-2.0
"""A larger detector teaching a smaller one: ``train(extra_loss=Distiller(...))``.

    teacher = Detector("runs/coco/ed-sq-320/weights/best.pt")
    student = Detector("runs/coco/ed-nq-320/weights/best.pt")
    student.train(data="coco.yaml", imgsz=320, extra_loss=Distiller(teacher, student))

Two signals, both on the pictures the student trains on:

* **the encoder's scores, place by place.** Before choosing its queries,
  ed scores every position of its feature maps for every class. The
  student's maps (stride 16 and 32 for ed-n) are the teacher's last ones,
  position for position, so each student score learns the teacher's
  probability there (quality focal loss: the places the two disagree count most).
* **the final boxes, query by query.** The student's queries are matched to
  the teacher's (Hungarian, on box distance and class agreement); each
  matched student query learns the teacher's class probabilities and box,
  weighted by how sure the teacher is.

The teacher must know the same classes in the same order (two COCO models,
say); it runs in eval mode and is never trained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .devices import bf16_supported
from .nn.ops import box_cxcywh_to_xyxy

EPS = 1e-7


def _giou(a, b):
    """Pairwise generalized IoU of xyxy boxes that stays finite when boxes have
    no area: a box whose width rounds to 0 in bfloat16 made 0 / 0 in the plain
    one, and the matching refused the NaN in its cost matrix."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[:, :2])
    rb = torch.min(a[:, None, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b - inter
    iou = inter / (union + EPS)
    hull = (torch.max(a[:, None, 2:], b[:, 2:]) - torch.min(a[:, None, :2], b[:, :2])) \
        .clamp(min=0).prod(-1)
    return iou - (hull - union) / (hull + EPS)


def _net(model):
    """A Detector's network, or a network as given."""
    if hasattr(model, "_ensure_net"):
        model._ensure_net()
        return model.net
    return model


def _quality_focal(logits, target, gamma=2.0):
    """BCE towards soft targets, scaled by how far off the prediction is."""
    p = logits.sigmoid()
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none") \
        * (p - target).abs().pow(gamma)


class Distiller:
    """The extra loss a student trains with; called once a batch by the trainer."""

    def __init__(self, teacher, student, enc_weight: float = 1.0, dec_weight: float = 1.0,
                 box_weight: float = 5.0, giou_weight: float = 2.0):
        self.teacher = _net(teacher)
        student_net = _net(student)
        if self.teacher.num_classes != student_net.num_classes:
            raise ValueError("teacher and student must know the same classes")
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.enc_weight, self.dec_weight = float(enc_weight), float(dec_weight)
        self.box_weight, self.giou_weight = float(box_weight), float(giou_weight)
        self._seen: dict[str, torch.Tensor] = {}
        # the encoder's per-position class scores, caught on their way out
        self.teacher.decoder.enc_score_head.register_forward_hook(
            lambda m, i, o: self._seen.__setitem__("teacher", o))
        student_net.decoder.enc_score_head.register_forward_hook(
            lambda m, i, o: self._seen.__setitem__("student", o))

    @torch.no_grad()
    def _teach(self, imgs):
        device = imgs.device
        if next(self.teacher.parameters()).device != device:
            self.teacher.to(device)
        reduced = device.type in ("cuda", "xpu") and bf16_supported(device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16 if reduced
                                else torch.float32, enabled=reduced):
            out = self.teacher(imgs)
        return out["pred_logits"].float(), out["pred_boxes"].float(), \
            self._seen["teacher"].float()

    def __call__(self, imgs, out) -> tuple[torch.Tensor, dict[str, float]]:
        s_enc = self._seen["student"].float()          # this batch's, from the forward just done
        t_logits, t_boxes, t_enc = self._teach(imgs)
        logs: dict[str, float] = {}
        total = imgs.new_zeros(())

        if self.enc_weight:
            n = s_enc.shape[1]
            if t_enc.shape[1] < n:
                raise ValueError("the teacher's feature maps do not cover the student's")
            target = t_enc[:, -n:].sigmoid()
            enc = _quality_focal(s_enc, target).sum() / target.sum().clamp(min=1.0)
            total = total + self.enc_weight * enc
            logs["kd_enc"] = float(enc.detach())

        if self.dec_weight:
            s_logits = out["pred_logits"].float()
            s_boxes = out["pred_boxes"].float()
            t_prob = t_logits.sigmoid()
            cls_sum = box_sum = weight_sum = s_logits.new_zeros(())
            for b in range(s_logits.shape[0]):
                sp = s_logits[b].sigmoid()
                sx, tx = box_cxcywh_to_xyxy(s_boxes[b]), box_cxcywh_to_xyxy(t_boxes[b])
                with torch.no_grad():
                    cost = (self.box_weight * torch.cdist(s_boxes[b], t_boxes[b], p=1)
                            - self.giou_weight * _giou(sx, tx)
                            + torch.cdist(sp, t_prob[b], p=1))
                    # a NaN from a broken prediction must not end a day of training
                    cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
                    rows, cols = linear_sum_assignment(cost.cpu().numpy())
                rows = torch.as_tensor(rows, device=s_logits.device)
                cols = torch.as_tensor(cols, device=s_logits.device)
                w = t_prob[b, cols].max(-1).values              # how sure the teacher is
                # a pair with a broken (non-finite) prediction teaches nothing
                ok = (torch.isfinite(s_boxes[b, rows]).all(-1) & torch.isfinite(t_boxes[b, cols])
                      .all(-1) & torch.isfinite(s_logits[b, rows]).all(-1))
                w = torch.where(ok, w, torch.zeros_like(w))
                cls = _quality_focal(s_logits[b, rows], t_prob[b, cols]).sum(-1)
                l1 = (s_boxes[b, rows] - t_boxes[b, cols]).abs().sum(-1)
                giou = torch.diag(_giou(sx[rows], tx[cols]))
                cls, l1, giou = (torch.where(ok, v, torch.zeros_like(v)) for v in (cls, l1, giou))
                cls_sum = cls_sum + (w * cls).sum()
                box_sum = box_sum + (w * (self.box_weight * l1
                                          + self.giou_weight * (1 - giou))).sum()
                weight_sum = weight_sum + w.sum()
            dec = (cls_sum + box_sum) / weight_sum.clamp(min=1.0)
            total = total + self.dec_weight * dec
            logs["kd_dec"] = float(dec.detach())
        return total, logs


__all__ = ["Distiller"]
