# Apache-2.0
"""The training loss: Hungarian matching, varifocal classification, L1 +
GIoU boxes, fine-grained localisation (FGL) and decoupled distillation (DDF),
over every decoder layer, the encoder proposals and the denoising queries.

Adapted from D-FINE (https://github.com/Peterande/D-FINE), Copyright (c) 2024
The D-FINE Authors, itself modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR),
Copyright (c) 2023 lyuwenyu — both Apache-2.0. The distributed all-reduce is
gone (training here is one process); everything else follows the reference.
See NOTICE.
"""

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment

from ..nn.ops import bbox2distance, box_cxcywh_to_xyxy, box_iou, generalized_box_iou


def _indices_to(device, groups):
    """Each layer's ``[(query idx, target idx), ...]``, moved to ``device`` in
    one transfer (pinned, so it does not wait for the GPU)."""
    flat = [t for group in groups for pair in group for t in pair]
    if not flat:
        return groups
    joined = torch.cat([t.reshape(-1).long() for t in flat])
    if device.type == "cuda":
        joined = joined.pin_memory().to(device, non_blocking=True)
    else:
        joined = joined.to(device)
    parts = iter(joined.split([t.numel() for t in flat]))
    return [[(next(parts), next(parts)) for _ in group] for group in groups]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        C = self._cost(outputs, targets).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    @torch.no_grad()
    def _cost(self, outputs, targets):
        """The matching cost of every query against every target box in the
        batch, ``(batch, queries, all targets)``, on the outputs' device."""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        return torch.nan_to_num(C.view(bs, num_queries, -1), nan=1.0)

    @torch.no_grad()
    def match_many(self, outputs_list, targets):
        """``forward``'s indices for several layers' outputs at once: the cost
        matrices cross to the CPU in one transfer, not one per layer (each
        transfer waits for the GPU)."""
        sizes = [len(v["boxes"]) for v in targets]
        costs = [self._cost(o, targets) for o in outputs_list]
        flat = torch.cat([c.reshape(-1) for c in costs]).cpu()
        out, start = [], 0
        for c in costs:
            C = flat[start:start + c.numel()].view(c.shape)
            start += c.numel()
            pairs = [linear_sum_assignment(m[i]) for i, m in enumerate(C.split(sizes, -1))]
            out.append([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                        for i, j in pairs])
        return out

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        # C.copy_(C_original)
        return indices_list


class EDCriterion(nn.Module):
    """This class computes the loss for ed."""

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in the original.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]), box_cxcywh_to_xyxy(target_boxes)
                )[0]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                # the same tensor (a layer that is its own teacher): ed
                # compared the values, which waits for the GPU; for two
                # different layers the KL term is computed (0 if they agree)
                if outputs["pred_corners"] is outputs["teacher_corners"]:
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = torch.ones((), dtype=torch.bool, device=mask.device)
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    # means over the matched and the other entries (0 when
                    # none), as sums: no boolean selection, so no waiting
                    on = mask.to(loss_match_local.dtype)
                    loss_match_local1 = (loss_match_local * on).sum() / on.sum().clamp(min=1)
                    loss_match_local2 = ((loss_match_local * (1 - on)).sum()
                                         / (1 - on).sum().clamp(min=1))
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the
        # targets, and of every other layer: all in one go (one GPU -> CPU copy)
        self._clear_cache()
        layers = [outputs_without_aux]
        if "aux_outputs" in outputs:
            decoder = outputs["aux_outputs"] + [outputs["pre_outputs"]]
            layers += decoder + list(outputs["enc_aux_outputs"])
        matched = self.matcher.match_many(layers, targets)
        indices = matched[0]

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            cached_indices = matched[1:1 + len(decoder)]
            cached_indices_enc = matched[1 + len(decoder):]
            indices_aux_list = cached_indices + cached_indices_enc
            indices_go = self._get_go_indices(indices, indices_aux_list)

            # counted on the CPU: a round trip through the GPU would wait for it
            num_boxes_go = float(max(sum(len(x[0]) for x in indices_go), 1))
            # the losses index GPU tensors with these: send them over in one
            # copy now, instead of one blocking copy at every indexing
            moved = _indices_to(outputs["pred_logits"].device,
                                [indices, indices_go, *cached_indices, *cached_indices_enc])
            indices, indices_go = moved[0], moved[1]
            cached_indices = moved[2:2 + len(cached_indices)]
            cached_indices_enc = moved[2 + len(cached_indices):]
        else:
            assert "aux_outputs" in outputs, ""

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = float(max(sum(len(t["labels"]) for t in targets), 1))

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses. For ed
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        else:
            raise AttributeError()

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("vfl",):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list


#: the original COCO settings.
LOSS_WEIGHTS = {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2, "loss_fgl": 0.15, "loss_ddf": 1.5}
MATCH_WEIGHTS = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}

#: log name -> the main-layer loss it reports
_LOGGED = {"vfl": "loss_vfl", "l1": "loss_bbox", "giou": "loss_giou", "fgl": "loss_fgl",
           "ddf": "loss_ddf"}


class DetectionLoss(nn.Module):
    """``loss, logs = criterion(outputs, targets)`` — the total to backpropagate,
    plus the final layer's parts for the log."""

    def __init__(self, num_classes: int, reg_max: int = 32):
        super().__init__()
        matcher = HungarianMatcher(MATCH_WEIGHTS, use_focal_loss=True, alpha=0.25, gamma=2.0)
        self.criterion = EDCriterion(
            matcher, dict(LOSS_WEIGHTS), ["vfl", "boxes", "local"], alpha=0.75, gamma=2.0,
            num_classes=num_classes, reg_max=reg_max,
        )

    def forward(self, outputs, targets):
        losses = self.criterion(outputs, targets)
        total = sum(losses.values())
        logs = {k: losses[v].detach() for k, v in _LOGGED.items() if v in losses}
        # distillation only exists on the layers taught by the last one
        ddf = [v for k, v in losses.items() if k.startswith("loss_ddf")]
        if ddf:
            logs["ddf"] = sum(ddf).detach()
        return total, logs
