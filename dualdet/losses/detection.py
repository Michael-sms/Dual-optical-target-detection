"""Combined classification, box and DFL losses for QAF-SOD outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dualdet.losses.assigner import TaskAlignedAssigner
from dualdet.losses.bbox import bbox_ciou, decode_dfl, make_grid_points


@dataclass
class DetectionLossOutput:
    """Scalar loss terms aggregated over a mini-batch."""

    total: Tensor
    box: Tensor
    cls: Tensor
    dfl: Tensor


class DetectionLoss(nn.Module):
    """Compute YOLO-style detection losses from raw multi-scale predictions."""

    default_feature_levels = ("p3", "p4", "p5")
    default_strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}

    def __init__(
        self,
        num_classes: int = 5,
        reg_max: int = 16,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        feature_levels: tuple[str, ...] | None = None,
        strides: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.feature_levels = feature_levels or self.default_feature_levels
        self.strides = strides or {
            level: self.default_strides[level] for level in self.feature_levels
        }
        self.assigner = TaskAlignedAssigner()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        predictions: dict[str, object],
        batch_targets: list[dict[str, Tensor]],
    ) -> DetectionLossOutput:
        device = predictions[self.feature_levels[0]].box_distribution.device  # type: ignore[attr-defined]
        pred_dist, pred_scores, anchor_points, stride_tensor = self._flatten_predictions(
            predictions, device
        )
        pred_bboxes = decode_dfl(
            pred_dist,
            anchor_points,
            reg_max=self.reg_max,
            strides=stride_tensor,
        )

        gt_labels, gt_bboxes, mask_gt = self._pack_targets(batch_targets, device)
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner.assign(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach(),
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        num_pos = max(int(fg_mask.sum().item()), 1)
        cls_loss = self.bce(pred_scores, target_scores).sum() / num_pos * self.cls_gain

        if fg_mask.any():
            pos_pred = pred_bboxes[fg_mask]
            pos_target = target_bboxes[fg_mask]
            box_loss = (
                (1.0 - bbox_ciou(pos_pred, pos_target).diag()).sum()
                / num_pos
                * self.box_gain
            )
            pos_indices = fg_mask.nonzero(as_tuple=False)
            pos_strides = stride_tensor[pos_indices[:, 1]]
            dfl_loss = (
                self._dfl_loss(
                    pred_dist[fg_mask],
                    anchor_points[pos_indices[:, 1]],
                    pos_strides,
                    target_bboxes[fg_mask],
                )
                / num_pos
                * self.dfl_gain
            )
        else:
            zero = pred_scores.sum() * 0.0
            box_loss = zero
            dfl_loss = zero

        total = cls_loss + box_loss + dfl_loss
        return DetectionLossOutput(total=total, box=box_loss, cls=cls_loss, dfl=dfl_loss)

    def _flatten_predictions(
        self,
        predictions: dict[str, object],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dist_parts: list[Tensor] = []
        score_parts: list[Tensor] = []
        anchor_parts: list[Tensor] = []
        stride_parts: list[Tensor] = []

        for level in self.feature_levels:
            scale_pred = predictions[level]
            box_distribution = scale_pred.box_distribution  # type: ignore[attr-defined]
            class_logits = scale_pred.class_logits  # type: ignore[attr-defined]
            _, _, height, width = box_distribution.shape
            stride = self.strides[level]

            grid = make_grid_points(
                {level: (height, width)},
                {level: stride},
                device=device,
                dtype=box_distribution.dtype,
            )[level]
            dist_parts.append(
                box_distribution.permute(0, 2, 3, 1).reshape(
                    box_distribution.shape[0], -1, 4 * self.reg_max
                )
            )
            score_parts.append(
                class_logits.permute(0, 2, 3, 1).reshape(
                    class_logits.shape[0], -1, self.num_classes
                )
            )
            anchor_parts.append(grid)
            stride_parts.append(
                torch.full((height * width,), stride, device=device, dtype=box_distribution.dtype)
            )

        return (
            torch.cat(dist_parts, dim=1),
            torch.cat(score_parts, dim=1),
            torch.cat(anchor_parts, dim=0),
            torch.cat(stride_parts, dim=0),
        )

    def _pack_targets(
        self,
        batch_targets: list[dict[str, Tensor]],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        max_boxes = max(item["boxes"].shape[0] for item in batch_targets)
        max_boxes = max(max_boxes, 1)
        batch_size = len(batch_targets)

        gt_labels = torch.zeros((batch_size, max_boxes), dtype=torch.int64, device=device)
        gt_bboxes = torch.zeros((batch_size, max_boxes, 4), dtype=torch.float32, device=device)
        mask_gt = torch.zeros((batch_size, max_boxes), dtype=torch.bool, device=device)

        for index, item in enumerate(batch_targets):
            boxes = item["boxes"].to(device)
            labels = item["labels"].to(device)
            count = boxes.shape[0]
            if count == 0:
                continue
            gt_labels[index, :count] = labels
            gt_bboxes[index, :count] = boxes
            mask_gt[index, :count] = True

        return gt_labels, gt_bboxes, mask_gt

    def _dfl_loss(
        self,
        pred_dist: Tensor,
        anchor_points: Tensor,
        stride_tensor: Tensor,
        target_boxes: Tensor,
    ) -> Tensor:
        left = anchor_points[:, 0] - target_boxes[:, 0]
        top = anchor_points[:, 1] - target_boxes[:, 1]
        right = target_boxes[:, 2] - anchor_points[:, 0]
        bottom = target_boxes[:, 3] - anchor_points[:, 1]
        target_ltrb = torch.stack((left, top, right, bottom), dim=-1) / stride_tensor.unsqueeze(
            -1
        )

        pred = pred_dist.reshape(-1, 4, self.reg_max)
        target = target_ltrb.clamp(min=0, max=self.reg_max - 1.01)
        left_idx = target.long()
        right_idx = left_idx + 1
        weight_right = target - left_idx.float()
        weight_left = 1.0 - weight_right

        loss_left = F.cross_entropy(
            pred.reshape(-1, self.reg_max),
            left_idx.reshape(-1),
            reduction="none",
        ).reshape(-1, 4)
        loss_right = F.cross_entropy(
            pred.reshape(-1, self.reg_max),
            right_idx.clamp(max=self.reg_max - 1).reshape(-1),
            reduction="none",
        ).reshape(-1, 4)
        return ((loss_left * weight_left + loss_right * weight_right).mean(-1)).sum()
