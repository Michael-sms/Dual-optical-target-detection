"""Task-aligned label assignment for anchor-free detection heads."""

from __future__ import annotations

import torch
from torch import Tensor

from dualdet.losses.bbox import bbox_iou


class TaskAlignedAssigner:
    """Assign ground-truth boxes to multi-scale anchor points."""

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-7,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def assign(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return target labels, boxes, score targets and foreground mask."""

        batch_size, num_anchors, num_classes = pred_scores.shape
        device = pred_scores.device
        target_labels = torch.zeros(
            (batch_size, num_anchors), dtype=torch.int64, device=device
        )
        target_bboxes = torch.zeros(
            (batch_size, num_anchors, 4), dtype=pred_bboxes.dtype, device=device
        )
        target_scores = torch.zeros(
            (batch_size, num_anchors, num_classes),
            dtype=pred_scores.dtype,
            device=device,
        )
        fg_mask = torch.zeros(
            (batch_size, num_anchors), dtype=torch.bool, device=device
        )

        for batch_index in range(batch_size):
            valid = mask_gt[batch_index].bool()
            if not valid.any():
                continue

            labels = gt_labels[batch_index][valid]
            boxes = gt_bboxes[batch_index][valid]
            num_gt = boxes.shape[0]
            if num_gt == 0:
                continue

            scores = pred_scores[batch_index]
            bboxes = pred_bboxes[batch_index].detach()
            overlaps = bbox_iou(boxes, bboxes).clamp(min=self.eps)
            class_indices = labels.long() - 1
            class_scores = scores[:, class_indices].T
            alignment = (class_scores**self.alpha) * (overlaps**self.beta)

            topk = min(self.topk, num_anchors)
            _, candidate_indices = alignment.topk(topk, dim=1, largest=True)
            candidate_mask = torch.zeros(
                (num_gt, num_anchors), dtype=torch.bool, device=device
            )
            candidate_mask.scatter_(1, candidate_indices, True)
            candidate_mask &= self._anchor_in_boxes(anchor_points, boxes)

            max_overlaps, matched_gt = overlaps.max(dim=0)
            positive = candidate_mask.any(dim=0)
            positive &= max_overlaps > 0
            if not positive.any():
                continue

            positive_indices = positive.nonzero(as_tuple=False).squeeze(1)
            matched_labels = labels[matched_gt[positive_indices]]
            matched_boxes = boxes[matched_gt[positive_indices]]
            fg_mask[batch_index, positive_indices] = True
            target_labels[batch_index, positive_indices] = matched_labels
            target_bboxes[batch_index, positive_indices] = matched_boxes
            class_indices = matched_labels.long() - 1
            target_scores[batch_index, positive_indices, class_indices] = 1.0

        return target_labels, target_bboxes, target_scores, fg_mask

    @staticmethod
    def _anchor_in_boxes(anchor_points: Tensor, gt_bboxes: Tensor) -> Tensor:
        num_gt = gt_bboxes.shape[0]
        num_anchors = anchor_points.shape[0]
        x = anchor_points[:, 0].view(1, num_anchors)
        y = anchor_points[:, 1].view(1, num_anchors)
        x1 = gt_bboxes[:, 0].view(num_gt, 1)
        y1 = gt_bboxes[:, 1].view(num_gt, 1)
        x2 = gt_bboxes[:, 2].view(num_gt, 1)
        y2 = gt_bboxes[:, 3].view(num_gt, 1)
        return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
