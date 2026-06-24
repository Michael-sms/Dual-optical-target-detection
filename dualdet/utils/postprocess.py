"""Decode raw head outputs and apply confidence filtering plus NMS."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dualdet.losses.bbox import decode_dfl, make_grid_points
from dualdet.models.head import ScalePredictions


DEFAULT_FEATURE_LEVELS = ("p3", "p4", "p5")
DEFAULT_STRIDES = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}


def non_max_suppression(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Return indices to keep after class-wise NMS."""

    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        index = int(order[0])
        keep.append(index)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(boxes[index, 0], boxes[rest, 0])
        yy1 = torch.maximum(boxes[index, 1], boxes[rest, 1])
        xx2 = torch.minimum(boxes[index, 2], boxes[rest, 2])
        yy2 = torch.minimum(boxes[index, 3], boxes[rest, 3])
        widths = (xx2 - xx1).clamp(min=0)
        heights = (yy2 - yy1).clamp(min=0)
        inter = widths * heights
        area_index = (boxes[index, 2] - boxes[index, 0]) * (boxes[index, 3] - boxes[index, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_index + area_rest - inter
        iou = inter / union.clamp(min=1e-7)
        order = rest[iou <= iou_threshold]

    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


@dataclass
class ImageDetections:
    """Decoded detections for one image."""

    boxes: Tensor
    scores: Tensor
    labels: Tensor


def flatten_predictions(
    predictions: dict[str, ScalePredictions],
    *,
    num_classes: int,
    reg_max: int,
    feature_levels: tuple[str, ...] = DEFAULT_FEATURE_LEVELS,
    strides: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Flatten multi-scale predictions into anchor-aligned tensors."""

    stride_map = strides or DEFAULT_STRIDES
    device = predictions[feature_levels[0]].box_distribution.device
    dist_parts: list[Tensor] = []
    score_parts: list[Tensor] = []
    anchor_parts: list[Tensor] = []
    stride_parts: list[Tensor] = []

    for level in feature_levels:
        scale_pred = predictions[level]
        box_distribution = scale_pred.box_distribution
        class_logits = scale_pred.class_logits
        _, _, height, width = box_distribution.shape
        stride = stride_map[level]
        grid = make_grid_points(
            {level: (height, width)},
            {level: stride},
            device=device,
            dtype=box_distribution.dtype,
        )[level]
        dist_parts.append(
            box_distribution.permute(0, 2, 3, 1).reshape(
                box_distribution.shape[0], -1, 4 * reg_max
            )
        )
        score_parts.append(
            class_logits.permute(0, 2, 3, 1).reshape(
                class_logits.shape[0], -1, num_classes
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


def decode_batch_predictions(
    predictions: dict[str, ScalePredictions],
    *,
    num_classes: int,
    reg_max: int,
    feature_levels: tuple[str, ...] = DEFAULT_FEATURE_LEVELS,
    strides: dict[str, int] | None = None,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
) -> list[ImageDetections]:
    """Decode and NMS-filter predictions for each image in the batch."""

    pred_dist, pred_scores, anchor_points, stride_tensor = flatten_predictions(
        predictions,
        num_classes=num_classes,
        reg_max=reg_max,
        feature_levels=feature_levels,
        strides=strides,
    )
    pred_bboxes = decode_dfl(
        pred_dist,
        anchor_points,
        reg_max=reg_max,
        strides=stride_tensor,
    )
    probabilities = pred_scores.sigmoid()
    batch_size = probabilities.shape[0]
    results: list[ImageDetections] = []

    for batch_index in range(batch_size):
        scores, class_indices = probabilities[batch_index].max(dim=-1)
        keep = scores > conf_threshold
        boxes = pred_bboxes[batch_index][keep]
        scores = scores[keep]
        labels = class_indices[keep] + 1

        if boxes.numel() == 0:
            results.append(
                ImageDetections(
                    boxes=boxes.reshape(0, 4),
                    scores=scores.reshape(0),
                    labels=labels.reshape(0),
                )
            )
            continue

        selected_boxes: list[Tensor] = []
        selected_scores: list[Tensor] = []
        selected_labels: list[Tensor] = []
        for class_id in labels.unique():
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            keep = non_max_suppression(class_boxes, class_scores, iou_threshold)
            selected_boxes.append(class_boxes[keep])
            selected_scores.append(class_scores[keep])
            selected_labels.append(labels[class_mask][keep])

        merged_boxes = torch.cat(selected_boxes, dim=0)
        merged_scores = torch.cat(selected_scores, dim=0)
        merged_labels = torch.cat(selected_labels, dim=0)
        order = merged_scores.argsort(descending=True)[:max_detections]
        results.append(
            ImageDetections(
                boxes=merged_boxes[order],
                scores=merged_scores[order],
                labels=merged_labels[order],
            )
        )

    return results


def detections_to_coco_records(
    detections: ImageDetections,
    *,
    image_id: int,
    input_height: int,
    input_width: int,
) -> list[dict[str, float | int | list[float]]]:
    """Convert xyxy detections to COCO result dictionaries."""

    records: list[dict[str, float | int | list[float]]] = []
    for box, score, label in zip(
        detections.boxes.tolist(),
        detections.scores.tolist(),
        detections.labels.tolist(),
        strict=True,
    ):
        x1 = max(0.0, min(float(input_width), box[0]))
        y1 = max(0.0, min(float(input_height), box[1]))
        x2 = max(0.0, min(float(input_width), box[2]))
        y2 = max(0.0, min(float(input_height), box[3]))
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            continue
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [round(x1, 1), round(y1, 1), round(width, 1), round(height, 1)],
                "score": float(score),
            }
        )
    return records
