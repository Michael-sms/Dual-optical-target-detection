"""Bounding-box geometry helpers for anchor-free detection."""

from __future__ import annotations

import torch
from torch import Tensor


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert ``[x1, y1, x2, y2]`` boxes to center-size format."""

    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def bbox_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute pairwise IoU for ``[N, 4]`` and ``[M, 4]`` xyxy boxes."""

    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[1] != 4 or boxes2.shape[1] != 4:
        raise ValueError("boxes1 and boxes2 must both have shape [*, 4]")

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter + eps
    return inter / union


def bbox_ciou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute pairwise CIoU for xyxy boxes."""

    iou = bbox_iou(boxes1, boxes2, eps=eps)
    x1, y1, x2, y2 = boxes1.unbind(-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(-1)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cxg = (x1g + x2g) / 2
    cyg = (y1g + y2g) / 2
    center_dist = (cx[:, None] - cxg[None, :]) ** 2 + (cy[:, None] - cyg[None, :]) ** 2

    x_max = torch.max(x2[:, None], x2g[None, :])
    x_min = torch.min(x1[:, None], x1g[None, :])
    y_max = torch.max(y2[:, None], y2g[None, :])
    y_min = torch.min(y1[:, None], y1g[None, :])
    enclosing = (x_max - x_min) ** 2 + (y_max - y_min) ** 2 + eps

    w1 = (x2 - x1).clamp(min=eps)
    h1 = (y2 - y1).clamp(min=eps)
    w2 = (x2g - x1g).clamp(min=eps)
    h2 = (y2g - y1g).clamp(min=eps)
    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(w2[None, :] / h2[None, :]) - torch.atan(w1[:, None] / h1[:, None]), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - center_dist / enclosing - alpha * v


def make_grid_points(
    feature_sizes: dict[str, tuple[int, int]],
    strides: dict[str, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Build center coordinates for each scale in absolute image space."""

    grid_points: dict[str, Tensor] = {}
    for level, (height, width) in feature_sizes.items():
        stride = strides[level]
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid_points[level] = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    return grid_points


def decode_dfl(
    distribution: Tensor,
    anchor_points: Tensor,
    *,
    reg_max: int,
    strides: Tensor,
) -> Tensor:
    """Decode flattened DFL logits into xyxy boxes."""

    if distribution.ndim == 4:
        batch_size, channels, height, width = distribution.shape
        num_points = height * width
        logits = distribution.reshape(batch_size, 4, reg_max, num_points)
    elif distribution.ndim == 3:
        batch_size, num_points, _ = distribution.shape
        logits = distribution.reshape(batch_size, num_points, 4, reg_max).permute(
            0, 2, 3, 1
        )
    else:
        raise ValueError("distribution must have shape [B, C, H, W] or [B, N, 4*reg_max]")

    probs = logits.softmax(dim=2)
    project = torch.arange(reg_max, device=distribution.device, dtype=probs.dtype)
    distances = (probs * project.view(1, 1, reg_max, 1)).sum(dim=2)
    if distribution.ndim == 3:
        distances = distances * strides.view(1, 1, num_points)
    else:
        distances = distances * strides.view(1, 1, 1, num_points)

    x_center = anchor_points[:, 0].view(1, num_points)
    y_center = anchor_points[:, 1].view(1, num_points)
    left, top, right, bottom = distances.unbind(dim=1)
    x1 = x_center - left
    y1 = y_center - top
    x2 = x_center + right
    y2 = y_center + bottom
    boxes = torch.stack((x1, y1, x2, y2), dim=1)
    if distribution.ndim == 3:
        return boxes.transpose(1, 2)
    return boxes.transpose(1, 2)
