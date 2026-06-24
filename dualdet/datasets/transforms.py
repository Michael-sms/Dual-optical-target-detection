"""Synchronized geometric augmentation for paired RGB-T samples."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SyncAugmentConfig:
    """Configuration for paired-image training augmentation."""

    input_height: int = 512
    input_width: int = 640
    horizontal_flip_prob: float = 0.5
    enable_flip: bool = True


class SyncAugmentPipeline:
    """Apply identical geometric transforms to RGB, TIR and bounding boxes."""

    def __init__(self, config: SyncAugmentConfig | None = None) -> None:
        self.config = config or SyncAugmentConfig()

    def __call__(
        self,
        rgb: Tensor,
        tir: Tensor,
        boxes: Tensor,
        *,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Resize to target resolution and optionally flip all modalities together."""

        rgb, tir, boxes = self._resize(rgb, tir, boxes)
        if training and self.config.enable_flip:
            rgb, tir, boxes = self._maybe_flip(rgb, tir, boxes)
        return rgb, tir, boxes

    def _resize(
        self,
        rgb: Tensor,
        tir: Tensor,
        boxes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        height, width = rgb.shape[-2:]
        target_h = self.config.input_height
        target_w = self.config.input_width
        if height == target_h and width == target_w:
            return rgb, tir, boxes

        rgb = torch.nn.functional.interpolate(
            rgb.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tir = torch.nn.functional.interpolate(
            tir.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if boxes.numel() == 0:
            return rgb, tir, boxes

        scale_x = target_w / width
        scale_y = target_h / height
        scaled = boxes.clone()
        scaled[:, [0, 2]] *= scale_x
        scaled[:, [1, 3]] *= scale_y
        return rgb, tir, scaled

    def _maybe_flip(
        self,
        rgb: Tensor,
        tir: Tensor,
        boxes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if torch.rand(1).item() >= self.config.horizontal_flip_prob:
            return rgb, tir, boxes

        width = rgb.shape[-1]
        rgb = torch.flip(rgb, dims=(-1,))
        tir = torch.flip(tir, dims=(-1,))
        if boxes.numel() == 0:
            return rgb, tir, boxes

        flipped = boxes.clone()
        x1 = boxes[:, 0].clone()
        x2 = boxes[:, 2].clone()
        flipped[:, 0] = width - x2
        flipped[:, 2] = width - x1
        return rgb, tir, flipped
