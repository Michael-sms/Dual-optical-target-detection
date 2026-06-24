"""Fixed-ratio multi-scale fusion for dual-stream features."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from dualdet.models.backbone import ConvBNAct, FeaturePyramid
from dualdet.models.qaf import QAFOutput


class FixedFusionBlock(nn.Module):
    """Fuse RGB/TIR features with equal weights followed by projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.projection = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, rgb: Tensor, tir: Tensor) -> tuple[Tensor, Tensor]:
        if rgb.shape != tir.shape:
            raise ValueError("rgb and tir features must have identical shapes")
        fused = self.projection(0.5 * rgb + 0.5 * tir)
        weights = rgb.new_tensor([0.5, 0.5]).view(1, 2).expand(rgb.shape[0], -1)
        return fused, weights


class FixedMultiScaleFusion(nn.Module):
    """Apply fixed 0.5/0.5 fusion on P3, P4 and P5 feature pairs."""

    feature_names = ("p3", "p4", "p5")

    def __init__(self, feature_channels: Mapping[str, int]) -> None:
        super().__init__()
        expected = set(self.feature_names)
        if set(feature_channels) != expected:
            raise ValueError("feature_channels must contain exactly p3, p4 and p5")
        self.blocks = nn.ModuleDict(
            {level: FixedFusionBlock(feature_channels[level]) for level in self.feature_names}
        )

    def forward(
        self,
        rgb_features: Mapping[str, Tensor],
        tir_features: Mapping[str, Tensor],
    ) -> QAFOutput:
        fused_features: FeaturePyramid = {}
        modality_weights: dict[str, Tensor] = {}
        for level in self.feature_names:
            fused_features[level], modality_weights[level] = self.blocks[level](
                rgb_features[level], tir_features[level]
            )
        return QAFOutput(fused_features, modality_weights)
