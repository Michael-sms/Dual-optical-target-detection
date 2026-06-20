"""Quality-aware fusion for paired RGB and thermal feature pyramids."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import Tensor, nn

from dualdet.models.backbone import ConvBNAct, FeaturePyramid


class QAFOutput(NamedTuple):
    """Fused features and per-sample modality weights for each scale."""

    features: FeaturePyramid
    modality_weights: dict[str, Tensor]


class QAFFusionBlock(nn.Module):
    """Fuse one pair of same-shaped feature maps using learned quality weights.

    Average and standard-deviation pooling describe the response strength and
    contrast of each modality.  A small MLP predicts one RGB and one TIR logit
    per sample; Softmax then produces normalized scalar modality weights.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        min_hidden_channels: int = 8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0 or min_hidden_channels <= 0:
            raise ValueError("reduction and min_hidden_channels must be positive")

        hidden_channels = max(min_hidden_channels, channels // reduction)
        self.channels = channels
        self.quality_mlp = nn.Sequential(
            nn.Linear(channels * 4, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, 2),
        )
        self.projection = ConvBNAct(channels, channels, kernel_size=1)
        self._reset_gate_parameters()

    def _reset_gate_parameters(self) -> None:
        """Start from equal fusion while keeping the gate trainable."""

        final_layer = self.quality_mlp[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("the final quality gate layer must be Linear")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def _validate_inputs(self, rgb: Tensor, tir: Tensor) -> None:
        for name, tensor in (("rgb", rgb), ("tir", tir)):
            if tensor.ndim != 4:
                raise ValueError(f"{name} feature must have shape [N, C, H, W]")
            if tensor.shape[1] != self.channels:
                raise ValueError(
                    f"{name} feature has {tensor.shape[1]} channels; "
                    f"expected {self.channels}"
                )
        if rgb.shape != tir.shape:
            raise ValueError("rgb and tir features must have identical shapes")
        if rgb.device != tir.device:
            raise ValueError("rgb and tir features must be on the same device")
        if rgb.dtype != tir.dtype:
            raise ValueError("rgb and tir features must have the same dtype")

    @staticmethod
    def _quality_descriptor(features: Tensor) -> Tensor:
        average = features.mean(dim=(2, 3))
        deviation = features.std(dim=(2, 3), unbiased=False)
        return torch.cat((average, deviation), dim=1)

    def forward(self, rgb: Tensor, tir: Tensor) -> tuple[Tensor, Tensor]:
        """Return a fused feature map and normalized ``[RGB, TIR]`` weights."""

        self._validate_inputs(rgb, tir)
        descriptor = torch.cat(
            (self._quality_descriptor(rgb), self._quality_descriptor(tir)), dim=1
        )
        weights = self.quality_mlp(descriptor).softmax(dim=1)
        rgb_weight = weights[:, 0].view(-1, 1, 1, 1)
        tir_weight = weights[:, 1].view(-1, 1, 1, 1)
        fused = self.projection(rgb_weight * rgb + tir_weight * tir)
        return fused, weights


class MultiScaleQAF(nn.Module):
    """Apply independent quality gates to P3, P4 and P5 feature pairs."""

    feature_names = ("p3", "p4", "p5")

    def __init__(
        self,
        feature_channels: Mapping[str, int],
        reduction: int = 16,
        min_hidden_channels: int = 8,
    ) -> None:
        super().__init__()
        missing = set(self.feature_names) - set(feature_channels)
        extra = set(feature_channels) - set(self.feature_names)
        if missing or extra:
            raise ValueError(
                "feature_channels must contain exactly p3, p4 and p5; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        self.blocks = nn.ModuleDict(
            {
                level: QAFFusionBlock(
                    feature_channels[level], reduction, min_hidden_channels
                )
                for level in self.feature_names
            }
        )

    @staticmethod
    def _validate_pyramid(name: str, pyramid: Mapping[str, Tensor]) -> None:
        expected = set(MultiScaleQAF.feature_names)
        actual = set(pyramid)
        if actual != expected:
            raise ValueError(
                f"{name} pyramid must contain exactly p3, p4 and p5; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    def forward(
        self,
        rgb_features: Mapping[str, Tensor],
        tir_features: Mapping[str, Tensor],
    ) -> QAFOutput:
        """Fuse matching RGB/TIR pyramid levels and expose their gate weights."""

        self._validate_pyramid("rgb", rgb_features)
        self._validate_pyramid("tir", tir_features)
        fused_features: FeaturePyramid = {}
        modality_weights: dict[str, Tensor] = {}
        for level in self.feature_names:
            fused_features[level], modality_weights[level] = self.blocks[level](
                rgb_features[level], tir_features[level]
            )
        return QAFOutput(fused_features, modality_weights)
