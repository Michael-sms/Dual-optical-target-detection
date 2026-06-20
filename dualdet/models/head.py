"""Lightweight anchor-free detection head."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

from torch import Tensor, nn

from dualdet.models.backbone import ConvBNAct


class ScalePredictions(NamedTuple):
    """Raw regression distributions and class logits for one feature scale."""

    box_distribution: Tensor
    class_logits: Tensor


class DepthwiseSeparableConv(nn.Module):
    """Efficient spatial mixing followed by channel projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels, in_channels, kernel_size=3, groups=in_channels
        )
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(inputs))


class DetectionScaleHead(nn.Module):
    """Decoupled classification and DFL regression branches for one scale."""

    def __init__(self, in_channels: int, num_classes: int, reg_max: int) -> None:
        super().__init__()
        regression_channels = max(32, reg_max * 4, in_channels // 4)
        classification_channels = max(32, min(in_channels, 128))
        self.regression = nn.Sequential(
            DepthwiseSeparableConv(in_channels, regression_channels),
            nn.Conv2d(regression_channels, reg_max * 4, kernel_size=1),
        )
        self.classification = nn.Sequential(
            DepthwiseSeparableConv(in_channels, classification_channels),
            nn.Conv2d(classification_channels, num_classes, kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> ScalePredictions:
        return ScalePredictions(self.regression(inputs), self.classification(inputs))


class AnchorFreeDetectHead(nn.Module):
    """Produce YOLOv8-style raw predictions without an objectness branch."""

    feature_names = ("p3", "p4", "p5")
    strides = {"p3": 8, "p4": 16, "p5": 32}

    def __init__(
        self,
        feature_channels: Mapping[str, int],
        num_classes: int = 5,
        reg_max: int = 16,
        class_prior_probability: float = 0.01,
    ) -> None:
        super().__init__()
        expected = set(self.feature_names)
        if set(feature_channels) != expected:
            raise ValueError("feature_channels must contain exactly p3, p4 and p5")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if reg_max <= 1:
            raise ValueError("reg_max must be greater than 1")
        if not 0 < class_prior_probability < 1:
            raise ValueError("class_prior_probability must be between 0 and 1")

        self.num_classes = num_classes
        self.reg_max = reg_max
        self._feature_channels = dict(feature_channels)
        self.scale_heads = nn.ModuleDict(
            {
                level: DetectionScaleHead(
                    feature_channels[level], num_classes, reg_max
                )
                for level in self.feature_names
            }
        )
        self._initialize_class_bias(class_prior_probability)

    def _initialize_class_bias(self, prior_probability: float) -> None:
        bias = math.log(prior_probability / (1 - prior_probability))
        for scale_head in self.scale_heads.values():
            classifier = scale_head.classification[-1]
            if not isinstance(classifier, nn.Conv2d):
                raise TypeError("classification output layer must be Conv2d")
            nn.init.constant_(classifier.bias, bias)

    def _validate_features(self, features: Mapping[str, Tensor]) -> None:
        if set(features) != set(self.feature_names):
            raise ValueError("features must contain exactly p3, p4 and p5")
        for level in self.feature_names:
            tensor = features[level]
            if tensor.ndim != 4:
                raise ValueError(f"{level} must have shape [N, C, H, W]")
            expected_channels = self._feature_channels[level]
            if tensor.shape[1] != expected_channels:
                raise ValueError(
                    f"{level} has {tensor.shape[1]} channels; expected {expected_channels}"
                )

    def forward(
        self, features: Mapping[str, Tensor]
    ) -> dict[str, ScalePredictions]:
        """Return raw DFL distributions and class logits at three scales."""

        self._validate_features(features)
        return {
            level: self.scale_heads[level](features[level])
            for level in self.feature_names
        }
