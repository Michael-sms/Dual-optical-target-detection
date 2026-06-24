"""Single-stream RGB backbone for the RGB-only baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping

from torch import Tensor, nn

from dualdet.models.backbone import (
    FeaturePyramid,
    IndependentStem,
    SharedStage,
    _make_divisible,
)


class RGBStreamBackbone(nn.Module):
    """Extract P3/P4/P5 features from RGB images only."""

    feature_names = ("p3", "p4", "p5")

    def __init__(
        self,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        max_channels: int = 512,
    ) -> None:
        super().__init__()
        if width_multiple <= 0 or depth_multiple <= 0:
            raise ValueError("width_multiple and depth_multiple must be positive")

        base_channels = (64, 128, 256, 512, 1024)
        channels = tuple(
            min(_make_divisible(value * width_multiple), max_channels)
            for value in base_channels
        )
        base_repeats = (3, 6, 6, 3)
        repeats = tuple(max(1, round(value * depth_multiple)) for value in base_repeats)

        self.rgb_stem = IndependentStem(channels[:2], repeats[0])
        self.p3_stage = SharedStage(channels[1], channels[2], repeats[1])
        self.p4_stage = SharedStage(channels[2], channels[3], repeats[2])
        self.p5_stage = SharedStage(channels[3], channels[4], repeats[3])
        self._feature_channels = dict(zip(self.feature_names, channels[2:], strict=True))

    @property
    def feature_channels(self) -> Mapping[str, int]:
        return self._feature_channels.copy()

    def forward(self, rgb: Tensor) -> FeaturePyramid:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [N, 3, H, W]")
        features = self.rgb_stem(rgb)
        p3 = self.p3_stage(features, "rgb")
        p4 = self.p4_stage(p3, "rgb")
        p5 = self.p5_stage(p4, "rgb")
        return {"p3": p3, "p4": p4, "p5": p5}
