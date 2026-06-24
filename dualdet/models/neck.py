"""PAN-FPN detection neck for fused P3/P4/P5 features."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dualdet.models.backbone import C2f, ConvBNAct, FeaturePyramid
from dualdet.models.head import DepthwiseSeparableConv


class PANFPN(nn.Module):
    """Aggregate fused features with top-down and bottom-up paths."""

    input_feature_names = ("p3", "p4", "p5")
    feature_names = ("p3", "p4", "p5")

    def __init__(self, feature_channels: Mapping[str, int], repeats: int = 1) -> None:
        super().__init__()
        self._validate_channel_config(feature_channels)
        if repeats <= 0:
            raise ValueError("repeats must be positive")

        c3, c4, c5 = (feature_channels[level] for level in self.input_feature_names)
        self._feature_channels = dict(feature_channels)

        self.p5_reduce = ConvBNAct(c5, c4, kernel_size=1)
        self.p4_top_down = C2f(c4 + c4, c4, repeats)
        self.p4_reduce = ConvBNAct(c4, c3, kernel_size=1)
        self.p3_out = C2f(c3 + c3, c3, repeats)

        self.p3_downsample = ConvBNAct(c3, c3, kernel_size=3, stride=2)
        self.p4_out = C2f(c3 + c4, c4, repeats)
        self.p4_downsample = ConvBNAct(c4, c4, kernel_size=3, stride=2)
        self.p5_out = C2f(c4 + c5, c5, repeats)

    @classmethod
    def _validate_channel_config(cls, feature_channels: Mapping[str, int]) -> None:
        expected = set(cls.input_feature_names)
        actual = set(feature_channels)
        if actual != expected:
            raise ValueError("feature_channels must contain exactly p3, p4 and p5")
        if any(feature_channels[level] <= 0 for level in cls.feature_names):
            raise ValueError("all feature channel counts must be positive")

    def _validate_features(self, features: Mapping[str, Tensor]) -> None:
        expected = set(self.input_feature_names)
        if set(features) != expected:
            raise ValueError("features must contain exactly p3, p4 and p5")
        for level in self.input_feature_names:
            tensor = features[level]
            if tensor.ndim != 4:
                raise ValueError(f"{level} must have shape [N, C, H, W]")
            expected_channels = self._feature_channels[level]
            if tensor.shape[1] != expected_channels:
                raise ValueError(
                    f"{level} has {tensor.shape[1]} channels; expected {expected_channels}"
                )
        p3, p4, p5 = (features[level] for level in self.input_feature_names)
        if p3.shape[0] != p4.shape[0] or p4.shape[0] != p5.shape[0]:
            raise ValueError("all feature levels must have the same batch size")
        if p3.shape[2] != p4.shape[2] * 2 or p3.shape[3] != p4.shape[3] * 2:
            raise ValueError("p3 spatial size must be exactly twice p4")
        if p4.shape[2] != p5.shape[2] * 2 or p4.shape[3] != p5.shape[3] * 2:
            raise ValueError("p4 spatial size must be exactly twice p5")
        if len({p3.device, p4.device, p5.device}) != 1:
            raise ValueError("all feature levels must be on the same device")
        if len({p3.dtype, p4.dtype, p5.dtype}) != 1:
            raise ValueError("all feature levels must have the same dtype")

    @property
    def feature_channels(self) -> Mapping[str, int]:
        """Channel count for each output level."""

        return self._feature_channels.copy()

    def forward(self, features: Mapping[str, Tensor]) -> FeaturePyramid:
        """Return context-enhanced P3/P4/P5 features with unchanged shapes."""

        self._validate_features(features)
        p3, p4, p5 = (features[level] for level in self.input_feature_names)

        p5_reduced = self.p5_reduce(p5)
        p4_td = self.p4_top_down(
            torch.cat((p4, F.interpolate(p5_reduced, size=p4.shape[2:], mode="nearest")), dim=1)
        )
        p4_reduced = self.p4_reduce(p4_td)
        p3_out = self.p3_out(
            torch.cat((p3, F.interpolate(p4_reduced, size=p3.shape[2:], mode="nearest")), dim=1)
        )

        p4_out = self.p4_out(torch.cat((self.p3_downsample(p3_out), p4_td), dim=1))
        p5_out = self.p5_out(torch.cat((self.p4_downsample(p4_out), p5), dim=1))
        return {"p3": p3_out, "p4": p4_out, "p5": p5_out}


class PANFPNWithP2(PANFPN):
    """Extend PAN-FPN with a lightweight stride-4 P2 output for small objects."""

    feature_names = ("p2", "p3", "p4", "p5")

    @classmethod
    def _validate_channel_config(cls, feature_channels: Mapping[str, int]) -> None:
        PANFPN._validate_channel_config(feature_channels)

    def __init__(
        self,
        feature_channels: Mapping[str, int],
        repeats: int = 1,
        p2_channels: int | None = None,
    ) -> None:
        super().__init__(feature_channels, repeats=repeats)
        c3 = feature_channels["p3"]
        self._p2_channels = p2_channels or c3
        self.p2_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            DepthwiseSeparableConv(c3, self._p2_channels),
        )
        self._feature_channels = {
            **self._feature_channels,
            "p2": self._p2_channels,
        }

    def forward(self, features: Mapping[str, Tensor]) -> FeaturePyramid:
        outputs = super().forward(features)
        outputs["p2"] = self.p2_out(outputs["p3"])
        return outputs
