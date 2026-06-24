"""Experiment-specific detector builders for E0/E1/E2/E3 training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from torch import Tensor, nn

from dualdet.models.backbone import DualStreamBackbone
from dualdet.models.detector import DetectorOutput
from dualdet.models.fusion import FixedMultiScaleFusion
from dualdet.models.head import AnchorFreeDetectHead
from dualdet.models.neck import PANFPN, PANFPNWithP2
from dualdet.models.qaf import MultiScaleQAF
from dualdet.models.rgb_baseline import RGBStreamBackbone

ExperimentName = Literal["e0", "e1", "e2", "e3"]


class RGBBaselineDetector(nn.Module):
    """RGB-only single-stream detector used by experiment E0."""

    def __init__(
        self,
        num_classes: int = 5,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        max_channels: int = 512,
        neck_repeats: int = 1,
        reg_max: int = 16,
    ) -> None:
        super().__init__()
        self.use_p2 = False
        self.backbone = RGBStreamBackbone(
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            max_channels=max_channels,
        )
        self.neck = PANFPN(self.backbone.feature_channels, repeats=neck_repeats)
        self.head = AnchorFreeDetectHead(
            self.neck.feature_channels,
            num_classes=num_classes,
            reg_max=reg_max,
        )

    @property
    def feature_levels(self) -> tuple[str, ...]:
        return self.head.feature_names

    @property
    def level_strides(self) -> dict[str, int]:
        return dict(self.head.strides)

    def parameter_breakdown(self) -> Mapping[str, int]:
        counts = {
            "backbone": sum(parameter.numel() for parameter in self.backbone.parameters()),
            "neck": sum(parameter.numel() for parameter in self.neck.parameters()),
            "head": sum(parameter.numel() for parameter in self.head.parameters()),
        }
        return {**counts, "total": sum(counts.values())}

    def forward(self, rgb: Tensor, tir: Tensor | None = None) -> DetectorOutput:
        del tir
        neck_features = self.neck(self.backbone(rgb))
        return DetectorOutput(self.head(neck_features), {})


class DualStreamDetector(nn.Module):
    """Dual-stream detector with configurable fusion strategy."""

    def __init__(
        self,
        fusion: Literal["fixed", "qaf"],
        num_classes: int = 5,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        max_channels: int = 512,
        neck_repeats: int = 1,
        reg_max: int = 16,
        use_p2: bool = False,
    ) -> None:
        super().__init__()
        self.fusion_type = fusion
        self.use_p2 = use_p2
        self.backbone = DualStreamBackbone(
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            max_channels=max_channels,
        )
        if fusion == "fixed":
            self.fusion = FixedMultiScaleFusion(self.backbone.feature_channels)
        elif fusion == "qaf":
            self.fusion = MultiScaleQAF(self.backbone.feature_channels)
        else:
            raise ValueError(f"unsupported fusion mode: {fusion!r}")

        if use_p2:
            self.neck = PANFPNWithP2(self.backbone.feature_channels, repeats=neck_repeats)
        else:
            self.neck = PANFPN(self.backbone.feature_channels, repeats=neck_repeats)
        self.head = AnchorFreeDetectHead(
            self.neck.feature_channels,
            num_classes=num_classes,
            reg_max=reg_max,
            use_p2=use_p2,
        )

    @property
    def feature_levels(self) -> tuple[str, ...]:
        return self.head.feature_names

    @property
    def level_strides(self) -> dict[str, int]:
        return dict(self.head.strides)

    def parameter_breakdown(self) -> Mapping[str, int]:
        counts = {
            "backbone": sum(parameter.numel() for parameter in self.backbone.parameters()),
            "fusion": sum(parameter.numel() for parameter in self.fusion.parameters()),
            "neck": sum(parameter.numel() for parameter in self.neck.parameters()),
            "head": sum(parameter.numel() for parameter in self.head.parameters()),
        }
        return {**counts, "total": sum(counts.values())}

    def forward(self, rgb: Tensor, tir: Tensor) -> DetectorOutput:
        dual_features = self.backbone(rgb, tir)
        fused = self.fusion(dual_features["rgb"], dual_features["tir"])
        neck_features = self.neck(fused.features)
        return DetectorOutput(self.head(neck_features), fused.modality_weights)


def build_detector(
    experiment: ExperimentName | str,
    *,
    num_classes: int = 5,
    width_multiple: float = 0.5,
    depth_multiple: float = 0.33,
    max_channels: int = 512,
    neck_repeats: int = 1,
    reg_max: int = 16,
) -> nn.Module:
    """Create the detector architecture for an experiment id."""

    name = experiment.lower()
    common = dict(
        num_classes=num_classes,
        width_multiple=width_multiple,
        depth_multiple=depth_multiple,
        max_channels=max_channels,
        neck_repeats=neck_repeats,
        reg_max=reg_max,
    )
    if name == "e0":
        return RGBBaselineDetector(**common)
    if name == "e1":
        return DualStreamDetector(fusion="fixed", **common)
    if name == "e2":
        return DualStreamDetector(fusion="qaf", **common)
    if name == "e3":
        return DualStreamDetector(fusion="qaf", use_p2=True, **common)
    raise ValueError(f"unsupported experiment: {experiment!r}")
