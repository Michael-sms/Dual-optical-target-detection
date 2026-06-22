"""End-to-end assembly of backbone, QAF, neck and detection head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from torch import Tensor, nn

from dualdet.models.backbone import DualStreamBackbone
from dualdet.models.head import AnchorFreeDetectHead, ScalePredictions
from dualdet.models.neck import PANFPN
from dualdet.models.qaf import FusionMode, MultiScaleQAF


class DetectorOutput(NamedTuple):
    """Raw multi-scale predictions plus QAF weights for analysis."""

    predictions: dict[str, ScalePredictions]
    modality_weights: dict[str, Tensor]


class QAFDetector(nn.Module):
    """QAF-SOD model up to raw anchor-free detection predictions."""

    def __init__(
        self,
        num_classes: int = 5,
        width_multiple: float = 0.5,
        depth_multiple: float = 0.33,
        max_channels: int = 512,
        neck_repeats: int = 1,
        reg_max: int = 16,
        fusion_mode: FusionMode = "qaf",
    ) -> None:
        super().__init__()
        self.backbone = DualStreamBackbone(
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            max_channels=max_channels,
        )
        self.qaf = MultiScaleQAF(
            self.backbone.feature_channels, fusion_mode=fusion_mode
        )
        self.neck = PANFPN(self.backbone.feature_channels, repeats=neck_repeats)
        self.head = AnchorFreeDetectHead(
            self.neck.feature_channels,
            num_classes=num_classes,
            reg_max=reg_max,
        )

    def parameter_breakdown(self) -> Mapping[str, int]:
        """Return parameter counts by top-level component and in total."""

        counts = {
            "backbone": sum(parameter.numel() for parameter in self.backbone.parameters()),
            "qaf": sum(parameter.numel() for parameter in self.qaf.parameters()),
            "neck": sum(parameter.numel() for parameter in self.neck.parameters()),
            "head": sum(parameter.numel() for parameter in self.head.parameters()),
        }
        return {**counts, "total": sum(counts.values())}

    def forward(self, rgb: Tensor, tir: Tensor) -> DetectorOutput:
        dual_features = self.backbone(rgb, tir)
        fused = self.qaf(dual_features["rgb"], dual_features["tir"])
        neck_features = self.neck(fused.features)
        predictions = self.head(neck_features)
        return DetectorOutput(predictions, fused.modality_weights)
