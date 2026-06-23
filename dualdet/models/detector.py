"""End-to-end assembly of backbone, QAF, neck and detection head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from torch import Tensor, nn

from dualdet.models.backbone import DualFeaturePyramid, DualStreamBackbone
from dualdet.models.head import AnchorFreeDetectHead, ScalePredictions
from dualdet.models.neck import P2PANFPN, PANFPN
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
        use_p2_head: bool = False,
    ) -> None:
        super().__init__()
        self.use_p2_head = use_p2_head
        self.backbone = DualStreamBackbone(
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            max_channels=max_channels,
        )
        self.qaf = MultiScaleQAF(
            self.backbone.feature_channels, fusion_mode=fusion_mode
        )
        if use_p2_head:
            self.neck = P2PANFPN(
                self.backbone.feature_channels_with_p2, repeats=neck_repeats
            )
            head_feature_names = ("p2", "p3", "p4", "p5")
        else:
            self.neck = PANFPN(self.backbone.feature_channels, repeats=neck_repeats)
            head_feature_names = ("p3", "p4", "p5")
        self.head = AnchorFreeDetectHead(
            self.neck.feature_channels,
            num_classes=num_classes,
            reg_max=reg_max,
            feature_names=head_feature_names,
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

    @staticmethod
    def _fixed_p2_fusion(
        dual_features: DualFeaturePyramid,
    ) -> tuple[Tensor, Tensor]:
        """Fuse shallow P2 features with fixed equal modality weights."""

        rgb_p2 = dual_features["rgb"]["p2"]
        tir_p2 = dual_features["tir"]["p2"]
        if rgb_p2.shape != tir_p2.shape:
            raise ValueError("rgb and tir P2 features must have identical shapes")
        weights = rgb_p2.new_full((rgb_p2.shape[0], 2), 0.5)
        return 0.5 * (rgb_p2 + tir_p2), weights

    @staticmethod
    def _standard_qaf_features(
        dual_features: DualFeaturePyramid,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Select only P3/P4/P5 levels expected by the QAF module."""

        levels = ("p3", "p4", "p5")
        return (
            {level: dual_features["rgb"][level] for level in levels},
            {level: dual_features["tir"][level] for level in levels},
        )

    def forward(self, rgb: Tensor, tir: Tensor) -> DetectorOutput:
        dual_features = (
            self.backbone.forward_with_p2(rgb, tir)
            if self.use_p2_head
            else self.backbone(rgb, tir)
        )
        rgb_qaf, tir_qaf = self._standard_qaf_features(dual_features)
        fused = self.qaf(rgb_qaf, tir_qaf)
        modality_weights = dict(fused.modality_weights)
        if self.use_p2_head:
            p2_feature, p2_weights = self._fixed_p2_fusion(dual_features)
            neck_inputs = {"p2": p2_feature, **fused.features}
            modality_weights = {"p2": p2_weights, **modality_weights}
        else:
            neck_inputs = fused.features
        neck_features = self.neck(neck_inputs)
        predictions = self.head(neck_features)
        return DetectorOutput(predictions, modality_weights)
