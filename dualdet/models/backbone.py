"""Dual-stream CSP backbone for paired RGB and thermal images.

The two modalities use independent shallow stems.  From P3 onward, every
convolution is shared while batch-normalization statistics and affine
parameters remain modality-specific.  The returned P3/P4/P5 feature maps are
intended to be consumed by the later QAF fusion module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor, nn

Modality = Literal["rgb", "tir"]
FeaturePyramid = dict[str, Tensor]
DualFeaturePyramid = dict[Modality, FeaturePyramid]


def _make_divisible(value: float, divisor: int = 8) -> int:
    """Round a channel count up to a hardware-friendly multiple."""

    return max(divisor, int(math.ceil(value / divisor) * divisor))


class ConvBNAct(nn.Module):
    """Convolution followed by BatchNorm and SiLU for one modality."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(inputs)))


class ModalityBatchNorm2d(nn.Module):
    """Keep separate normalization parameters and statistics per modality."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.norms = nn.ModuleDict(
            {
                "rgb": nn.BatchNorm2d(num_features),
                "tir": nn.BatchNorm2d(num_features),
            }
        )

    def forward(self, inputs: Tensor, modality: Modality) -> Tensor:
        if modality not in self.norms:
            raise ValueError(f"unsupported modality: {modality!r}")
        return self.norms[modality](inputs)


class SharedConvBNAct(nn.Module):
    """A shared convolution with modality-specific BatchNorm layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = ModalityBatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor, modality: Modality) -> Tensor:
        return self.act(self.bn(self.conv(inputs), modality))


class Bottleneck(nn.Module):
    """CSP bottleneck used in each independent shallow stem."""

    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        self.cv1 = ConvBNAct(channels, channels, kernel_size=3)
        self.cv2 = ConvBNAct(channels, channels, kernel_size=3)
        self.shortcut = shortcut

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.cv2(self.cv1(inputs))
        return inputs + outputs if self.shortcut else outputs


class C2f(nn.Module):
    """Compact CSP block compatible with YOLO-style feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, repeats: int) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden_channels * 2, kernel_size=1)
        self.blocks = nn.ModuleList(Bottleneck(hidden_channels) for _ in range(repeats))
        self.cv2 = ConvBNAct(
            hidden_channels * (2 + repeats), out_channels, kernel_size=1
        )

    def forward(self, inputs: Tensor) -> Tensor:
        features = list(self.cv1(inputs).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1]))
        return self.cv2(torch.cat(features, dim=1))


class SharedBottleneck(nn.Module):
    """Bottleneck whose convolutions are reused for RGB and TIR."""

    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        self.cv1 = SharedConvBNAct(channels, channels, kernel_size=3)
        self.cv2 = SharedConvBNAct(channels, channels, kernel_size=3)
        self.shortcut = shortcut

    def forward(self, inputs: Tensor, modality: Modality) -> Tensor:
        outputs = self.cv2(self.cv1(inputs, modality), modality)
        return inputs + outputs if self.shortcut else outputs


class SharedC2f(nn.Module):
    """C2f block with shared convolutions and independent normalization."""

    def __init__(self, in_channels: int, out_channels: int, repeats: int) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.cv1 = SharedConvBNAct(in_channels, hidden_channels * 2, kernel_size=1)
        self.blocks = nn.ModuleList(
            SharedBottleneck(hidden_channels) for _ in range(repeats)
        )
        self.cv2 = SharedConvBNAct(
            hidden_channels * (2 + repeats), out_channels, kernel_size=1
        )

    def forward(self, inputs: Tensor, modality: Modality) -> Tensor:
        features = list(self.cv1(inputs, modality).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1], modality))
        return self.cv2(torch.cat(features, dim=1), modality)


class IndependentStem(nn.Module):
    """Modality-specific stride-4 stem."""

    def __init__(self, channels: tuple[int, int], repeats: int) -> None:
        super().__init__()
        c1, c2 = channels
        self.stem = ConvBNAct(3, c1, kernel_size=3, stride=2)
        self.downsample = ConvBNAct(c1, c2, kernel_size=3, stride=2)
        self.c2f = C2f(c2, c2, repeats)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.c2f(self.downsample(self.stem(inputs)))


class SharedStage(nn.Module):
    """One downsampling CSP stage reused by both modalities."""

    def __init__(self, in_channels: int, out_channels: int, repeats: int) -> None:
        super().__init__()
        self.downsample = SharedConvBNAct(
            in_channels, out_channels, kernel_size=3, stride=2
        )
        self.c2f = SharedC2f(out_channels, out_channels, repeats)

    def forward(self, inputs: Tensor, modality: Modality) -> Tensor:
        return self.c2f(self.downsample(inputs, modality), modality)


class DualStreamBackbone(nn.Module):
    """Extract paired P3/P4/P5 features from RGB and TIR images.

    Args:
        width_multiple: Width multiplier applied to base channels. ``0.5``
            follows the YOLOv8s channel scale.
        depth_multiple: Depth multiplier applied to CSP repeat counts. ``0.33``
            follows the YOLOv8s depth scale.
        max_channels: Upper bound for any stage's channel count.
    """

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
        if max_channels < 8:
            raise ValueError("max_channels must be at least 8")

        base_channels = (64, 128, 256, 512, 1024)
        channels = tuple(
            min(_make_divisible(value * width_multiple), max_channels)
            for value in base_channels
        )
        base_repeats = (3, 6, 6, 3)
        repeats = tuple(max(1, round(value * depth_multiple)) for value in base_repeats)

        self.rgb_stem = IndependentStem(channels[:2], repeats[0])
        self.tir_stem = IndependentStem(channels[:2], repeats[0])
        self.p3_stage = SharedStage(channels[1], channels[2], repeats[1])
        self.p4_stage = SharedStage(channels[2], channels[3], repeats[2])
        self.p5_stage = SharedStage(channels[3], channels[4], repeats[3])
        self._stem_feature_channels = channels[1]
        self._feature_channels = dict(zip(self.feature_names, channels[2:], strict=True))

    @property
    def feature_channels(self) -> Mapping[str, int]:
        """Channel count for each returned pyramid level."""

        return self._feature_channels.copy()

    @property
    def stem_feature_channels(self) -> int:
        """Channel count of the stride-4 shallow stem feature used as P2."""

        return self._stem_feature_channels

    @property
    def feature_channels_with_p2(self) -> Mapping[str, int]:
        """Channel count for optional P2 plus the standard P3/P4/P5 levels."""

        return {"p2": self.stem_feature_channels, **self.feature_channels}

    @staticmethod
    def _validate_inputs(rgb: Tensor, tir: Tensor) -> None:
        for name, tensor in (("rgb", rgb), ("tir", tir)):
            if tensor.ndim != 4:
                raise ValueError(f"{name} must have shape [N, 3, H, W]")
            if tensor.shape[1] != 3:
                raise ValueError(f"{name} must contain exactly 3 channels")
        if rgb.shape[0] != tir.shape[0] or rgb.shape[2:] != tir.shape[2:]:
            raise ValueError("rgb and tir must have the same batch and spatial shape")
        if rgb.device != tir.device:
            raise ValueError("rgb and tir must be on the same device")

    def _forward_shared(
        self, inputs: Tensor, modality: Modality
    ) -> FeaturePyramid:
        p3 = self.p3_stage(inputs, modality)
        p4 = self.p4_stage(p3, modality)
        p5 = self.p5_stage(p4, modality)
        return {"p3": p3, "p4": p4, "p5": p5}

    def forward(self, rgb: Tensor, tir: Tensor) -> DualFeaturePyramid:
        """Return separate modality features at strides 8, 16 and 32."""

        self._validate_inputs(rgb, tir)
        rgb_features = self._forward_shared(self.rgb_stem(rgb), "rgb")
        tir_features = self._forward_shared(self.tir_stem(tir), "tir")
        return {"rgb": rgb_features, "tir": tir_features}

    def forward_with_p2(self, rgb: Tensor, tir: Tensor) -> DualFeaturePyramid:
        """Return stride-4 P2 plus standard P3/P4/P5 modality features."""

        self._validate_inputs(rgb, tir)
        rgb_p2 = self.rgb_stem(rgb)
        tir_p2 = self.tir_stem(tir)
        rgb_features = {"p2": rgb_p2, **self._forward_shared(rgb_p2, "rgb")}
        tir_features = {"p2": tir_p2, **self._forward_shared(tir_p2, "tir")}
        return {"rgb": rgb_features, "tir": tir_features}
