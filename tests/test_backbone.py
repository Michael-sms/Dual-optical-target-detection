"""Tests for the dual-stream backbone contract."""

import pytest
import torch

from dualdet.models.backbone import DualStreamBackbone, SharedConvBNAct


@pytest.fixture()
def tiny_backbone() -> DualStreamBackbone:
    return DualStreamBackbone(width_multiple=0.125, depth_multiple=0.1)


def test_forward_returns_expected_multiscale_shapes(
    tiny_backbone: DualStreamBackbone,
) -> None:
    tiny_backbone.eval()
    rgb = torch.randn(2, 3, 128, 160)
    tir = torch.randn(2, 3, 128, 160)

    with torch.no_grad():
        outputs = tiny_backbone(rgb, tir)

    assert set(outputs) == {"rgb", "tir"}
    expected_spatial_shapes = {"p3": (16, 20), "p4": (8, 10), "p5": (4, 5)}
    for modality in ("rgb", "tir"):
        assert set(outputs[modality]) == set(expected_spatial_shapes)
        for level, spatial_shape in expected_spatial_shapes.items():
            assert outputs[modality][level].shape[:2] == (
                2,
                tiny_backbone.feature_channels[level],
            )
            assert outputs[modality][level].shape[2:] == spatial_shape


def test_high_level_convolution_is_shared_but_batch_norm_is_not(
    tiny_backbone: DualStreamBackbone,
) -> None:
    shared_layer = tiny_backbone.p3_stage.downsample

    assert isinstance(shared_layer, SharedConvBNAct)
    assert set(shared_layer.bn.norms) == {"rgb", "tir"}
    assert (
        shared_layer.bn.norms["rgb"].weight
        is not shared_layer.bn.norms["tir"].weight
    )
    assert tiny_backbone.rgb_stem.stem.conv is not tiny_backbone.tir_stem.stem.conv


def test_backward_reaches_both_stems_and_shared_stage(
    tiny_backbone: DualStreamBackbone,
) -> None:
    outputs = tiny_backbone(
        torch.randn(1, 3, 64, 64),
        torch.randn(1, 3, 64, 64),
    )
    loss = sum(feature.mean() for pyramid in outputs.values() for feature in pyramid.values())
    loss.backward()

    assert tiny_backbone.rgb_stem.stem.conv.weight.grad is not None
    assert tiny_backbone.tir_stem.stem.conv.weight.grad is not None
    assert tiny_backbone.p3_stage.downsample.conv.weight.grad is not None
    assert tiny_backbone.p3_stage.downsample.bn.norms["rgb"].weight.grad is not None
    assert tiny_backbone.p3_stage.downsample.bn.norms["tir"].weight.grad is not None


@pytest.mark.parametrize(
    ("rgb_shape", "tir_shape", "message"),
    [
        ((1, 1, 64, 64), (1, 3, 64, 64), "exactly 3 channels"),
        ((1, 3, 64, 64), (1, 3, 32, 64), "same batch and spatial shape"),
    ],
)
def test_invalid_input_is_rejected(
    tiny_backbone: DualStreamBackbone,
    rgb_shape: tuple[int, ...],
    tir_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tiny_backbone(torch.randn(rgb_shape), torch.randn(tir_shape))
