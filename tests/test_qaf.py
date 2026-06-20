"""Tests for quality-aware multi-scale feature fusion."""

import pytest
import torch

from dualdet.models import DualStreamBackbone, MultiScaleQAF, QAFFusionBlock


def _feature_pyramid(
    channels: dict[str, int], batch_size: int = 2
) -> dict[str, torch.Tensor]:
    spatial_shapes = {"p3": (16, 20), "p4": (8, 10), "p5": (4, 5)}
    return {
        level: torch.randn(batch_size, channel_count, *spatial_shapes[level])
        for level, channel_count in channels.items()
    }


def test_qaf_returns_fused_pyramid_and_normalized_weights() -> None:
    channels = {"p3": 16, "p4": 32, "p5": 64}
    qaf = MultiScaleQAF(channels).eval()
    rgb = _feature_pyramid(channels)
    tir = _feature_pyramid(channels)

    with torch.no_grad():
        output = qaf(rgb, tir)

    assert set(output.features) == set(channels)
    assert set(output.modality_weights) == set(channels)
    for level in channels:
        assert output.features[level].shape == rgb[level].shape
        assert output.modality_weights[level].shape == (2, 2)
        assert torch.isfinite(output.modality_weights[level]).all()
        assert (output.modality_weights[level] >= 0).all()
        assert (output.modality_weights[level] <= 1).all()
        assert torch.allclose(
            output.modality_weights[level].sum(dim=1), torch.ones(2)
        )


def test_gate_starts_from_equal_rgb_tir_weights() -> None:
    block = QAFFusionBlock(channels=16).eval()

    with torch.no_grad():
        _, weights = block(
            torch.randn(3, 16, 8, 8),
            torch.randn(3, 16, 8, 8),
        )

    assert torch.allclose(weights, torch.full((3, 2), 0.5))


def test_gate_bias_can_change_modality_preference() -> None:
    block = QAFFusionBlock(channels=8).eval()
    final_layer = block.quality_mlp[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    with torch.no_grad():
        final_layer.bias.copy_(torch.tensor([2.0, -2.0]))
        _, weights = block(
            torch.randn(1, 8, 4, 4),
            torch.randn(1, 8, 4, 4),
        )

    assert weights[0, 0] > 0.95
    assert weights[0, 1] < 0.05


def test_backward_reaches_inputs_gate_and_projection() -> None:
    block = QAFFusionBlock(channels=8)
    rgb = torch.randn(2, 8, 8, 8, requires_grad=True)
    tir = torch.randn(2, 8, 8, 8, requires_grad=True)

    fused, weights = block(rgb, tir)
    (fused.mean() + weights[:, 0].mean()).backward()

    final_layer = block.quality_mlp[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    assert rgb.grad is not None
    assert tir.grad is not None
    assert final_layer.weight.grad is not None
    assert block.projection.conv.weight.grad is not None


def test_qaf_integrates_with_backbone_contract() -> None:
    backbone = DualStreamBackbone(width_multiple=0.125, depth_multiple=0.1).eval()
    qaf = MultiScaleQAF(backbone.feature_channels).eval()
    inputs = torch.randn(1, 3, 128, 160)

    with torch.no_grad():
        dual_features = backbone(inputs, inputs)
        output = qaf(dual_features["rgb"], dual_features["tir"])

    assert output.features["p3"].shape[2:] == (16, 20)
    assert output.features["p4"].shape[2:] == (8, 10)
    assert output.features["p5"].shape[2:] == (4, 5)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_level", "must contain exactly p3, p4 and p5"),
        ("shape_mismatch", "identical shapes"),
        ("channel_mismatch", "expected 16"),
    ],
)
def test_invalid_features_are_rejected(case: str, message: str) -> None:
    channels = {"p3": 16, "p4": 32, "p5": 64}
    qaf = MultiScaleQAF(channels)
    rgb = _feature_pyramid(channels)
    tir = _feature_pyramid(channels)
    if case == "missing_level":
        tir.pop("p5")
    elif case == "shape_mismatch":
        tir["p3"] = torch.randn(2, 16, 8, 8)
    else:
        tir["p3"] = torch.randn(2, 8, 16, 20)

    with pytest.raises(ValueError, match=message):
        qaf(rgb, tir)
