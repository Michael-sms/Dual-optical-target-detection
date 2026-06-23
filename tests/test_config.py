"""Tests for fixed/QAF experiment control and model configuration."""

from pathlib import Path

import pytest
import torch

from dualdet.config import ModelConfig
from dualdet.models import MultiScaleQAF, QAFDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_e1_e2_configs_differ_only_by_experiment_and_fusion_mode() -> None:
    fixed = ModelConfig.from_json(PROJECT_ROOT / "configs/model_e1_fixed.json")
    qaf = ModelConfig.from_json(PROJECT_ROOT / "configs/model_e2_qaf.json")

    assert fixed.fusion_mode == "fixed"
    assert qaf.fusion_mode == "qaf"
    assert not fixed.use_p2_head
    assert not qaf.use_p2_head
    assert fixed.controlled_signature() == qaf.controlled_signature()


def test_e2_e3_configs_differ_only_by_experiment_and_p2_head() -> None:
    qaf = ModelConfig.from_json(PROJECT_ROOT / "configs/model_e2_qaf.json")
    p2 = ModelConfig.from_json(PROJECT_ROOT / "configs/model_e3_qaf_p2.json")

    assert qaf.fusion_mode == p2.fusion_mode == "qaf"
    assert not qaf.use_p2_head
    assert p2.use_p2_head
    assert qaf.p2_controlled_signature() == p2.p2_controlled_signature()


def test_fixed_and_qaf_detectors_have_identical_parameter_counts() -> None:
    fixed = QAFDetector(
        width_multiple=0.125,
        depth_multiple=0.1,
        max_channels=128,
        fusion_mode="fixed",
    )
    qaf = QAFDetector(
        width_multiple=0.125,
        depth_multiple=0.1,
        max_channels=128,
        fusion_mode="qaf",
    )

    assert fixed.parameter_breakdown() == qaf.parameter_breakdown()


def test_fixed_mode_forces_equal_weights_even_if_gate_is_biased() -> None:
    fusion = MultiScaleQAF(
        {"p3": 8, "p4": 16, "p5": 32}, fusion_mode="fixed"
    ).eval()
    for block in fusion.blocks.values():
        final_layer = block.quality_mlp[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        with torch.no_grad():
            final_layer.bias.copy_(torch.tensor([8.0, -8.0]))
    features = {
        "p3": torch.randn(2, 8, 8, 8),
        "p4": torch.randn(2, 16, 4, 4),
        "p5": torch.randn(2, 32, 2, 2),
    }

    with torch.no_grad():
        output = fusion(features, features)

    for weights in output.modality_weights.values():
        assert torch.equal(weights, torch.full_like(weights, 0.5))


def test_config_builds_model_with_requested_mode() -> None:
    config = ModelConfig.from_json(PROJECT_ROOT / "configs/model_e1_fixed.json")
    model = config.build_model()

    assert model.qaf.fusion_mode == "fixed"
    assert model.head.num_classes == 5


def test_config_builds_p2_model_when_requested() -> None:
    config = ModelConfig.from_json(PROJECT_ROOT / "configs/model_e3_qaf_p2.json")
    model = config.build_model()

    assert model.use_p2_head
    assert set(model.head.feature_names) == {"p2", "p3", "p4", "p5"}


@pytest.mark.parametrize("mode", ["mean", "dynamic", ""])
def test_invalid_fusion_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="fusion_mode"):
        MultiScaleQAF({"p3": 8, "p4": 16, "p5": 32}, fusion_mode=mode)  # type: ignore[arg-type]
