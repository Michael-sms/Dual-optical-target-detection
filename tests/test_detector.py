"""Tests for the PAN-FPN, anchor-free head and assembled detector."""

import pytest
import torch

from dualdet.models import AnchorFreeDetectHead, P2PANFPN, PANFPN, QAFDetector


CHANNELS = {"p3": 16, "p4": 32, "p5": 64}
P2_CHANNELS = {"p2": 8, **CHANNELS}


def _features(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "p3": torch.randn(batch_size, 16, 16, 20),
        "p4": torch.randn(batch_size, 32, 8, 10),
        "p5": torch.randn(batch_size, 64, 4, 5),
    }


def _p2_features(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "p2": torch.randn(batch_size, 8, 32, 40),
        **_features(batch_size),
    }


def test_pan_fpn_preserves_feature_contract() -> None:
    neck = PANFPN(CHANNELS).eval()

    with torch.no_grad():
        outputs = neck(_features())

    assert outputs["p3"].shape == (2, 16, 16, 20)
    assert outputs["p4"].shape == (2, 32, 8, 10)
    assert outputs["p5"].shape == (2, 64, 4, 5)


def test_detection_head_outputs_expected_raw_channels() -> None:
    head = AnchorFreeDetectHead(CHANNELS, num_classes=5, reg_max=16).eval()

    with torch.no_grad():
        predictions = head(_features())

    for level, height, width in (("p3", 16, 20), ("p4", 8, 10), ("p5", 4, 5)):
        assert predictions[level].box_distribution.shape == (2, 64, height, width)
        assert predictions[level].class_logits.shape == (2, 5, height, width)


def test_p2_pan_fpn_and_detection_head_output_four_scales() -> None:
    neck = P2PANFPN(P2_CHANNELS).eval()
    head = AnchorFreeDetectHead(
        P2_CHANNELS,
        num_classes=5,
        reg_max=16,
        feature_names=("p2", "p3", "p4", "p5"),
    ).eval()

    with torch.no_grad():
        neck_outputs = neck(_p2_features())
        predictions = head(neck_outputs)

    expected_shapes = {
        "p2": (32, 40),
        "p3": (16, 20),
        "p4": (8, 10),
        "p5": (4, 5),
    }
    for level, spatial_shape in expected_shapes.items():
        assert neck_outputs[level].shape[2:] == spatial_shape
        assert predictions[level].box_distribution.shape == (2, 64, *spatial_shape)
        assert predictions[level].class_logits.shape == (2, 5, *spatial_shape)


def test_end_to_end_detector_forward_and_parameter_breakdown() -> None:
    model = QAFDetector(
        width_multiple=0.125,
        depth_multiple=0.1,
        max_channels=128,
    ).eval()
    inputs = torch.randn(1, 3, 128, 160)

    with torch.no_grad():
        output = model(inputs, inputs)

    assert output.predictions["p3"].class_logits.shape == (1, 5, 16, 20)
    assert output.predictions["p4"].class_logits.shape == (1, 5, 8, 10)
    assert output.predictions["p5"].class_logits.shape == (1, 5, 4, 5)
    assert output.predictions["p3"].box_distribution.shape[1] == 64
    assert output.modality_weights["p3"].shape == (1, 2)

    counts = model.parameter_breakdown()
    assert counts["total"] == sum(counts[name] for name in ("backbone", "qaf", "neck", "head"))


def test_p2_detector_forward_adds_stride_four_predictions() -> None:
    model = QAFDetector(
        width_multiple=0.125,
        depth_multiple=0.1,
        max_channels=128,
        use_p2_head=True,
    ).eval()
    inputs = torch.randn(1, 3, 128, 160)

    with torch.no_grad():
        output = model(inputs, inputs)

    assert set(output.predictions) == {"p2", "p3", "p4", "p5"}
    assert output.predictions["p2"].class_logits.shape == (1, 5, 32, 40)
    assert output.predictions["p3"].class_logits.shape == (1, 5, 16, 20)
    assert torch.equal(output.modality_weights["p2"], torch.full((1, 2), 0.5))


def test_end_to_end_backward_reaches_all_components() -> None:
    model = QAFDetector(
        width_multiple=0.125,
        depth_multiple=0.1,
        max_channels=128,
    )
    output = model(
        torch.randn(2, 3, 64, 64),
        torch.randn(2, 3, 64, 64),
    )
    loss = sum(
        prediction.box_distribution.mean() + prediction.class_logits.mean()
        for prediction in output.predictions.values()
    )
    loss.backward()

    assert model.backbone.rgb_stem.stem.conv.weight.grad is not None
    assert model.qaf.blocks["p3"].projection.conv.weight.grad is not None
    assert model.neck.p5_reduce.conv.weight.grad is not None
    assert model.head.scale_heads["p3"].regression[-1].weight.grad is not None


@pytest.mark.parametrize(
    ("component", "message"),
    [
        ("neck_shape", "p3 spatial size must be exactly twice p4"),
        ("head_channels", "expected 16"),
    ],
)
def test_invalid_neck_or_head_features_are_rejected(
    component: str, message: str
) -> None:
    features = _features()
    if component == "neck_shape":
        features["p3"] = torch.randn(2, 16, 15, 20)
        module = PANFPN(CHANNELS)
    else:
        features["p3"] = torch.randn(2, 8, 16, 20)
        module = AnchorFreeDetectHead(CHANNELS)

    with pytest.raises(ValueError, match=message):
        module(features)
