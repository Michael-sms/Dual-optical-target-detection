"""Tests for detector builders and training losses."""

import torch

from dualdet.losses.detection import DetectionLoss
from dualdet.models.build import build_detector


def test_build_detector_variants_forward() -> None:
    rgb = torch.randn(1, 3, 128, 160)
    tir = torch.randn(1, 3, 128, 160)
    expected_levels = {
        "e0": ("p3", "p4", "p5"),
        "e1": ("p3", "p4", "p5"),
        "e2": ("p3", "p4", "p5"),
        "e3": ("p2", "p3", "p4", "p5"),
    }
    for experiment, levels in expected_levels.items():
        model = build_detector(
            experiment,
            width_multiple=0.125,
            depth_multiple=0.1,
            max_channels=128,
        ).eval()
        with torch.no_grad():
            output = model(rgb, tir)
        assert tuple(output.predictions) == levels


def test_e3_detection_loss_runs_backward() -> None:
    model = build_detector(
        "e3",
        width_multiple=0.125,
        depth_multiple=0.1,
        max_channels=128,
    )
    criterion = DetectionLoss(
        num_classes=5,
        reg_max=16,
        feature_levels=model.feature_levels,
        strides=model.level_strides,
    )
    rgb = torch.randn(1, 3, 128, 160)
    tir = torch.randn(1, 3, 128, 160)
    output = model(rgb, tir)
    loss = criterion(
        output.predictions,
        [{"boxes": torch.tensor([[20.0, 20.0, 80.0, 90.0]]), "labels": torch.tensor([1]), "image_id": 1}],
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()


def test_detection_loss_runs_backward() -> None:
    model = build_detector(
        "e1",
        width_multiple=0.125,
        depth_multiple=0.1,
        max_channels=128,
    )
    criterion = DetectionLoss(num_classes=5, reg_max=16)
    rgb = torch.randn(2, 3, 128, 160)
    tir = torch.randn(2, 3, 128, 160)
    output = model(rgb, tir)
    targets = [
        {
            "boxes": torch.tensor([[20.0, 20.0, 80.0, 90.0]]),
            "labels": torch.tensor([1]),
            "image_id": 1,
        },
        {
            "boxes": torch.tensor([[30.0, 40.0, 100.0, 120.0], [10.0, 10.0, 40.0, 40.0]]),
            "labels": torch.tensor([2, 1]),
            "image_id": 2,
        },
    ]
    loss = criterion(output.predictions, targets)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.backbone.rgb_stem.stem.conv.weight.grad is not None
