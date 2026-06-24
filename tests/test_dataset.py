"""Tests for paired RGB-T dataset loading."""

from pathlib import Path

import pytest
import torch

from dualdet.datasets.coco import load_coco_annotations
from dualdet.datasets.rgb_t import RGBTDataset
from dualdet.datasets.transforms import SyncAugmentConfig, SyncAugmentPipeline


DATA_ROOT = Path("data")


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="course dataset not available")
def test_dataset_returns_paired_tensors_and_valid_boxes() -> None:
    dataset = RGBTDataset(
        data_root=DATA_ROOT,
        split="val",
        training=False,
        augment=SyncAugmentPipeline(SyncAugmentConfig()),
    )
    sample = dataset[0]
    assert sample["rgb"].shape == sample["tir"].shape
    assert sample["rgb"].shape[-2:] == (512, 640)
    assert sample["boxes"].ndim == 2
    assert sample["labels"].ndim == 1


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="course dataset not available")
def test_coco_index_matches_image_count() -> None:
    index = load_coco_annotations(DATA_ROOT / "val" / "val.json")
    rgb_count = len(list((DATA_ROOT / "val" / "rgb").glob("*.jpg")))
    assert index.num_images == rgb_count


def test_sync_flip_keeps_box_alignment() -> None:
    pipeline = SyncAugmentPipeline(SyncAugmentConfig(enable_flip=True, horizontal_flip_prob=1.0))
    rgb = torch.zeros(3, 512, 640)
    tir = torch.ones(3, 512, 640)
    boxes = torch.tensor([[100.0, 50.0, 200.0, 150.0]])
    rgb_out, tir_out, boxes_out = pipeline(rgb, tir, boxes, training=True)
    assert rgb_out.shape[-1] == 640
    assert torch.allclose(
        boxes_out[:, 2] - boxes_out[:, 0],
        boxes[:, 2] - boxes[:, 0],
    )
    assert tir_out.mean() > rgb_out.mean()
