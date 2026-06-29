"""Tests for COCO-to-YOLO RGB dataset conversion."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dualdet.utils.yolo_conversion import convert_modality_dataset, convert_rgb_dataset


def _write_split(root: Path, split: str, modality: str = "rgb") -> None:
    image_dir = root / split / modality
    image_dir.mkdir(parents=True)
    (image_dir / "00001.jpg").write_bytes(f"fake-{modality}".encode())
    payload = {
        "images": [{"id": 1, "file_name": "00001.jpg", "width": 100, "height": 50}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [90, 45, 20, 20]},
        ],
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}],
    }
    (root / split / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_convert_rgb_dataset_writes_normalized_labels_and_yaml(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_split(data_root, "train")
    _write_split(data_root, "val")
    test_dir = data_root / "test" / "rgb"
    test_dir.mkdir(parents=True)
    (test_dir / "00002.jpg").write_bytes(b"test-jpeg")
    output_root = data_root / "yolo_rgb"

    report = convert_rgb_dataset(data_root, output_root, image_mode="hardlink")

    label_lines = (output_root / "labels/train/00001.txt").read_text().splitlines()
    assert label_lines == [
        "0 0.20000000 0.20000000 0.20000000 0.20000000",
        "1 0.95000000 0.95000000 0.10000000 0.10000000",
    ]
    assert (output_root / "images/train/00001.jpg").read_bytes() == b"fake-rgb"
    assert (output_root / "images/test/00002.jpg").read_bytes() == b"test-jpeg"
    dataset_yaml = yaml.safe_load((output_root / "rgb_dataset.yaml").read_text())
    assert dataset_yaml["names"] == {0: "car", 1: "truck"}
    assert report["splits"]["train"] == {"images": 1, "boxes": 2, "skipped_boxes": 0}


def test_convert_tir_dataset_uses_tir_images(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_split(data_root, "train", "tir")
    _write_split(data_root, "val", "tir")
    test_dir = data_root / "test" / "tir"
    test_dir.mkdir(parents=True)
    (test_dir / "00002.jpg").write_bytes(b"test-tir")
    output_root = data_root / "yolo_tir"

    report = convert_modality_dataset(
        data_root, output_root, modality="tir", image_mode="hardlink"
    )

    assert report["modality"] == "tir"
    assert report["dataset_yaml"].endswith("tir_dataset.yaml")
    assert (output_root / "images/train/00001.jpg").read_bytes() == b"fake-tir"
    assert (output_root / "images/test/00002.jpg").read_bytes() == b"test-tir"
