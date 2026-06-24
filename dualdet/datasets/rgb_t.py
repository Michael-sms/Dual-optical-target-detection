"""Paired RGB-T dataset backed by COCO-style annotations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from dualdet.datasets.coco import COCOAnnotationIndex, load_coco_annotations
from dualdet.datasets.transforms import SyncAugmentConfig, SyncAugmentPipeline
from dualdet.utils.submission import filename_to_image_id


@dataclass(frozen=True)
class TestImageRecord:
    """Metadata for one paired RGB/TIR test image."""

    image_id: int
    file_name: str


class RGBTTestDataset(Dataset):
    """Load paired RGB/TIR test images without public annotations."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "test",
        *,
        augment: SyncAugmentPipeline | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.rgb_dir = self.data_root / split / "rgb"
        self.tir_dir = self.data_root / split / "tir"
        self.augment = augment or SyncAugmentPipeline(SyncAugmentConfig())
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        if not self.tir_dir.is_dir():
            raise FileNotFoundError(f"TIR directory not found: {self.tir_dir}")

        records: list[TestImageRecord] = []
        for rgb_path in sorted(self.rgb_dir.glob("*.jpg")):
            file_name = rgb_path.name
            tir_path = self.tir_dir / file_name
            if not tir_path.is_file():
                raise FileNotFoundError(f"Missing paired TIR image: {tir_path}")
            records.append(
                TestImageRecord(
                    image_id=filename_to_image_id(file_name),
                    file_name=file_name,
                )
            )
        if not records:
            raise FileNotFoundError(f"No test images found under {self.rgb_dir}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | int | str]:
        record = self.records[index]
        rgb_path = self.rgb_dir / record.file_name
        tir_path = self.tir_dir / record.file_name

        rgb = RGBTDataset._load_image(rgb_path)
        tir = RGBTDataset._load_image(tir_path)
        if rgb.shape[-2:] != tir.shape[-2:]:
            raise ValueError(
                f"RGB/TIR size mismatch for {record.file_name}: "
                f"{tuple(rgb.shape[-2:])} vs {tuple(tir.shape[-2:])}"
            )

        empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
        rgb, tir, _ = self.augment(rgb, tir, empty_boxes, training=False)
        return {
            "rgb": rgb,
            "tir": tir,
            "boxes": empty_boxes,
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": record.image_id,
            "file_name": record.file_name,
        }


class RGBTDataset(Dataset):
    """Load synchronized RGB/TIR image pairs and COCO bounding boxes."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        annotation_file: str | Path | None = None,
        *,
        training: bool = True,
        min_box_size: float = 2.0,
        augment: SyncAugmentPipeline | None = None,
        valid_category_ids: Sequence[int] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.training = training
        self.min_box_size = min_box_size
        self.valid_category_ids = set(valid_category_ids or (1, 2, 3, 4, 5))
        self.rgb_dir = self.data_root / split / "rgb"
        self.tir_dir = self.data_root / split / "tir"
        self.annotation_path = Path(
            annotation_file or self.data_root / split / f"{split}.json"
        )
        self.index = load_coco_annotations(self.annotation_path)
        self.augment = augment or SyncAugmentPipeline(SyncAugmentConfig())
        self._validate_layout()

    def _validate_layout(self) -> None:
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        if not self.tir_dir.is_dir():
            raise FileNotFoundError(f"TIR directory not found: {self.tir_dir}")
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

    def __len__(self) -> int:
        return self.index.num_images

    def __getitem__(self, index: int) -> dict[str, Tensor | int | str]:
        record = self.index.images[index]
        rgb_path = self.rgb_dir / record.file_name
        tir_path = self.tir_dir / record.file_name
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB image: {rgb_path}")
        if not tir_path.is_file():
            raise FileNotFoundError(f"Missing TIR image: {tir_path}")

        rgb = self._load_image(rgb_path)
        tir = self._load_image(tir_path)
        if rgb.shape[-2:] != tir.shape[-2:]:
            raise ValueError(
                f"RGB/TIR size mismatch for {record.file_name}: "
                f"{tuple(rgb.shape[-2:])} vs {tuple(tir.shape[-2:])}"
            )

        boxes, labels = self._load_targets(record.image_id, record.width, record.height)
        rgb, tir, boxes = self.augment(rgb, tir, boxes, training=self.training)
        boxes, labels = self._filter_boxes(boxes, labels)

        return {
            "rgb": rgb,
            "tir": tir,
            "boxes": boxes,
            "labels": labels,
            "image_id": record.image_id,
            "file_name": record.file_name,
        }

    @staticmethod
    def _load_image(path: Path) -> Tensor:
        with Image.open(path) as image:
            tensor = torch.from_numpy(
                np.array(image.convert("RGB"), copy=True)
            ).permute(2, 0, 1).float()
        return tensor / 255.0

    def _load_targets(
        self,
        image_id: int,
        width: int,
        height: int,
    ) -> tuple[Tensor, Tensor]:
        boxes: list[list[float]] = []
        labels: list[int] = []
        for annotation in self.index.annotations_for(image_id):
            if annotation.iscrowd:
                continue
            if annotation.category_id not in self.valid_category_ids:
                continue
            x1, y1, x2, y2 = annotation.bbox_xyxy
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 - x1 < self.min_box_size or y2 - y1 < self.min_box_size:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(annotation.category_id)

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                (0,), dtype=torch.int64
            )
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.int64
        )

    def _filter_boxes(
        self,
        boxes: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if boxes.numel() == 0:
            return boxes, labels
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= self.min_box_size) & (heights >= self.min_box_size)
        return boxes[keep], labels[keep]
