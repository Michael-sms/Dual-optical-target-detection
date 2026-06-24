"""COCO annotation parsing and indexing utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """Single image entry from a COCO annotation file."""

    image_id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class AnnotationRecord:
    """Single bounding-box annotation in absolute xyxy pixel coordinates."""

    image_id: int
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    area: float
    iscrowd: int


@dataclass
class COCOAnnotationIndex:
    """In-memory index built from a COCO JSON annotation file."""

    images: list[ImageRecord]
    annotations_by_image: dict[int, list[AnnotationRecord]]
    categories: list[Mapping[str, object]]
    image_by_id: dict[int, ImageRecord]

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_annotations(self) -> int:
        return sum(len(items) for items in annotations_by_image.values())

    def annotations_for(self, image_id: int) -> list[AnnotationRecord]:
        return self.annotations_by_image.get(image_id, [])


def _coco_bbox_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert COCO ``[x, y, width, height]`` to ``[x1, y1, x2, y2]``."""

    x, y, width, height = bbox
    return (x, y, x + width, y + height)


def load_coco_annotations(annotation_path: str | Path) -> COCOAnnotationIndex:
    """Load and validate a COCO detection annotation file."""

    path = Path(annotation_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images = [
        ImageRecord(
            image_id=int(item["id"]),
            file_name=str(item["file_name"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in payload["images"]
    ]
    image_by_id = {record.image_id: record for record in images}

    annotations_by_image: dict[int, list[AnnotationRecord]] = {}
    for item in payload.get("annotations", []):
        image_id = int(item["image_id"])
        record = AnnotationRecord(
            image_id=image_id,
            category_id=int(item["category_id"]),
            bbox_xyxy=_coco_bbox_to_xyxy(item["bbox"]),
            area=float(item.get("area", 0.0)),
            iscrowd=int(item.get("iscrowd", 0)),
        )
        annotations_by_image.setdefault(image_id, []).append(record)

    categories = list(payload.get("categories", []))
    return COCOAnnotationIndex(
        images=images,
        annotations_by_image=annotations_by_image,
        categories=categories,
        image_by_id=image_by_id,
    )
