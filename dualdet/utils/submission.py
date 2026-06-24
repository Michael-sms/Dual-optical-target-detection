"""Competition submission JSON helpers and validation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

VALID_CATEGORY_IDS = frozenset({1, 2, 3, 4, 5})
REQUIRED_FIELDS = frozenset({"image_id", "category_id", "bbox", "score"})
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512


def filename_to_image_id(file_name: str) -> int:
    """Convert ``00001.jpg`` to integer image id ``1``."""

    stem = Path(file_name).stem
    if not stem.isdigit():
        raise ValueError(f"file name stem must be numeric for image_id mapping: {file_name!r}")
    return int(stem)


def load_submission(path: str | Path) -> list[dict[str, Any]]:
    """Load a submission JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("submission root must be a JSON list")
    return payload


def _format_score(score: float) -> str:
    """Format confidence as a compact JSON float."""

    if not math.isfinite(score):
        raise ValueError("score must be finite")
    text = f"{score:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _format_record_lines(
    record: Mapping[str, Any],
    *,
    is_first: bool,
    is_last: bool,
) -> list[str]:
    """Format one submission record into exactly six lines."""

    image_id = int(record["image_id"])
    category_id = int(record["category_id"])
    bbox = record["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must contain four numeric values")
    x, y, width, height = (round(float(value), 1) for value in bbox)
    score = _format_score(float(record["score"]))

    open_line = "[{" if is_first else "{"
    if is_last:
        close_line = "}]"
    else:
        close_line = "},"

    return [
        open_line,
        f'"image_id": {image_id},',
        f'"category_id": {category_id},',
        f'"bbox": [{x},{y},{width},{height}],',
        f'"score": {score}',
        close_line,
    ]


def format_submission_text(records: Sequence[Mapping[str, Any]]) -> str:
    """Render submission records using the competition's six-line layout."""

    if not records:
        return "[]"

    lines: list[str] = []
    total = len(records)
    for index, record in enumerate(records):
        lines.extend(
            _format_record_lines(
                record,
                is_first=index == 0,
                is_last=index == total - 1,
            )
        )
    return "\n".join(lines) + "\n"


def save_submission(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> None:
    """Write submission records to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_submission_text(records), encoding="utf-8")


def validate_submission(
    records: Sequence[Mapping[str, Any]],
    *,
    valid_image_ids: Iterable[int] | None = None,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    duplicate_iou_threshold: float = 1.0,
) -> list[str]:
    """Validate submission format and return human-readable error messages."""

    errors: list[str] = []
    allowed_ids = set(valid_image_ids) if valid_image_ids is not None else None
    seen_exact: set[tuple[int, int, tuple[float, float, float, float], float]] = set()

    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{prefix}: must be an object")
            continue

        extra = set(record.keys()) - REQUIRED_FIELDS
        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            errors.append(f"{prefix}: missing fields {sorted(missing)}")
        if extra:
            errors.append(f"{prefix}: unexpected fields {sorted(extra)}")

        image_id = record.get("image_id")
        category_id = record.get("category_id")
        bbox = record.get("bbox")
        score = record.get("score")

        if not isinstance(image_id, int) or isinstance(image_id, bool):
            errors.append(f"{prefix}: image_id must be int, got {type(image_id).__name__}")
        elif allowed_ids is not None and image_id not in allowed_ids:
            errors.append(f"{prefix}: image_id {image_id} is not in allowed test ids")

        if not isinstance(category_id, int) or isinstance(category_id, bool):
            errors.append(f"{prefix}: category_id must be int, got {type(category_id).__name__}")
        elif category_id not in VALID_CATEGORY_IDS:
            errors.append(f"{prefix}: invalid category_id {category_id}")

        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{prefix}: bbox must be a list of length 4")
        else:
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox):
                errors.append(f"{prefix}: bbox values must be numeric")
            else:
                x, y, width, height = (float(value) for value in bbox)
                if not all(math.isfinite(value) for value in (x, y, width, height)):
                    errors.append(f"{prefix}: bbox must contain finite numbers")
                if x < 0.0 or y < 0.0:
                    errors.append(f"{prefix}: bbox x/y must be >= 0")
                if width <= 0.0 or height <= 0.0:
                    errors.append(f"{prefix}: bbox width/height must be > 0")
                if x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
                    errors.append(
                        f"{prefix}: bbox exceeds image bounds "
                        f"({image_width}x{image_height})"
                    )

        if not isinstance(score, (int, float)) or isinstance(score, bool):
            errors.append(f"{prefix}: score must be numeric, got {type(score).__name__}")
        elif not math.isfinite(float(score)):
            errors.append(f"{prefix}: score must be finite")
        elif not 0.0 <= float(score) <= 1.0:
            errors.append(f"{prefix}: score must be within [0, 1]")

        if (
            isinstance(image_id, int)
            and not isinstance(image_id, bool)
            and isinstance(category_id, int)
            and not isinstance(category_id, bool)
            and isinstance(bbox, list)
            and len(bbox) == 4
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox)
        ):
            key = (
                int(image_id),
                int(category_id),
                tuple(round(float(value), 1) for value in bbox),
                round(float(score), 6),
            )
            if key in seen_exact:
                errors.append(f"{prefix}: exact duplicate prediction detected")
            seen_exact.add(key)

    if duplicate_iou_threshold < 1.0:
        errors.extend(
            _find_duplicate_boxes(records, iou_threshold=duplicate_iou_threshold)
        )

    return errors


def _find_duplicate_boxes(
    records: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> list[str]:
    """Flag near-identical boxes on the same image and category."""

    grouped: dict[tuple[int, int], list[tuple[int, list[float]]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        image_id = record.get("image_id")
        category_id = record.get("category_id")
        bbox = record.get("bbox")
        if (
            not isinstance(image_id, int)
            or isinstance(image_id, bool)
            or not isinstance(category_id, int)
            or isinstance(category_id, bool)
            or not isinstance(bbox, list)
            or len(bbox) != 4
        ):
            continue
        grouped.setdefault((int(image_id), int(category_id)), []).append(
            (index, [float(value) for value in bbox])
        )

    errors: list[str] = []
    for (image_id, category_id), items in grouped.items():
        for left in range(len(items)):
            index_a, bbox_a = items[left]
            box_a = _xywh_to_xyxy(bbox_a)
            for right in range(left + 1, len(items)):
                index_b, bbox_b = items[right]
                box_b = _xywh_to_xyxy(bbox_b)
                if _bbox_iou(box_a, box_b) >= iou_threshold:
                    errors.append(
                        "record[{index_a}] and record[{index_b}] are duplicate-like "
                        f"boxes on image_id={image_id}, category_id={category_id}".format(
                            index_a=index_a,
                            index_b=index_b,
                            image_id=image_id,
                            category_id=category_id,
                        )
                    )
    return errors


def _xywh_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = bbox
    return (x, y, x + width, y + height)


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union
