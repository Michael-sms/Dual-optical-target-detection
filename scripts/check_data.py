#!/usr/bin/env python3
"""Verify paired RGB-T dataset layout and print summary statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from dualdet.datasets.coco import load_coco_annotations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check RGB-T dataset integrity.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_check_report.json"),
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--max-dimension-errors",
        type=int,
        default=20,
        help="Maximum dimension mismatch examples to include in the report.",
    )
    return parser.parse_args()


def _validate_image_pair(
    rgb_path: Path,
    tir_path: Path,
    *,
    expected_width: int | None,
    expected_height: int | None,
) -> dict[str, object] | None:
    """Return a mismatch record when RGB/TIR dimensions are invalid."""

    if not rgb_path.is_file():
        return {"issue": "missing_rgb", "file_name": rgb_path.name}
    if not tir_path.is_file():
        return {"issue": "missing_tir", "file_name": tir_path.name}

    with Image.open(rgb_path) as rgb_image, Image.open(tir_path) as tir_image:
        rgb_size = rgb_image.size
        tir_size = tir_image.size

    if rgb_size != tir_size:
        return {
            "issue": "rgb_tir_size_mismatch",
            "file_name": rgb_path.name,
            "rgb_size": list(rgb_size),
            "tir_size": list(tir_size),
        }

    width, height = rgb_size
    if expected_width is not None and expected_height is not None:
        if width != expected_width or height != expected_height:
            return {
                "issue": "coco_size_mismatch",
                "file_name": rgb_path.name,
                "image_size": [width, height],
                "coco_size": [expected_width, expected_height],
            }

    return None


def _check_split(
    data_root: Path,
    split: str,
    *,
    max_dimension_errors: int,
) -> dict[str, object]:
    rgb_dir = data_root / split / "rgb"
    tir_dir = data_root / split / "tir"
    annotation_path = data_root / split / f"{split}.json"

    rgb_files = sorted(path.name for path in rgb_dir.glob("*.jpg"))
    tir_files = sorted(path.name for path in tir_dir.glob("*.jpg"))
    missing_tir = sorted(set(rgb_files) - set(tir_files))
    missing_rgb = sorted(set(tir_files) - set(rgb_files))
    if missing_tir or missing_rgb:
        raise RuntimeError(
            f"{split}: RGB/TIR pairing mismatch. "
            f"missing_tir={len(missing_tir)}, missing_rgb={len(missing_rgb)}"
        )

    report: dict[str, object] = {
        "split": split,
        "rgb_count": len(rgb_files),
        "tir_count": len(tir_files),
        "paired": len(rgb_files) == len(tir_files),
    }

    dimension_errors: list[dict[str, object]] = []
    dimension_error_count = 0
    missing_annotation_files = 0

    if annotation_path.is_file():
        index = load_coco_annotations(annotation_path)
        category_counter: Counter[int] = Counter()
        small_objects = 0
        total_boxes = 0
        for record in index.images:
            rgb_path = rgb_dir / record.file_name
            tir_path = tir_dir / record.file_name
            mismatch = _validate_image_pair(
                rgb_path,
                tir_path,
                expected_width=record.width,
                expected_height=record.height,
            )
            if mismatch is not None:
                if mismatch["issue"] in {"missing_rgb", "missing_tir"}:
                    missing_annotation_files += 1
                else:
                    dimension_error_count += 1
                if len(dimension_errors) < max_dimension_errors:
                    mismatch["image_id"] = record.image_id
                    dimension_errors.append(mismatch)

        for annotations in index.annotations_by_image.values():
            for item in annotations:
                if item.iscrowd:
                    continue
                category_counter[item.category_id] += 1
                total_boxes += 1
                x1, y1, x2, y2 = item.bbox_xyxy
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area < 32 * 32:
                    small_objects += 1

        report.update(
            {
                "annotated_images": index.num_images,
                "annotation_count": total_boxes,
                "category_counts": dict(sorted(category_counter.items())),
                "small_object_ratio": (
                    round(small_objects / total_boxes, 4) if total_boxes else 0.0
                ),
                "categories": index.categories,
                "missing_annotation_files": missing_annotation_files,
                "dimension_error_count": len(dimension_errors),
                "dimension_errors": dimension_errors,
            }
        )
    else:
        for file_name in rgb_files:
            mismatch = _validate_image_pair(
                rgb_dir / file_name,
                tir_dir / file_name,
                expected_width=None,
                expected_height=None,
            )
            if mismatch is not None:
                dimension_error_count += 1
                if len(dimension_errors) < max_dimension_errors:
                    dimension_errors.append(mismatch)
        report.update(
            {
                "annotated_images": len(rgb_files),
                "annotation_count": 0,
                "dimension_error_count": len(dimension_errors),
                "dimension_errors": dimension_errors,
            }
        )

    if report.get("missing_annotation_files", 0):
        raise RuntimeError(
            f"{split}: {report['missing_annotation_files']} COCO entries are missing RGB/TIR files."
        )
    if report.get("dimension_error_count", 0):
        raise RuntimeError(
            f"{split}: found {report['dimension_error_count']} dimension mismatches."
        )

    return report


def main() -> None:
    args = _parse_args()
    splits = ["train", "val", "test"]
    reports = [
        _check_split(
            args.data_root,
            split,
            max_dimension_errors=args.max_dimension_errors,
        )
        for split in splits
    ]
    summary = {
        "data_root": str(args.data_root.resolve()),
        "splits": reports,
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
