#!/usr/bin/env python3
"""Validate competition submission JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dualdet.datasets.rgb_t import RGBTTestDataset
from dualdet.utils.submission import load_submission, validate_submission


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate result.json submission format.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/result.json"),
        help="Submission JSON path.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root used to infer allowed test image ids.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split used to infer allowed image ids.",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=1.0,
        help="Flag duplicate boxes when IoU >= threshold (1.0 checks exact duplicates only).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = load_submission(args.input)
    valid_image_ids = None
    test_root = args.data_root / args.split / "rgb"
    if test_root.is_dir():
        dataset = RGBTTestDataset(data_root=args.data_root, split=args.split)
        valid_image_ids = {record.image_id for record in dataset.records}

    errors = validate_submission(
        records,
        valid_image_ids=valid_image_ids,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
    )
    payload = {
        "input": str(args.input.resolve()),
        "num_records": len(records),
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
