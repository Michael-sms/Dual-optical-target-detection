#!/usr/bin/env python3
"""Convert train/val COCO annotations and RGB or TIR images to YOLO format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dualdet.utils.yolo_conversion import convert_modality_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--modality", choices=("rgb", "tir"), default="rgb")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <data-root>/yolo_<modality>.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="Hard links avoid duplicating the image dataset and are recommended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.data_root / f"yolo_{args.modality}"
    report = convert_modality_dataset(
        args.data_root,
        output_root,
        modality=args.modality,
        image_mode=args.image_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
