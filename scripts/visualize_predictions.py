#!/usr/bin/env python3
"""Visualize predictions from a submission JSON on random test images."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from dualdet.utils.submission import load_submission

CATEGORY_COLORS = {
    1: (255, 64, 64),
    2: (64, 160, 255),
    3: (64, 200, 64),
    4: (255, 180, 64),
    5: (180, 64, 255),
}
CATEGORY_NAMES = {
    1: "car",
    2: "truck",
    3: "bus",
    4: "van",
    5: "freight_car",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize submission boxes on test images.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/result.json"),
        help="Submission JSON path.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prediction_visualization.jpg"),
    )
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def _image_path(data_root: Path, split: str, file_name: str, modality: str) -> Path:
    return data_root / split / modality / file_name


def _file_name_for_image_id(data_root: Path, split: str, image_id: int) -> str:
    return f"{image_id:05d}.jpg"


def _draw_predictions(
    image: Image.Image,
    records: list[dict[str, object]],
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for record in records:
        bbox = record["bbox"]
        category_id = int(record["category_id"])
        score = float(record["score"])
        x, y, width, height = bbox
        color = CATEGORY_COLORS.get(category_id, (255, 255, 255))
        draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
        label = CATEGORY_NAMES.get(category_id, str(category_id))
        draw.text((x, max(0.0, y - 12)), f"{label}:{score:.2f}", fill=color)
    return canvas


def main() -> None:
    args = _parse_args()
    records = load_submission(args.input)
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["image_id"])].append(record)

    image_ids = sorted(grouped)
    if not image_ids:
        raise SystemExit("No predictions found in submission file.")

    random.seed(args.seed)
    sample_ids = random.sample(image_ids, k=min(args.num_samples, len(image_ids)))

    tile_width = 640
    tile_height = 512
    pair_height = tile_height * 2
    columns = max(1, args.columns)
    rows = (len(sample_ids) + columns - 1) // columns
    grid = Image.new("RGB", (tile_width * columns, pair_height * rows), color=(24, 24, 24))

    for index, image_id in enumerate(sample_ids):
        file_name = _file_name_for_image_id(args.data_root, args.split, image_id)
        rgb_path = _image_path(args.data_root, args.split, file_name, "rgb")
        tir_path = _image_path(args.data_root, args.split, file_name, "tir")
        rgb = Image.open(rgb_path).convert("RGB")
        tir = Image.open(tir_path).convert("RGB")
        preds = grouped[image_id]
        rgb_vis = _draw_predictions(rgb, preds)
        tir_vis = _draw_predictions(tir, preds)

        row = index // columns
        col = index % columns
        x_offset = col * tile_width
        y_offset = row * pair_height
        grid.paste(rgb_vis, (x_offset, y_offset))
        grid.paste(tir_vis, (x_offset, y_offset + tile_height))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output, quality=95)
    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "num_samples": len(sample_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
