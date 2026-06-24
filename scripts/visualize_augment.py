#!/usr/bin/env python3
"""Visualize synchronized RGB-T augmentation with bounding boxes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from dualdet.datasets.rgb_t import RGBTDataset
from dualdet.datasets.transforms import SyncAugmentConfig, SyncAugmentPipeline
from dualdet.utils.seed import set_seed

CATEGORY_COLORS = {
    1: (255, 64, 64),
    2: (64, 160, 255),
    3: (64, 200, 64),
    4: (255, 180, 64),
    5: (180, 64, 255),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw augmented RGB/TIR pairs with synchronized boxes."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/augment_visualization.jpg"),
    )
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def _draw_boxes(image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box, label in zip(boxes.tolist(), labels.tolist(), strict=True):
        color = CATEGORY_COLORS.get(int(label), (255, 255, 0))
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0] + 2, max(0, box[1] - 12)), str(int(label)), fill=color)
    return canvas


def _compose_tile(rgb: Image.Image, tir: Image.Image) -> Image.Image:
    tile = Image.new("RGB", (rgb.width + tir.width, rgb.height))
    tile.paste(rgb, (0, 0))
    tile.paste(tir, (rgb.width, 0))
    return tile


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    dataset = RGBTDataset(
        data_root=args.data_root,
        split=args.split,
        training=True,
        augment=SyncAugmentPipeline(SyncAugmentConfig()),
    )
    if len(dataset) < args.num_samples:
        raise ValueError(
            f"Split {args.split!r} has only {len(dataset)} samples, "
            f"but {args.num_samples} were requested."
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num_samples].tolist()

    tiles: list[Image.Image] = []
    for index in indices:
        sample = dataset[index]
        rgb = _draw_boxes(_tensor_to_image(sample["rgb"]), sample["boxes"], sample["labels"])
        tir = _draw_boxes(_tensor_to_image(sample["tir"]), sample["boxes"], sample["labels"])
        tiles.append(_compose_tile(rgb, tir))

    columns = max(1, args.columns)
    rows = math.ceil(len(tiles) / columns)
    tile_width, tile_height = tiles[0].size
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(24, 24, 24))

    for tile_index, tile in enumerate(tiles):
        row = tile_index // columns
        column = tile_index % columns
        grid.paste(tile, (column * tile_width, row * tile_height))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output, quality=95)
    print(f"Saved {len(tiles)} augmented RGB|TIR samples to {args.output}")


if __name__ == "__main__":
    main()
