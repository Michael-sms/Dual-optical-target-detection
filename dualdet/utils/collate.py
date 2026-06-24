"""Batch collation for variable-length detection targets."""

from __future__ import annotations

from typing import Any

import torch


def detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack image tensors and keep per-sample box/label lists."""

    rgb = torch.stack([item["rgb"] for item in batch], dim=0)
    tir = torch.stack([item["tir"] for item in batch], dim=0)
    targets = [
        {
            "boxes": item["boxes"],
            "labels": item["labels"],
            "image_id": item["image_id"],
        }
        for item in batch
    ]
    return {
        "rgb": rgb,
        "tir": tir,
        "targets": targets,
        "image_ids": [item["image_id"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
    }
