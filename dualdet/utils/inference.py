"""Run batched inference and build competition submission records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dualdet.utils.amp import autocast_context
from dualdet.utils.postprocess import decode_batch_predictions, detections_to_coco_records


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    reg_max: int,
    feature_levels: tuple[str, ...],
    strides: dict[str, int],
    use_amp: bool = True,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    input_height: int = 512,
    input_width: int = 640,
) -> list[dict[str, Any]]:
    """Decode model outputs into competition submission records."""

    model.eval()
    records: list[dict[str, Any]] = []

    for batch in loader:
        rgb = batch["rgb"].to(device)
        tir = batch["tir"].to(device)
        with autocast_context(device, enabled=use_amp):
            output = model(rgb, tir)
        batch_predictions = decode_batch_predictions(
            output.predictions,
            num_classes=num_classes,
            reg_max=reg_max,
            feature_levels=feature_levels,
            strides=strides,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        for sample_index, detections in enumerate(batch_predictions):
            image_id = int(batch["targets"][sample_index]["image_id"])
            records.extend(
                detections_to_coco_records(
                    detections,
                    image_id=image_id,
                    input_height=input_height,
                    input_width=input_width,
                )
            )

    return records
