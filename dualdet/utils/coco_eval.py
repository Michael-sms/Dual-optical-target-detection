"""COCO AP evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dualdet.datasets.coco import load_coco_annotations
from dualdet.utils.amp import autocast_context
from dualdet.utils.postprocess import (
    DEFAULT_FEATURE_LEVELS,
    DEFAULT_STRIDES,
    decode_batch_predictions,
    detections_to_coco_records,
)


def _empty_metrics() -> dict[str, float]:
    return {
        "ap": 0.0,
        "ap50": 0.0,
        "ap75": 0.0,
        "ap_small": 0.0,
        "ap_medium": 0.0,
        "ap_large": 0.0,
    }


def _summarize_coco_eval(coco_eval: Any) -> dict[str, float]:
    stats = coco_eval.stats
    return {
        "ap": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
    }


@torch.no_grad()
def evaluate_coco_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    annotation_path: str | Path,
    device: torch.device,
    *,
    num_classes: int,
    reg_max: int,
    feature_levels: tuple[str, ...] = DEFAULT_FEATURE_LEVELS,
    strides: dict[str, int] | None = None,
    use_amp: bool = True,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    input_height: int = 512,
    input_width: int = 640,
) -> dict[str, float]:
    """Run validation inference and compute COCO AP metrics."""

    try:
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for AP evaluation. Install with `pip install pycocotools`."
        ) from exc

    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    index = load_coco_annotations(annotation_path)
    if index.num_images == 0:
        return _empty_metrics()

    model.eval()
    predictions: list[dict[str, float | int | list[float]]] = []
    stride_map = strides or DEFAULT_STRIDES

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
            strides=stride_map,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        for sample_index, detections in enumerate(batch_predictions):
            image_id = int(batch["targets"][sample_index]["image_id"])
            predictions.extend(
                detections_to_coco_records(
                    detections,
                    image_id=image_id,
                    input_height=input_height,
                    input_width=input_width,
                )
            )

    if not predictions:
        return _empty_metrics()

    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for AP evaluation. Install with `pip install pycocotools`."
        ) from exc

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return _summarize_coco_eval(coco_eval)
