#!/usr/bin/env python3
"""Fuse COCO-style submission JSON files with class-wise NMS."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional confidence multiplier for each input.",
    )
    parser.add_argument(
        "--input-score-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Optional unscaled confidence threshold for each input.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.002)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-detections", type=int, default=300)
    return parser.parse_args()


def xywh_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lw, lh = (float(value) for value in left)
    rx1, ry1, rw, rh = (float(value) for value in right)
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0.0 else 0.0


def classwise_nms(records: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: float(record["score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        if all(
            xywh_iou(candidate["bbox"], selected["bbox"]) < iou_threshold
            for selected in kept
        ):
            kept.append(candidate)
    return kept


def main() -> None:
    args = parse_args()
    weights = args.weights or [1.0] * len(args.inputs)
    input_thresholds = args.input_score_thresholds or [args.score_threshold] * len(args.inputs)
    if len(weights) != len(args.inputs):
        raise ValueError("--weights must contain one value per input")
    if len(input_thresholds) != len(args.inputs):
        raise ValueError("--input-score-thresholds must contain one value per input")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("input weights must be positive")
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    input_counts: dict[str, int] = {}

    for input_path, weight, input_threshold in zip(
        args.inputs, weights, input_thresholds, strict=True
    ):
        records = json.loads(input_path.read_text(encoding="utf-8"))
        input_counts[str(input_path)] = len(records)
        for record in records:
            original_score = float(record["score"])
            scaled_score = original_score * weight
            if original_score >= input_threshold and scaled_score >= args.score_threshold:
                weighted_record = dict(record)
                weighted_record["score"] = round(min(scaled_score, 1.0), 6)
                grouped[(int(record["image_id"]), int(record["category_id"]))].append(
                    weighted_record
                )

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (image_id, _), records in grouped.items():
        by_image[image_id].extend(classwise_nms(records, args.nms_iou))

    fused: list[dict[str, Any]] = []
    for image_id in sorted(by_image):
        ranked = sorted(
            by_image[image_id],
            key=lambda record: float(record["score"]),
            reverse=True,
        )
        fused.extend(ranked[: args.max_detections])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fused, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "inputs": input_counts,
                "weights": weights,
                "input_score_thresholds": input_thresholds,
                "output": str(args.output),
                "records": len(fused),
            }
        )
    )


if __name__ == "__main__":
    main()
