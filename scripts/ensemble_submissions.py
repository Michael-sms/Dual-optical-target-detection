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
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    input_counts: dict[str, int] = {}

    for input_path in args.inputs:
        records = json.loads(input_path.read_text(encoding="utf-8"))
        input_counts[str(input_path)] = len(records)
        for record in records:
            if float(record["score"]) >= args.score_threshold:
                grouped[(int(record["image_id"]), int(record["category_id"]))].append(record)

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
    print(json.dumps({"inputs": input_counts, "output": str(args.output), "records": len(fused)}))


if __name__ == "__main__":
    main()
