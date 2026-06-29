#!/usr/bin/env python3
"""Export Ultralytics single-modality predictions to competition COCO JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dualdet.utils.submission import filename_to_image_id, validate_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("data/yolo_rgb/images/test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/result_yolov8m_rgb.json"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _records_from_result(result: Any) -> list[dict[str, Any]]:
    image_id = filename_to_image_id(Path(result.path).name)
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    records: list[dict[str, Any]] = []
    for box, score, class_index in zip(xyxy, scores, classes, strict=True):
        x1, y1, x2, y2 = (float(value) for value in box)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            continue
        records.append(
            {
                "image_id": image_id,
                "category_id": int(class_index) + 1,
                "bbox": [round(x1, 1), round(y1, 1), round(width, 1), round(height, 1)],
                "score": round(float(score), 6),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install -r requirements-ultralytics.txt"
        ) from error

    source_images = sorted(args.source.glob("*.jpg"))
    if not source_images:
        raise FileNotFoundError(f"no test images found under {args.source}")
    valid_image_ids = {filename_to_image_id(path.name) for path in source_images}

    model = YOLO(str(args.checkpoint))
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        stream=True,
        verbose=False,
        save=False,
    )
    records: list[dict[str, Any]] = []
    for result in results:
        records.extend(_records_from_result(result))
    records.sort(key=lambda item: (int(item["image_id"]), -float(item["score"])))

    errors = validate_submission(records, valid_image_ids=valid_image_ids)
    if errors:
        raise ValueError("submission validation failed: " + "; ".join(errors[:10]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    size_bytes = args.output.stat().st_size
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "source_images": len(source_images),
                "predictions": len(records),
                "output": str(args.output.resolve()),
                "output_bytes": size_bytes,
                "under_5mb_decimal": size_bytes <= 5_000_000,
            },
            indent=2,
        )
    )
    if size_bytes > 5_000_000:
        print("WARNING: output exceeds 5,000,000 bytes; raise --conf before website upload.")


if __name__ == "__main__":
    main()
