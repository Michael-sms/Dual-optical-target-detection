#!/usr/bin/env python3
"""Fine-tune a pretrained Ultralytics YOLOv8 model on the RGB split."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/yolo_rgb/rgb_dataset.yaml"))
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("outputs/ultralytics"))
    parser.add_argument("--name", default="yolov8m_rgb_pretrained")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install -r requirements-ultralytics.txt"
        ) from error

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(args.project.resolve()),
        name=args.name,
        pretrained=True,
        amp=True,
        deterministic=True,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
