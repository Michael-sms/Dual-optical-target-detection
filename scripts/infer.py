#!/usr/bin/env python3
"""Run inference on the test split and export competition result.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dualdet.datasets.rgb_t import RGBTTestDataset
from dualdet.datasets.transforms import SyncAugmentConfig, SyncAugmentPipeline
from dualdet.models.build import build_detector
from dualdet.utils.collate import detection_collate
from dualdet.utils.config import load_config
from dualdet.utils.inference import run_inference
from dualdet.utils.submission import save_submission, validate_submission


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export COCO-style detection results for the test split."
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (best.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/result.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override config data root.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split name.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip automatic submission validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    data_root = args.data_root or Path(data_cfg["root"])

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    augment = SyncAugmentPipeline(SyncAugmentConfig())
    test_dataset = RGBTTestDataset(
        data_root=data_root,
        split=args.split,
        augment=augment,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
    )

    model = build_detector(
        config["experiment"],
        num_classes=int(model_cfg["num_classes"]),
        width_multiple=float(model_cfg.get("width_multiple", 0.5)),
        depth_multiple=float(model_cfg.get("depth_multiple", 0.33)),
        max_channels=int(model_cfg.get("max_channels", 512)),
        reg_max=int(model_cfg.get("reg_max", 16)),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    records = run_inference(
        model,
        test_loader,
        device,
        num_classes=int(model_cfg["num_classes"]),
        reg_max=int(model_cfg.get("reg_max", 16)),
        feature_levels=model.feature_levels,
        strides=model.level_strides,
        use_amp=bool(train_cfg.get("amp", True)),
        conf_threshold=float(train_cfg.get("conf_threshold", 0.001)),
        iou_threshold=float(train_cfg.get("nms_iou_threshold", 0.7)),
        max_detections=int(train_cfg.get("max_detections", 300)),
        input_height=augment.config.input_height,
        input_width=augment.config.input_width,
    )
    save_submission(records, args.output)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": int(checkpoint.get("epoch", 0)),
        "experiment": config["experiment"],
        "num_images": len(test_dataset),
        "num_predictions": len(records),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))

    if not args.skip_validate:
        valid_image_ids = {record.image_id for record in test_dataset.records}
        errors = validate_submission(
            records,
            valid_image_ids=valid_image_ids,
            duplicate_iou_threshold=1.0,
        )
        if errors:
            print("Submission validation failed:")
            for message in errors[:20]:
                print(f"- {message}")
            if len(errors) > 20:
                print(f"... and {len(errors) - 20} more issues")
            raise SystemExit(1)
        print("Submission validation passed.")


if __name__ == "__main__":
    main()
