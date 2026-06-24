#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dualdet.datasets.rgb_t import RGBTDataset
from dualdet.datasets.transforms import SyncAugmentConfig, SyncAugmentPipeline
from dualdet.models.build import build_detector
from dualdet.utils.coco_eval import evaluate_coco_metrics
from dualdet.utils.collate import detection_collate
from dualdet.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector checkpoint on validation set.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    augment = SyncAugmentPipeline(SyncAugmentConfig())
    val_dataset = RGBTDataset(
        data_root=data_cfg["root"],
        split=data_cfg["val_split"],
        training=False,
        augment=augment,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
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

    annotation_path = Path(data_cfg["root"]) / data_cfg["val_split"] / f"{data_cfg['val_split']}.json"
    metrics = evaluate_coco_metrics(
        model,
        val_loader,
        annotation_path,
        device,
        num_classes=int(model_cfg["num_classes"]),
        reg_max=int(model_cfg.get("reg_max", 16)),
        feature_levels=model.feature_levels,
        strides=model.level_strides,
        use_amp=bool(train_cfg.get("amp", True)),
        conf_threshold=float(train_cfg.get("conf_threshold", 0.001)),
        iou_threshold=float(train_cfg.get("nms_iou_threshold", 0.7)),
        max_detections=int(train_cfg.get("max_detections", 300)),
    )
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": int(checkpoint.get("epoch", 0)),
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2))
    output_path = output_dir / "eval_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
