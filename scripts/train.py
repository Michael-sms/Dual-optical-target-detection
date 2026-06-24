#!/usr/bin/env python3
"""Train E0/E1/E2/E3 detectors from YAML configuration files."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dualdet.datasets.rgb_t import RGBTDataset
from dualdet.datasets.transforms import SyncAugmentConfig, SyncAugmentPipeline
from dualdet.losses.detection import DetectionLoss
from dualdet.models.build import build_detector
from dualdet.utils.amp import autocast_context, create_grad_scaler
from dualdet.utils.coco_eval import evaluate_coco_metrics
from dualdet.utils.checkpoint import save_checkpoint
from dualdet.utils.collate import detection_collate
from dualdet.utils.config import load_config
from dualdet.utils.run_metadata import (
    build_training_summary,
    collect_git_commit,
    collect_hardware_info,
    copy_config_file,
    write_json,
)
from dualdet.utils.seed import set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-optical detectors.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _reset_peak_gpu_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gpu_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
    count = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        tir = batch["tir"].to(device)
        with autocast_context(device, enabled=use_amp):
            output = model(rgb, tir)
            loss = criterion(output.predictions, batch["targets"])
        totals["total"] += float(loss.total.item())
        totals["box"] += float(loss.box.item())
        totals["cls"] += float(loss.cls.item())
        totals["dfl"] += float(loss.dfl.item())
        count += 1
    if count == 0:
        return totals
    return {key: value / count for key, value in totals.items()}


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DetectionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
    log_interval: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
    count = 0
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        rgb = batch["rgb"].to(device)
        tir = batch["tir"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, enabled=use_amp):
            output = model(rgb, tir)
            loss = criterion(output.predictions, batch["targets"])

        if use_amp and device.type == "cuda":
            scaler.scale(loss.total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.total.backward()
            optimizer.step()

        totals["total"] += float(loss.total.item())
        totals["box"] += float(loss.box.item())
        totals["cls"] += float(loss.cls.item())
        totals["dfl"] += float(loss.dfl.item())
        count += 1

        if step % log_interval == 0:
            elapsed = time.time() - start
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={totals['total'] / count:.4f} elapsed={elapsed:.1f}s"
            )

    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    config_copy_path = copy_config_file(args.config, output_dir)
    run_metadata = {
        "experiment": config["experiment"],
        "config_path": str(args.config.resolve()),
        "config_copy": str(config_copy_path.resolve()),
        "seed": seed,
        "git_commit": collect_git_commit(),
        "hardware": collect_hardware_info(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output_dir / "run_metadata.json", run_metadata)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    epochs = args.epochs or int(train_cfg["epochs"])
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    use_amp = bool(train_cfg.get("amp", True))
    early_stop_patience = train_cfg.get("early_stop_patience")
    early_stop_min_epochs = int(train_cfg.get("early_stop_min_epochs", 20))
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 1e-4))
    eval_ap = bool(train_cfg.get("eval_ap", True))
    annotation_path = Path(data_cfg["root"]) / data_cfg["val_split"] / f"{data_cfg['val_split']}.json"

    augment = SyncAugmentPipeline(SyncAugmentConfig())
    train_dataset = RGBTDataset(
        data_root=data_cfg["root"],
        split=data_cfg["train_split"],
        training=True,
        min_box_size=float(train_cfg.get("min_box_size", 2.0)),
        augment=augment,
    )
    val_dataset = RGBTDataset(
        data_root=data_cfg["root"],
        split=data_cfg["val_split"],
        training=False,
        min_box_size=float(train_cfg.get("min_box_size", 2.0)),
        augment=augment,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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

    criterion = DetectionLoss(
        num_classes=int(model_cfg["num_classes"]),
        reg_max=int(model_cfg.get("reg_max", 16)),
        feature_levels=model.feature_levels,
        strides=model.level_strides,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0005)),
    )
    scheduler = _build_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=int(train_cfg.get("warmup_epochs", 3)),
    )
    scaler = create_grad_scaler(device, enabled=use_amp)

    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    param_breakdown = model.parameter_breakdown()
    print(json.dumps({"parameters": param_breakdown}, indent=2))

    history: list[dict[str, object]] = []
    best_val = float("inf")
    best_ap = -1.0
    best_epoch = 0
    best_val_ap = 0.0
    epochs_without_improvement = 0
    selection_metric = "val.ap" if eval_ap else "val.total"
    training_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        _reset_peak_gpu_memory(device)
        epoch_start = time.time()

        train_metrics = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=use_amp,
            log_interval=int(train_cfg.get("log_interval", 20)),
            epoch=epoch,
        )
        val_metrics = _evaluate(
            model, val_loader, criterion, device, use_amp=use_amp
        )
        if eval_ap:
            ap_metrics = evaluate_coco_metrics(
                model,
                val_loader,
                annotation_path,
                device,
                num_classes=int(model_cfg["num_classes"]),
                reg_max=int(model_cfg.get("reg_max", 16)),
                feature_levels=model.feature_levels,
                strides=model.level_strides,
                use_amp=use_amp,
                conf_threshold=float(train_cfg.get("conf_threshold", 0.001)),
                iou_threshold=float(train_cfg.get("nms_iou_threshold", 0.7)),
                max_detections=int(train_cfg.get("max_detections", 300)),
            )
            val_metrics.update(ap_metrics)
        scheduler.step()

        epoch_time_s = round(time.time() - epoch_start, 2)
        record: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": epoch_time_s,
            "peak_gpu_mem_mb": _peak_gpu_memory_mb(device),
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        last_path = output_dir / "last.pt"
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            config=config,
        )
        improved = False
        if eval_ap and "ap" in val_metrics and val_metrics["ap"] > best_ap + early_stop_min_delta:
            best_ap = float(val_metrics["ap"])
            best_val_ap = best_ap
            best_epoch = epoch
            epochs_without_improvement = 0
            improved = True
        elif not eval_ap and val_metrics["total"] < best_val - early_stop_min_delta:
            best_val = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            improved = True

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]

        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
            )
        else:
            epochs_without_improvement += 1

        history_path = output_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if (
            early_stop_patience is not None
            and epoch >= early_stop_min_epochs
            and epochs_without_improvement >= int(early_stop_patience)
        ):
            print(
                f"Early stopping at epoch {epoch}: "
                f"no validation improvement for {early_stop_patience} epochs "
                f"on {selection_metric}."
            )
            break

    total_training_time_s = time.time() - training_start
    summary = build_training_summary(
        experiment=str(config["experiment"]),
        config_path=args.config,
        output_dir=output_dir,
        seed=seed,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        total_training_time_s=total_training_time_s,
        parameters=param_breakdown,
        selection_metric=selection_metric,
        best_val_ap=best_val_ap if eval_ap else None,
    )
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Training finished. Artifacts saved under {output_dir}")


if __name__ == "__main__":
    main()
