#!/usr/bin/env python3
"""Backfill config copies and training summaries for completed runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from dualdet.utils.run_metadata import (
    build_training_summary,
    collect_git_commit,
    collect_hardware_info,
    infer_best_epoch,
    write_json,
)

EXPERIMENT_CONFIGS = {
    "e0": Path("configs/e0_rgb_baseline.yaml"),
    "e0_rgb_baseline": Path("configs/e0_rgb_baseline.yaml"),
    "e1": Path("configs/e1_fixed_fusion.yaml"),
    "e1_fixed_fusion": Path("configs/e1_fixed_fusion.yaml"),
    "e2": Path("configs/e2_qaf.yaml"),
    "e2_qaf": Path("configs/e2_qaf.yaml"),
    "e3": Path("configs/e3_qaf_p2.yaml"),
    "e3_qaf_p2": Path("configs/e3_qaf_p2.yaml"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Member 2 training artifacts.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Directory containing experiment output folders.",
    )
    return parser.parse_args()


def _resolve_config_path(output_dir: Path) -> Path | None:
    for candidate in (output_dir / "config.yaml", output_dir / "config.yml"):
        if candidate.is_file():
            return candidate

    checkpoint_path = output_dir / "best.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        experiment = str(config.get("experiment", output_dir.name))
        mapped = EXPERIMENT_CONFIGS.get(experiment) or EXPERIMENT_CONFIGS.get(output_dir.name)
        if mapped and mapped.is_file():
            destination = output_dir / "config.yaml"
            shutil.copy2(mapped, destination)
            return destination
    mapped = EXPERIMENT_CONFIGS.get(output_dir.name)
    if mapped and mapped.is_file():
        destination = output_dir / "config.yaml"
        shutil.copy2(mapped, destination)
        return destination
    return None


def _backfill_output_dir(output_dir: Path) -> None:
    history_path = output_dir / "history.json"
    if not history_path.is_file():
        print(f"Skip {output_dir}: history.json not found")
        return

    history = json.loads(history_path.read_text(encoding="utf-8"))
    best_epoch, best_val_loss = infer_best_epoch(history)
    config_path = _resolve_config_path(output_dir)

    checkpoint_path = output_dir / "best.pt"
    parameters: dict[str, int] = {}
    experiment = output_dir.name
    seed = 42
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        experiment = str(config.get("experiment", experiment))
        seed = int(config.get("train", {}).get("seed", seed))
        metrics = checkpoint.get("metrics", {})
        if metrics:
            best_val_loss = float(metrics.get("total", best_val_loss))
        best_epoch = int(checkpoint.get("epoch", best_epoch))

    total_training_time_s = sum(
        float(record.get("epoch_time_s", 0.0)) for record in history
    )
    if total_training_time_s <= 0.0:
        total_training_time_s = float(len(history))

    summary = build_training_summary(
        experiment=experiment,
        config_path=config_path or Path("unknown"),
        output_dir=output_dir,
        seed=seed,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        total_training_time_s=total_training_time_s,
        parameters=parameters,
    )
    write_json(output_dir / "training_summary.json", summary)

    run_metadata = {
        "experiment": experiment,
        "config_path": str(config_path.resolve()) if config_path else None,
        "config_copy": str((output_dir / config_path.name).resolve()) if config_path else None,
        "seed": seed,
        "git_commit": collect_git_commit(),
        "hardware": collect_hardware_info(),
        "backfilled": True,
    }
    write_json(output_dir / "run_metadata.json", run_metadata)
    print(f"Backfilled {output_dir}")


def main() -> None:
    args = _parse_args()
    for output_dir in sorted(path for path in args.output_root.iterdir() if path.is_dir()):
        _backfill_output_dir(output_dir)


if __name__ == "__main__":
    main()
