#!/usr/bin/env python3
"""Train E0-E3 sequentially on GPU with a single command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"

EXPERIMENT_CONFIGS: dict[str, Path] = {
    "e0": ROOT / "configs" / "e0_rgb_baseline.yaml",
    "e1": ROOT / "configs" / "e1_fixed_fusion.yaml",
    "e2": ROOT / "configs" / "e2_qaf.yaml",
    "e3": ROOT / "configs" / "e3_qaf_p2.yaml",
}
DEFAULT_ORDER = ("e0", "e1", "e2", "e3")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPU training for E0-E3 in one command."
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(DEFAULT_ORDER),
        help="Comma-separated experiment ids: e0,e1,e2,e3 (default: all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device passed to scripts/train.py (default: cuda).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running training.",
    )
    return parser.parse_args()


def _resolve_experiments(raw: str) -> list[str]:
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one experiment id is required.")

    unknown = [name for name in names if name not in EXPERIMENT_CONFIGS]
    if unknown:
        supported = ", ".join(DEFAULT_ORDER)
        raise ValueError(f"Unknown experiment(s): {', '.join(unknown)}. Supported: {supported}")

    return names


def _build_command(
    config_path: Path,
    *,
    device: str,
    epochs: int | None,
    batch_size: int | None,
) -> list[str]:
    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path), "--device", device]
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    return command


def main() -> None:
    args = _parse_args()
    experiments = _resolve_experiments(args.experiments)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use --device cpu or install GPU PyTorch.", file=sys.stderr)
        sys.exit(1)

    if args.device == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")

    for index, experiment in enumerate(experiments, start=1):
        config_path = EXPERIMENT_CONFIGS[experiment]
        if not config_path.is_file():
            print(f"Missing config for {experiment}: {config_path}", file=sys.stderr)
            sys.exit(1)

        command = _build_command(
            config_path,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(f"[{index}/{len(experiments)}] Training {experiment}: {' '.join(command)}")

        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"Training failed for {experiment} with exit code {result.returncode}.", file=sys.stderr)
            sys.exit(result.returncode)

    if args.dry_run:
        print("Dry run finished. No training was started.")
    else:
        print(f"All requested experiments finished: {', '.join(experiments)}")


if __name__ == "__main__":
    main()
