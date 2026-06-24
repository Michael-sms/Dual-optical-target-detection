#!/usr/bin/env python3
"""Plot training loss curves from experiment history.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS: dict[str, Path] = {
    "E0 RGB Baseline": ROOT / "outputs" / "e0_rgb_baseline" / "history.json",
    "E1 Fixed Fusion": ROOT / "outputs" / "e1_fixed_fusion" / "history.json",
    "E2 QAF": ROOT / "outputs" / "e2_qaf" / "history.json",
    "E3 QAF + P2": ROOT / "outputs" / "e3_qaf_p2" / "history.json",
}

LOSS_KEYS = ("total", "box", "cls", "dfl")
LOSS_TITLES = {
    "total": "Total Loss",
    "box": "Box Loss (CIoU)",
    "cls": "Classification Loss",
    "dfl": "DFL Loss",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curves from training history.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "plots",
        help="Directory to save loss figures.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "train", "both"),
        default="val",
        help="Which split to plot (default: val).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution.",
    )
    return parser.parse_args()


def _load_histories(
    experiments: dict[str, Path],
) -> dict[str, list[dict[str, object]]]:
    loaded: dict[str, list[dict[str, object]]] = {}
    for label, path in experiments.items():
        if not path.is_file():
            print(f"Skip missing history: {path}")
            continue
        loaded[label] = json.loads(path.read_text(encoding="utf-8"))
    if not loaded:
        raise FileNotFoundError("No history.json files found for plotting.")
    return loaded


def _series(
    history: list[dict[str, object]],
    split: str,
    key: str,
) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for record in history:
        split_metrics = record.get(split)
        if not isinstance(split_metrics, dict) or key not in split_metrics:
            continue
        epochs.append(int(record["epoch"]))
        values.append(float(split_metrics[key]))
    return epochs, values


def _plot_one_loss(
    histories: dict[str, list[dict[str, object]]],
    loss_key: str,
    *,
    split: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, history in histories.items():
        if split == "both":
            train_epochs, train_values = _series(history, "train", loss_key)
            val_epochs, val_values = _series(history, "val", loss_key)
            ax.plot(
                train_epochs,
                train_values,
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label=f"{label} (train)",
            )
            ax.plot(
                val_epochs,
                val_values,
                linewidth=2.0,
                label=f"{label} (val)",
            )
        else:
            epochs, values = _series(history, split, loss_key)
            ax.plot(epochs, values, linewidth=2.0, marker="o", markersize=3, label=label)

    ax.set_title(LOSS_TITLES[loss_key], fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    histories = _load_histories(DEFAULT_EXPERIMENTS)

    for loss_key in LOSS_KEYS:
        output_path = args.output_dir / f"loss_{loss_key}.png"
        _plot_one_loss(
            histories,
            loss_key,
            split=args.split,
            output_path=output_path,
            dpi=args.dpi,
        )
        print(f"Saved {output_path}")

    combined_path = args.output_dir / "loss_all.png"
    _plot_combined(histories, split=args.split, output_path=combined_path, dpi=args.dpi)
    print(f"Saved {combined_path}")


def _plot_combined(
    histories: dict[str, list[dict[str, object]]],
    *,
    split: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for axis, loss_key in zip(axes_flat, LOSS_KEYS, strict=True):
        for label, history in histories.items():
            if split == "both":
                train_epochs, train_values = _series(history, "train", loss_key)
                val_epochs, val_values = _series(history, "val", loss_key)
                axis.plot(
                    train_epochs,
                    train_values,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                    label=f"{label} (train)",
                )
                axis.plot(val_epochs, val_values, linewidth=1.8, label=f"{label} (val)")
            else:
                epochs, values = _series(history, split, loss_key)
                axis.plot(epochs, values, linewidth=1.8, marker="o", markersize=2.5, label=label)

        axis.set_title(LOSS_TITLES[loss_key], fontsize=11, fontweight="bold")
        axis.set_xlabel("Epoch", fontsize=10)
        axis.set_ylabel("Loss", fontsize=10)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(loc="best", fontsize=7)

    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
