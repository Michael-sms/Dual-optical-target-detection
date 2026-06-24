"""Training run metadata helpers for reproducibility records."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def collect_git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit hash when available."""

    root = repo_root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_hardware_info() -> dict[str, Any]:
    """Capture runtime hardware and library versions."""

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return info


def copy_config_file(config_path: Path, output_dir: Path) -> Path:
    """Copy the experiment config into the output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"config{config_path.suffix}"
    shutil.copy2(config_path, destination)
    return destination


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_training_summary(
    *,
    experiment: str,
    config_path: Path,
    output_dir: Path,
    seed: int,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_val_loss: float,
    total_training_time_s: float,
    parameters: dict[str, int],
    selection_metric: str = "val.ap",
    best_val_ap: float | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build the final training summary expected by the project plan."""

    summary = {
        "experiment": experiment,
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": seed,
        "git_commit": collect_git_commit(repo_root),
        "hardware": collect_hardware_info(),
        "total_epochs": len(history),
        "total_training_time_s": round(total_training_time_s, 2),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "selection_metric": selection_metric,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "history_path": str((output_dir / "history.json").resolve()),
        "parameters": parameters,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if best_val_ap is not None:
        summary["best_val_ap"] = round(best_val_ap, 6)
    return summary


def infer_best_epoch(history: list[dict[str, Any]], metric: str = "val.ap") -> tuple[int, float]:
    """Infer best epoch from validation history."""

    if metric == "val.ap":
        records = [item for item in history if "ap" in item.get("val", {})]
        if records:
            best_record = max(records, key=lambda item: item["val"]["ap"])
            return int(best_record["epoch"]), float(best_record["val"]["ap"])

    best_record = min(history, key=lambda item: item["val"]["total"])
    return int(best_record["epoch"]), float(best_record["val"]["total"])
