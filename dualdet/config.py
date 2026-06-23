"""Validated project configuration for controlled E1/E2 model experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from dualdet.models.detector import QAFDetector
from dualdet.models.qaf import FusionMode


@dataclass(frozen=True)
class ModelConfig:
    """Configuration fields that define the current detector architecture."""

    experiment_name: str
    fusion_mode: FusionMode
    class_names: tuple[str, ...]
    input_height: int = 512
    input_width: int = 640
    width_multiple: float = 0.5
    depth_multiple: float = 0.33
    max_channels: int = 512
    neck_repeats: int = 1
    reg_max: int = 16
    use_p2_head: bool = False

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if self.fusion_mode not in ("fixed", "qaf"):
            raise ValueError("fusion_mode must be 'fixed' or 'qaf'")
        if not isinstance(self.use_p2_head, bool):
            raise ValueError("use_p2_head must be a boolean")
        if not self.class_names or any(not name.strip() for name in self.class_names):
            raise ValueError("class_names must contain non-empty names")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique")
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input dimensions must be positive")
        if self.input_height % 32 or self.input_width % 32:
            raise ValueError("input dimensions must be divisible by 32")
        if self.width_multiple <= 0 or self.depth_multiple <= 0:
            raise ValueError("model width and depth multiples must be positive")
        if self.max_channels < 8 or self.neck_repeats <= 0 or self.reg_max <= 1:
            raise ValueError("invalid channel, neck repeat, or reg_max setting")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        """Load a strict JSON config and reject missing or unknown fields."""

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            values: dict[str, Any] = json.load(file)
        expected = {field.name for field in fields(cls)}
        actual = set(values)
        if actual != expected:
            raise ValueError(
                f"config fields mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        values["class_names"] = tuple(values["class_names"])
        return cls(**values)

    def build_model(self) -> QAFDetector:
        """Build the detector defined by this configuration."""

        return QAFDetector(
            num_classes=self.num_classes,
            width_multiple=self.width_multiple,
            depth_multiple=self.depth_multiple,
            max_channels=self.max_channels,
            neck_repeats=self.neck_repeats,
            reg_max=self.reg_max,
            fusion_mode=self.fusion_mode,
            use_p2_head=self.use_p2_head,
        )

    def controlled_signature(self) -> dict[str, Any]:
        """Return fields that must match between the E1 and E2 experiments."""

        values = asdict(self)
        values.pop("experiment_name")
        values.pop("fusion_mode")
        return values

    def p2_controlled_signature(self) -> dict[str, Any]:
        """Return fields that must match between E2 and E3 except P2 usage."""

        values = self.controlled_signature()
        values.pop("use_p2_head")
        return values
