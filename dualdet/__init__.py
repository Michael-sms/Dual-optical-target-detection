"""Dual-optical target detection package."""

__all__ = [
    "AnchorFreeDetectHead",
    "DualStreamBackbone",
    "MultiScaleQAF",
    "ModelConfig",
    "P2PANFPN",
    "PANFPN",
    "QAFDetector",
    "QAFFusionBlock",
    "QAFOutput",
]


def __getattr__(name: str) -> object:
    """Lazy-load torch-backed objects only when they are requested."""

    if name == "ModelConfig":
        from dualdet.config import ModelConfig

        return ModelConfig
    if name in {
        "AnchorFreeDetectHead",
        "DualStreamBackbone",
        "MultiScaleQAF",
        "P2PANFPN",
        "PANFPN",
        "QAFDetector",
        "QAFFusionBlock",
        "QAFOutput",
    }:
        from dualdet import models

        return getattr(models, name)
    raise AttributeError(f"module 'dualdet' has no attribute {name!r}")
