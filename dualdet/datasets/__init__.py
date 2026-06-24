"""Dataset loading and augmentation for paired RGB-T detection."""

from dualdet.datasets.coco import COCOAnnotationIndex, load_coco_annotations
from dualdet.datasets.rgb_t import RGBTDataset, RGBTTestDataset
from dualdet.datasets.transforms import SyncAugmentConfig, SyncAugmentPipeline

__all__ = [
    "COCOAnnotationIndex",
    "RGBTDataset",
    "RGBTTestDataset",
    "SyncAugmentConfig",
    "SyncAugmentPipeline",
    "load_coco_annotations",
]
