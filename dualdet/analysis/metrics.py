"""Lightweight validation analysis for COCO-style detection results.

This module intentionally avoids pycocotools so it can run in the minimal
project environment. It does not replace official COCO AP evaluation; it
produces member-3 report material such as per-class precision/recall, small
object recall and a localization-based confusion matrix.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CATEGORY_NAMES = {
    1: "car",
    2: "truck",
    3: "bus",
    4: "van",
    5: "freight_car",
}

AREA_BUCKETS = {
    "small": (0.0, 32.0 * 32.0),
    "medium": (32.0 * 32.0, 96.0 * 96.0),
    "large": (96.0 * 96.0, math.inf),
}


@dataclass(frozen=True)
class Annotation:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float


@dataclass(frozen=True)
class Prediction:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class ClassMetrics:
    category_id: int
    name: str
    gt_count: int
    pred_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float


@dataclass(frozen=True)
class BucketMetrics:
    bucket: str
    gt_count: int
    true_positive: int
    recall: float


@dataclass(frozen=True)
class AnalysisReport:
    iou_threshold: float
    score_threshold: float
    class_metrics: list[ClassMetrics]
    area_recall: list[BucketMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    totals: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "iou_threshold": self.iou_threshold,
            "score_threshold": self.score_threshold,
            "class_metrics": [asdict(item) for item in self.class_metrics],
            "area_recall": [asdict(item) for item in self.area_recall],
            "confusion_matrix": self.confusion_matrix,
            "totals": self.totals,
        }


def _as_bbox(value: Any, *, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"{field_name} must be a 4-number list")
    bbox = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError(f"{field_name} contains non-finite values")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"{field_name} width and height must be positive")
    return bbox


def _bbox_iou_xywh(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def _load_categories(coco: dict[str, Any]) -> dict[int, str]:
    categories = {}
    for item in coco.get("categories", []):
        category_id = int(item["id"])
        categories[category_id] = str(item.get("name", DEFAULT_CATEGORY_NAMES.get(category_id, category_id)))
    return categories or DEFAULT_CATEGORY_NAMES.copy()


def _load_annotations(coco: dict[str, Any]) -> list[Annotation]:
    annotations = []
    for item in coco.get("annotations", []):
        bbox = _as_bbox(item["bbox"], field_name="annotation bbox")
        area = float(item.get("area", bbox[2] * bbox[3]))
        annotations.append(
            Annotation(
                image_id=int(item["image_id"]),
                category_id=int(item["category_id"]),
                bbox=bbox,
                area=area,
            )
        )
    return annotations


def _load_predictions(path: str | Path, score_threshold: float) -> list[Prediction]:
    raw_predictions = _load_json(path)
    if not isinstance(raw_predictions, list):
        raise ValueError("prediction JSON must be a list")

    predictions = []
    for item in raw_predictions:
        score = float(item["score"])
        if not math.isfinite(score):
            raise ValueError("prediction score contains non-finite values")
        if score < score_threshold:
            continue
        predictions.append(
            Prediction(
                image_id=int(item["image_id"]),
                category_id=int(item["category_id"]),
                bbox=_as_bbox(item["bbox"], field_name="prediction bbox"),
                score=score,
            )
        )
    return sorted(predictions, key=lambda item: item.score, reverse=True)


def _group_by_category(items: list[Annotation] | list[Prediction]) -> dict[int, list[Annotation] | list[Prediction]]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for item in items:
        grouped[item.category_id].append(item)
    return dict(grouped)


def _match_same_class(
    annotations: list[Annotation],
    predictions: list[Prediction],
    iou_threshold: float,
) -> tuple[set[int], set[int]]:
    matched_annotations: set[int] = set()
    matched_predictions: set[int] = set()
    annotations_by_image: dict[int, list[tuple[int, Annotation]]] = defaultdict(list)
    for index, annotation in enumerate(annotations):
        annotations_by_image[annotation.image_id].append((index, annotation))

    for pred_index, prediction in enumerate(predictions):
        best_index = None
        best_iou = 0.0
        for ann_index, annotation in annotations_by_image.get(prediction.image_id, []):
            if ann_index in matched_annotations:
                continue
            iou = _bbox_iou_xywh(prediction.bbox, annotation.bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = ann_index
        if best_index is not None and best_iou >= iou_threshold:
            matched_annotations.add(best_index)
            matched_predictions.add(pred_index)

    return matched_annotations, matched_predictions


def _bucket_for_area(area: float) -> str:
    for name, (lower, upper) in AREA_BUCKETS.items():
        if lower <= area < upper:
            return name
    raise ValueError(f"invalid annotation area: {area}")


def _build_confusion_matrix(
    annotations: list[Annotation],
    predictions: list[Prediction],
    category_names: dict[int, str],
    iou_threshold: float,
) -> dict[str, dict[str, int]]:
    predictions_by_image: dict[int, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[prediction.image_id].append(prediction)

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for annotation in annotations:
        true_name = category_names.get(annotation.category_id, str(annotation.category_id))
        best_prediction = None
        best_iou = 0.0
        for prediction in predictions_by_image.get(annotation.image_id, []):
            iou = _bbox_iou_xywh(prediction.bbox, annotation.bbox)
            if iou > best_iou:
                best_iou = iou
                best_prediction = prediction
        if best_prediction is None or best_iou < iou_threshold:
            predicted_name = "missed"
        else:
            predicted_name = category_names.get(best_prediction.category_id, str(best_prediction.category_id))
        matrix[true_name][predicted_name] += 1
    return {row: dict(columns) for row, columns in matrix.items()}


def analyze_predictions(
    annotation_json: str | Path,
    prediction_json: str | Path,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
) -> AnalysisReport:
    """Analyze one COCO annotation file and one result-style prediction JSON."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be in [0, 1]")

    coco = _load_json(annotation_json)
    if not isinstance(coco, dict):
        raise ValueError("annotation JSON must be a COCO object")
    category_names = _load_categories(coco)
    annotations = _load_annotations(coco)
    predictions = _load_predictions(prediction_json, score_threshold)

    annotations_by_category = _group_by_category(annotations)
    predictions_by_category = _group_by_category(predictions)
    class_metrics = []
    matched_annotations_all: set[int] = set()

    for category_id in sorted(category_names):
        class_annotations = list(annotations_by_category.get(category_id, []))
        class_predictions = list(predictions_by_category.get(category_id, []))
        matched_annotations, matched_predictions = _match_same_class(
            class_annotations,
            class_predictions,
            iou_threshold,
        )
        true_positive = len(matched_annotations)
        pred_count = len(class_predictions)
        gt_count = len(class_annotations)
        class_metrics.append(
            ClassMetrics(
                category_id=category_id,
                name=category_names[category_id],
                gt_count=gt_count,
                pred_count=pred_count,
                true_positive=true_positive,
                false_positive=pred_count - len(matched_predictions),
                false_negative=gt_count - true_positive,
                precision=true_positive / pred_count if pred_count else 0.0,
                recall=true_positive / gt_count if gt_count else 0.0,
            )
        )

        original_indexes = [
            index
            for index, annotation in enumerate(annotations)
            if annotation.category_id == category_id
        ]
        matched_annotations_all.update(original_indexes[index] for index in matched_annotations)

    bucket_counts = {name: 0 for name in AREA_BUCKETS}
    bucket_hits = {name: 0 for name in AREA_BUCKETS}
    for index, annotation in enumerate(annotations):
        bucket = _bucket_for_area(annotation.area)
        bucket_counts[bucket] += 1
        if index in matched_annotations_all:
            bucket_hits[bucket] += 1

    area_recall = [
        BucketMetrics(
            bucket=name,
            gt_count=bucket_counts[name],
            true_positive=bucket_hits[name],
            recall=bucket_hits[name] / bucket_counts[name] if bucket_counts[name] else 0.0,
        )
        for name in AREA_BUCKETS
    ]
    true_positive_total = sum(item.true_positive for item in class_metrics)
    pred_total = len(predictions)
    gt_total = len(annotations)
    return AnalysisReport(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        class_metrics=class_metrics,
        area_recall=area_recall,
        confusion_matrix=_build_confusion_matrix(
            annotations,
            predictions,
            category_names,
            iou_threshold,
        ),
        totals={
            "gt_count": gt_total,
            "pred_count": pred_total,
            "true_positive": true_positive_total,
            "false_positive": pred_total - true_positive_total,
            "false_negative": gt_total - true_positive_total,
            "precision": true_positive_total / pred_total if pred_total else 0.0,
            "recall": true_positive_total / gt_total if gt_total else 0.0,
        },
    )


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_markdown_report(report: AnalysisReport) -> str:
    """Format analysis results as markdown tables for the report draft."""

    lines = [
        f"# Prediction Analysis @ IoU {report.iou_threshold:.2f}",
        "",
        f"Score threshold: `{report.score_threshold:.3f}`",
        "",
        "## Per-Class Summary",
        "",
        "| Class | GT | Pred | TP | FP | FN | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.class_metrics:
        lines.append(
            "| "
            f"{item.name} | {item.gt_count} | {item.pred_count} | "
            f"{item.true_positive} | {item.false_positive} | {item.false_negative} | "
            f"{_percent(item.precision)} | {_percent(item.recall)} |"
        )

    lines.extend(
        [
            "",
            "## Area Recall",
            "",
            "| Area Bucket | GT | TP | Recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in report.area_recall:
        lines.append(
            f"| {item.bucket} | {item.gt_count} | {item.true_positive} | {_percent(item.recall)} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "Rows are ground-truth classes; columns are the best-overlap prediction class or `missed`.",
            "",
        ]
    )
    columns = sorted(
        {column for row in report.confusion_matrix.values() for column in row}
    )
    lines.append("| GT \\ Pred | " + " | ".join(columns) + " |")
    lines.append("|---" + "|---:" * len(columns) + "|")
    for row_name in sorted(report.confusion_matrix):
        row = report.confusion_matrix[row_name]
        lines.append(
            f"| {row_name} | " + " | ".join(str(row.get(column, 0)) for column in columns) + " |"
        )

    totals = report.totals
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- GT boxes: {totals['gt_count']}",
            f"- Predictions: {totals['pred_count']}",
            f"- True positives: {totals['true_positive']}",
            f"- Overall precision: {_percent(float(totals['precision']))}",
            f"- Overall recall: {_percent(float(totals['recall']))}",
        ]
    )
    return "\n".join(lines) + "\n"
