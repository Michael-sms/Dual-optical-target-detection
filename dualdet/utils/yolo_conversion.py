"""Convert the project's COCO RGB splits to an Ultralytics YOLO dataset."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _materialize_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"existing output image differs from source: {destination}")

    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise OSError(
                f"cannot hard-link {source} to {destination}; place --output-root on "
                "the same filesystem as the dataset or use --image-mode copy"
            ) from error
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError("image mode must be hardlink, symlink or copy")


def _category_mapping(payload: dict[str, Any]) -> tuple[dict[int, int], list[str]]:
    categories = sorted(payload.get("categories", []), key=lambda item: int(item["id"]))
    if not categories:
        raise ValueError("COCO annotation file contains no categories")
    category_ids = [int(item["id"]) for item in categories]
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("COCO category ids must be unique")
    return (
        {category_id: index for index, category_id in enumerate(category_ids)},
        [str(item["name"]) for item in categories],
    )


def convert_annotated_split(
    *,
    annotation_path: Path,
    source_image_dir: Path,
    output_image_dir: Path,
    output_label_dir: Path,
    image_mode: str,
) -> tuple[dict[str, int], list[str]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_map, class_names = _category_mapping(payload)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    converted_boxes = 0
    skipped_boxes = 0
    images = payload.get("images", [])
    for image in images:
        image_id = int(image["id"])
        width = float(image["width"])
        height = float(image["height"])
        file_name = str(image["file_name"])
        source = source_image_dir / file_name
        if not source.is_file():
            raise FileNotFoundError(f"missing RGB image: {source}")
        _materialize_image(source, output_image_dir / file_name, image_mode)

        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            if int(annotation.get("iscrowd", 0)):
                skipped_boxes += 1
                continue
            category_id = int(annotation["category_id"])
            if category_id not in category_map:
                raise ValueError(f"unknown category id {category_id} in {annotation_path}")
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            x1 = max(0.0, min(width, x))
            y1 = max(0.0, min(height, y))
            x2 = max(0.0, min(width, x + box_width))
            y2 = max(0.0, min(height, y + box_height))
            clipped_width = x2 - x1
            clipped_height = y2 - y1
            if clipped_width <= 0.0 or clipped_height <= 0.0:
                skipped_boxes += 1
                continue
            center_x = (x1 + x2) / (2.0 * width)
            center_y = (y1 + y2) / (2.0 * height)
            normalized_width = clipped_width / width
            normalized_height = clipped_height / height
            lines.append(
                f"{category_map[category_id]} {center_x:.8f} {center_y:.8f} "
                f"{normalized_width:.8f} {normalized_height:.8f}"
            )
            converted_boxes += 1

        output_label_dir.mkdir(parents=True, exist_ok=True)
        (output_label_dir / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    return (
        {
            "images": len(images),
            "boxes": converted_boxes,
            "skipped_boxes": skipped_boxes,
        },
        class_names,
    )


def materialize_test_split(
    *, source_image_dir: Path, output_image_dir: Path, image_mode: str
) -> dict[str, int]:
    images = sorted(source_image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no test RGB images found under {source_image_dir}")
    for source in images:
        _materialize_image(source, output_image_dir / source.name, image_mode)
    return {"images": len(images)}


def convert_rgb_dataset(
    data_root: str | Path,
    output_root: str | Path,
    *,
    image_mode: str = "hardlink",
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    summaries: dict[str, Any] = {}
    class_names: list[str] | None = None

    for split in ("train", "val"):
        summary, split_class_names = convert_annotated_split(
            annotation_path=data_root / split / f"{split}.json",
            source_image_dir=data_root / split / "rgb",
            output_image_dir=output_root / "images" / split,
            output_label_dir=output_root / "labels" / split,
            image_mode=image_mode,
        )
        if class_names is None:
            class_names = split_class_names
        elif split_class_names != class_names:
            raise ValueError("train and val category definitions differ")
        summaries[split] = summary

    summaries["test"] = materialize_test_split(
        source_image_dir=data_root / "test" / "rgb",
        output_image_dir=output_root / "images" / "test",
        image_mode=image_mode,
    )
    if class_names is None:
        raise RuntimeError("class names were not initialized")

    dataset_yaml = {
        "path": output_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    yaml_path = output_root / "rgb_dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    report = {
        "source": str(data_root),
        "output": str(output_root),
        "image_mode": image_mode,
        "class_names": class_names,
        "splits": summaries,
        "dataset_yaml": str(yaml_path),
    }
    (output_root / "conversion_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
