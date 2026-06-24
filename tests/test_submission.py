"""Tests for competition submission validation."""

import json

from dualdet.utils.submission import filename_to_image_id, format_submission_text, validate_submission


def test_filename_to_image_id() -> None:
    assert filename_to_image_id("00001.jpg") == 1
    assert filename_to_image_id("01000.jpg") == 1000


def test_validate_submission_accepts_valid_record() -> None:
    records = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "score": 0.95,
        }
    ]
    assert validate_submission(records, valid_image_ids={1}) == []


def test_validate_submission_rejects_invalid_category() -> None:
    records = [
        {
            "image_id": 1,
            "category_id": 6,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "score": 0.5,
        }
    ]
    errors = validate_submission(records, valid_image_ids={1})
    assert any("category_id" in message for message in errors)


def test_format_submission_text_uses_six_lines_per_record() -> None:
    records = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [154.0, 158.0, 25.0, 63.0],
            "score": 0.95,
        },
        {
            "image_id": 2,
            "category_id": 3,
            "bbox": [10.4, 20.6, 30.0, 40.0],
            "score": 0.5,
        },
    ]
    text = format_submission_text(records)
    assert text.splitlines() == [
        "[{",
        '"image_id": 1,',
        '"category_id": 1,',
        '"bbox": [154.0,158.0,25.0,63.0],',
        '"score": 0.95',
        "},",
        "{",
        '"image_id": 2,',
        '"category_id": 3,',
        '"bbox": [10.4,20.6,30.0,40.0],',
        '"score": 0.5',
        "}]",
    ]
    assert json.loads(text) == records
