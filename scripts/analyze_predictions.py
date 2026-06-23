"""CLI for member-3 prediction analysis tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dualdet.analysis import analyze_predictions, format_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze COCO-style validation predictions for report tables."
    )
    parser.add_argument("--annotations", required=True, help="Path to val.json.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to COCO result-style prediction JSON.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for lightweight matching.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
        help="Predictions below this score are ignored.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Optional path for a markdown report.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path for a machine-readable analysis JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_predictions(
        args.annotations,
        args.predictions,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
    markdown = format_markdown_report(report)

    if args.markdown_output:
        output_path = Path(args.markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
