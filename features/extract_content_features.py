#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import print_stage_banner
from tabular_utils import (
    FEATURE_COLUMNS,
    PAPER_BASE_FEATURE_COLUMNS,
    build_target_feature_frame,
    ensure_paper_signature_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add engineered content features to a CSV file.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV containing normalized response records.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/processed/official_api_data_enhanced.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help="Optional target model. When set, materialize the full appendix signature including contrastive and ranking features.",
    )
    parser.add_argument("--model-column", default="model_name", help="Model identity column.")
    parser.add_argument("--test-id-column", default="test_id", help="Prompt id column.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Extract content features",
        "reference implementation",
        "Add the engineered columns used by the GateScope classifier pipeline.",
    )

    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}")
        return 1

    df = pd.read_csv(input_csv)
    if args.target_model:
        enhanced = build_target_feature_frame(
            df,
            target_model=args.target_model,
            model_column=args.model_column,
            test_id_column=args.test_id_column,
        )
        added_summary = (
            f"Added full appendix signature: {len(PAPER_BASE_FEATURE_COLUMNS)} base features + "
            f"{len(FEATURE_COLUMNS) - len(PAPER_BASE_FEATURE_COLUMNS)} target-specific contrastive/ranking features"
        )
    else:
        enhanced = ensure_paper_signature_features(df)
        added_summary = (
            "Added appendix base signature: "
            + ", ".join(PAPER_BASE_FEATURE_COLUMNS)
        )
    enhanced.to_csv(output_csv, index=False)

    print(f"Read  : {input_csv}")
    print(f"Wrote : {output_csv}")
    print(f"Rows  : {len(enhanced):,}")
    print(f"Cols  : {len(enhanced.columns):,}")
    print(f"Added : {added_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
