#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import print_stage_banner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the paper split: 10 training and 2 testing samples per (model_name, test_id)."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/processed/official_api_data_enhanced.csv",
        help="Enhanced official CSV with 12 repetitions per (model_name, test_id).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/processed/binary_protocol",
        help="Directory for paper-protocol CSVs and metadata.",
    )
    parser.add_argument("--model-column", default="model_name", help="Model identity column.")
    parser.add_argument("--test-id-column", default="test_id", help="Prompt id column.")
    parser.add_argument(
        "--sort-columns",
        default="timestamp,session_id,record_id",
        help="Comma-separated columns used to order repetitions within each group.",
    )
    parser.add_argument("--train-count", type=int, default=10, help="Training repetitions per group.")
    parser.add_argument("--test-count", type=int, default=2, help="Testing repetitions per group.")
    parser.add_argument(
        "--drop-incomplete-groups",
        action="store_true",
        help="Skip groups that do not have exactly train_count + test_count repetitions.",
    )
    return parser.parse_args()


def build_group_order(df: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    ordered = df.copy()
    ordered["_source_row_index"] = range(len(ordered))

    usable_columns = [column for column in sort_columns if column in ordered.columns]
    if usable_columns:
        ordered = ordered.sort_values(usable_columns + ["_source_row_index"], kind="stable")
    return ordered


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Create paper split",
        "paper-aligned implementation",
        "Split each (model_name, test_id) repetition group into 10 training and 2 testing rows.",
    )

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}")
        return 1

    df = pd.read_csv(input_csv)
    required = {args.model_column, args.test_id_column}
    missing = required.difference(df.columns)
    if missing:
        print(f"Missing required columns: {sorted(missing)}")
        return 1

    repetitions_per_group = args.train_count + args.test_count
    sort_columns = [column.strip() for column in args.sort_columns.split(",") if column.strip()]
    ordered = build_group_order(df, sort_columns)

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    invalid_groups: list[dict[str, object]] = []
    split_groups: list[dict[str, object]] = []

    grouped = ordered.groupby([args.model_column, args.test_id_column], sort=True, dropna=False)
    for (model_name, test_id), group_df in grouped:
        group_df = group_df.reset_index(drop=True)
        actual_size = len(group_df)
        if actual_size != repetitions_per_group:
            invalid_groups.append(
                {
                    "model_name": None if pd.isna(model_name) else str(model_name),
                    "test_id": None if pd.isna(test_id) else str(test_id),
                    "expected_repetitions": repetitions_per_group,
                    "actual_repetitions": int(actual_size),
                }
            )
            continue

        annotated = group_df.copy()
        annotated["paper_group_rank"] = range(1, actual_size + 1)

        train_df = annotated.iloc[: args.train_count].copy()
        train_df["paper_split"] = "train"
        test_df = annotated.iloc[args.train_count : repetitions_per_group].copy()
        test_df["paper_split"] = "test"

        train_parts.append(train_df)
        test_parts.append(test_df)
        split_groups.append(
            {
                "model_name": None if pd.isna(model_name) else str(model_name),
                "test_id": None if pd.isna(test_id) else str(test_id),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
            }
        )

    if invalid_groups and not args.drop_incomplete_groups:
        error_path = output_dir / "invalid_groups.json"
        error_path.write_text(json.dumps(invalid_groups, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Found {len(invalid_groups)} invalid groups. Details written to {error_path}")
        return 1

    if invalid_groups:
        skipped_path = output_dir / "invalid_groups.json"
        skipped_path.write_text(json.dumps(invalid_groups, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Skipped {len(invalid_groups)} incomplete groups. Details written to {skipped_path}")

    train_output = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test_output = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    train_path = output_dir / "binary_train.csv"
    test_path = output_dir / "binary_test.csv"
    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)

    metadata = {
        "protocol": "paper_binary_10_2_split",
        "input_csv": str(input_csv),
        "model_column": args.model_column,
        "test_id_column": args.test_id_column,
        "sort_columns_requested": sort_columns,
        "sort_columns_used": [column for column in sort_columns if column in ordered.columns],
        "train_count_per_group": args.train_count,
        "test_count_per_group": args.test_count,
        "expected_repetitions_per_group": repetitions_per_group,
        "drop_incomplete_groups": bool(args.drop_incomplete_groups),
        "skipped_incomplete_groups": invalid_groups,
        "groups": split_groups,
        "total_groups": int(len(split_groups)),
        "train_records": int(len(train_output)),
        "test_records": int(len(test_output)),
        "unique_models": sorted(train_output[args.model_column].dropna().astype(str).unique().tolist()),
        "unique_test_ids": sorted(train_output[args.test_id_column].dropna().astype(str).unique().tolist()),
    }

    metadata_path = output_dir / "binary_split_info.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Train CSV : {train_path}")
    print(f"Test CSV  : {test_path}")
    print(f"Groups    : {metadata['total_groups']}")
    print(f"Train rows: {metadata['train_records']:,}")
    print(f"Test rows : {metadata['test_records']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
