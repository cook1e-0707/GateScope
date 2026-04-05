#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import print_stage_banner, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GateScope latency from normalized single-turn collection logs."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="A JSONL file or a directory containing normalized collection JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/analysis/latency",
        help="Directory for flat and aggregated latency outputs.",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Include failed requests when latency is present. By default only successful calls are analyzed.",
    )
    return parser.parse_args()


def list_jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(child for child in path.glob("*.jsonl") if child.is_file())


def load_latency_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for jsonl_path in list_jsonl_files(path):
        for record in read_jsonl(jsonl_path):
            duration_seconds = record.get("request_duration_seconds", record.get("latency_seconds"))
            rows.append(
                {
                    "source_file": jsonl_path.name,
                    "provider_name": record.get("provider_name"),
                    "gateway_name": record.get("gateway_name") or record.get("provider_name"),
                    "model_name": record.get("model_name"),
                    "dataset": record.get("dataset"),
                    "test_id": record.get("test_id"),
                    "success": bool(record.get("success", not record.get("error_message"))),
                    "latency_seconds": duration_seconds,
                }
            )
    return rows


def summarize_latency(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for group_key, group_df in df.groupby(group_columns, dropna=False, sort=True):
        latency_series = pd.to_numeric(group_df["latency_seconds"], errors="coerce").dropna()
        if latency_series.empty:
            continue
        row = {column: value for column, value in zip(group_columns, group_key)}
        row.update(
            {
                "records": int(len(latency_series)),
                "latency_mean_seconds": float(latency_series.mean()),
                "latency_cv": float(latency_series.std(ddof=0) / latency_series.mean()) if latency_series.mean() else None,
                "latency_min_seconds": float(latency_series.min()),
                "latency_max_seconds": float(latency_series.max()),
                "latency_p50_seconds": float(latency_series.quantile(0.50)),
                "latency_p90_seconds": float(latency_series.quantile(0.90)),
                "latency_p99_seconds": float(latency_series.quantile(0.99)),
            }
        )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Analyze latency",
        "paper-aligned implementation",
        "Compute CV and latency ranges/percentiles from normalized single-turn collection logs.",
    )

    input_path = args.input_path.resolve()
    rows = load_latency_rows(input_path)
    if not rows:
        print(f"No JSONL collection records found under {input_path}")
        return 1

    df = pd.DataFrame(rows)
    df["latency_seconds"] = pd.to_numeric(df["latency_seconds"], errors="coerce")
    if not args.include_failures:
        df = df[df["success"] == True].copy()
    df = df[df["latency_seconds"].notna()].copy()
    if df.empty:
        print("No latency rows remain after filtering.")
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    flat_path = output_dir / "latency_records.csv"
    df.sort_values(["gateway_name", "model_name", "dataset", "test_id"], na_position="last").to_csv(flat_path, index=False)

    by_dataset_df = summarize_latency(df, ["gateway_name", "model_name", "dataset"])
    by_dataset_path = output_dir / "latency_by_dataset.csv"
    by_dataset_df.to_csv(by_dataset_path, index=False)

    overall_df = df.copy()
    overall_df["dataset"] = "ALL"
    overall_summary_df = summarize_latency(overall_df, ["gateway_name", "model_name", "dataset"])
    overall_path = output_dir / "latency_overall.csv"
    overall_summary_df.to_csv(overall_path, index=False)

    overview = {
        "protocol": "paper_latency_single_turn",
        "input_path": str(input_path),
        "records": int(len(df)),
        "by_dataset_csv": str(by_dataset_path),
        "overall_csv": str(overall_path),
        "flat_csv": str(flat_path),
        "include_failures": bool(args.include_failures),
    }
    write_json(output_dir / "latency_overview.json", overview)

    print(f"Flat CSV       : {flat_path}")
    print(f"By-dataset CSV : {by_dataset_path}")
    print(f"Overall CSV    : {overall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
