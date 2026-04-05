#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import list_json_files, print_stage_banner, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-turn GateScope conversation logs into paper-style 5-run tables."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Single progress JSON or a directory containing progress JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/analysis/conversation",
        help="Directory for run-level and aggregated outputs.",
    )
    return parser.parse_args()


def infer_gateway_name(path: Path, payload: dict[str, object], model_name: str | None) -> str | None:
    explicit_gateway = payload.get("gateway_name")
    if explicit_gateway:
        return str(explicit_gateway)
    stem = path.stem
    if model_name and stem.endswith(f"_{model_name}"):
        prefix = stem[: -(len(model_name) + 1)]
        return prefix or None
    return stem or None


def summarize_file(path: Path) -> dict[str, object] | None:
    payload = read_json(path)
    turn_records = payload.get("turn_records")
    validation = payload.get("validation")
    if not isinstance(turn_records, list) or not isinstance(validation, dict):
        return None

    model_name = payload.get("model_name")
    sequence_id = payload.get("sequence_id")
    details = validation.get("details", {})

    latencies = [
        float(record["latency_ms"])
        for record in turn_records
        if isinstance(record, dict) and record.get("latency_ms") is not None
    ]
    prompt_tokens = [
        float(record["tokens_prompt"])
        for record in turn_records
        if isinstance(record, dict) and record.get("tokens_prompt") is not None
    ]
    cached_tokens = [
        float(record["tokens_cached"])
        for record in turn_records
        if isinstance(record, dict) and record.get("tokens_cached") is not None
    ]
    fingerprint_values = [
        str(record.get("system_fingerprint"))
        for record in turn_records
        if isinstance(record, dict) and record.get("system_fingerprint")
    ]

    total_prompt_tokens = float(sum(prompt_tokens))
    total_cached_tokens = float(sum(cached_tokens))
    cache_rate = total_cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0
    checkpoint_total = int(validation.get("total_checkpoints", len(details)))
    checkpoint_passed = int(validation.get("passed", 0))
    gateway_name = infer_gateway_name(path, payload, None if pd.isna(model_name) else str(model_name))

    return {
        "file_name": path.name,
        "run_id": path.stem,
        "gateway_name": gateway_name,
        "model_name": model_name,
        "sequence_id": sequence_id,
        "status": payload.get("status"),
        "total_turns": int(payload.get("total_turns", len(turn_records))),
        "checkpoint_total": checkpoint_total,
        "checkpoint_passed": checkpoint_passed,
        "checkpoint_failed": int(validation.get("failed", checkpoint_total - checkpoint_passed)),
        "checkpoint_pass_rate": (checkpoint_passed / checkpoint_total) if checkpoint_total else 0.0,
        "turn_10_passed": bool(details.get("10", {}).get("passed", False)),
        "turn_24_passed": bool(details.get("24", {}).get("passed", False)),
        "turn_25_passed": bool(details.get("25", {}).get("passed", False)),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "latency_p50_ms": float(np.median(latencies)) if latencies else None,
        "latency_min_ms": float(np.min(latencies)) if latencies else None,
        "latency_max_ms": float(np.max(latencies)) if latencies else None,
        "total_prompt_tokens": int(total_prompt_tokens),
        "total_cached_tokens": int(total_cached_tokens),
        "cache_rate": float(cache_rate),
        "cache_rate_percent": float(cache_rate * 100.0),
        "unique_fingerprints": int(len(set(fingerprint_values))),
        "_fingerprints": fingerprint_values,
    }


def build_aggregate_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate_rows: list[dict[str, object]] = []
    df = pd.DataFrame(run_rows)
    group_columns = ["gateway_name", "model_name", "sequence_id"]

    for group_key, group_df in df.groupby(group_columns, dropna=False, sort=True):
        matching_rows = [
            row
            for row in run_rows
            if row.get("gateway_name") == group_key[0]
            and row.get("model_name") == group_key[1]
            and row.get("sequence_id") == group_key[2]
        ]
        all_fingerprints = sorted(
            {
                fingerprint
                for row in matching_rows
                for fingerprint in row.get("_fingerprints", [])
                if fingerprint
            }
        )
        total_prompt_tokens = float(group_df["total_prompt_tokens"].sum())
        total_cached_tokens = float(group_df["total_cached_tokens"].sum())
        cache_rate_percent = (total_cached_tokens / total_prompt_tokens * 100.0) if total_prompt_tokens else 0.0

        aggregate_rows.append(
            {
                "gateway_name": group_key[0],
                "model_name": group_key[1],
                "sequence_id": group_key[2],
                "runs": int(len(group_df)),
                "T10": int(group_df["turn_10_passed"].astype(int).sum()),
                "T24": int(group_df["turn_24_passed"].astype(int).sum()),
                "T25": int(group_df["turn_25_passed"].astype(int).sum()),
                "FC": int(len(all_fingerprints)),
                "CR_percent": float(cache_rate_percent),
                "checkpoint_pass_rate_mean": float(group_df["checkpoint_pass_rate"].mean()),
                "latency_mean_ms": float(group_df["latency_mean_ms"].dropna().mean()) if group_df["latency_mean_ms"].notna().any() else None,
                "total_prompt_tokens": int(total_prompt_tokens),
                "total_cached_tokens": int(total_cached_tokens),
            }
        )

    return aggregate_rows


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Analyze multi-turn conversations",
        "paper-aligned implementation",
        "Produce run-level traces and 5-run T10/T24/T25, fingerprint-count, and cache-rate aggregates.",
    )

    files = list_json_files(args.input_path.resolve())
    run_rows = [summary for summary in (summarize_file(path) for path in files) if summary]
    if not run_rows:
        print(f"No supported conversation logs found under {args.input_path.resolve()}")
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_export_rows = []
    for row in run_rows:
        clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
        run_export_rows.append(clean_row)

    runs_df = pd.DataFrame(run_export_rows).sort_values(
        ["gateway_name", "model_name", "sequence_id", "run_id"],
        na_position="last",
    )
    runs_path = output_dir / "conversation_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    aggregate_rows = build_aggregate_rows(run_rows)
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["gateway_name", "model_name", "sequence_id"],
        na_position="last",
    )
    aggregate_path = output_dir / "conversation_aggregate.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    overview = {
        "num_runs": int(len(runs_df)),
        "num_groups": int(len(aggregate_df)),
        "mean_checkpoint_pass_rate": float(runs_df["checkpoint_pass_rate"].mean()),
        "mean_cache_rate_percent": float(runs_df["cache_rate_percent"].mean()),
        "mean_latency_ms": float(runs_df["latency_mean_ms"].dropna().mean()) if runs_df["latency_mean_ms"].notna().any() else None,
        "run_level_csv": str(runs_path),
        "aggregate_csv": str(aggregate_path),
    }
    write_json(output_dir / "conversation_overview.json", overview)

    print(f"Input files   : {len(files)}")
    print(f"Run CSV       : {runs_path}")
    print(f"Aggregate CSV : {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
