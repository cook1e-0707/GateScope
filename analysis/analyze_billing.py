#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import list_json_files, print_stage_banner, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GateScope billing using the standardized 25-turn conversation workload."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Single conversation progress JSON or a directory containing progress JSON files.",
    )
    parser.add_argument(
        "--pricing-json",
        type=Path,
        required=True,
        help="Pricing JSON describing published per-1K token rates.",
    )
    parser.add_argument(
        "--actual-cost-csv",
        type=Path,
        default=None,
        help="Optional CSV with actual cost observations. Accepted columns: gateway_name, model_name, sequence_id, run_id(optional), actual_cost.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/analysis/billing",
        help="Directory for per-run and aggregate billing outputs.",
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


def normalize_pricing_catalog(payload: dict[str, object]) -> dict[str, object]:
    if "models" in payload or "gateways" in payload or "defaults" in payload:
        return payload
    return {
        "defaults": {},
        "models": payload,
        "gateways": {},
    }


def resolve_pricing_entry(catalog: dict[str, object], gateway_name: str | None, model_name: str | None) -> dict[str, object] | None:
    if not model_name:
        return None
    gateways = catalog.get("gateways", {})
    if isinstance(gateways, dict) and gateway_name in gateways:
        gateway_entry = gateways[gateway_name]
        if isinstance(gateway_entry, dict):
            models = gateway_entry.get("models", {})
            if isinstance(models, dict) and model_name in models and isinstance(models[model_name], dict):
                merged = dict(catalog.get("defaults", {}))
                merged.update(gateway_entry)
                merged.update(models[model_name])
                return merged
            if {"input_per_1k", "output_per_1k"}.issubset(gateway_entry.keys()):
                merged = dict(catalog.get("defaults", {}))
                merged.update(gateway_entry)
                return merged

    models = catalog.get("models", {})
    if isinstance(models, dict) and model_name in models and isinstance(models[model_name], dict):
        merged = dict(catalog.get("defaults", {}))
        merged.update(models[model_name])
        return merged
    return None


def expected_cost_from_tokens(
    *,
    input_tokens: float,
    cached_tokens: float,
    output_tokens: float,
    pricing: dict[str, object],
) -> tuple[float, bool]:
    input_per_1k = float(pricing.get("input_per_1k", 0.0))
    output_per_1k = float(pricing.get("output_per_1k", 0.0))
    apply_cache_discount = bool(pricing.get("apply_cache_discount", "cached_input_per_1k" in pricing))
    if apply_cache_discount:
        cached_input_per_1k = float(pricing.get("cached_input_per_1k", input_per_1k))
        effective_cached_tokens = cached_tokens
    else:
        cached_input_per_1k = input_per_1k
        effective_cached_tokens = 0.0

    expected_cost = (
        max(input_tokens - effective_cached_tokens, 0.0) / 1000.0 * input_per_1k
        + effective_cached_tokens / 1000.0 * cached_input_per_1k
        + output_tokens / 1000.0 * output_per_1k
    )
    return float(expected_cost), apply_cache_discount


def load_actual_cost_rows(path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if path is None:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(path.resolve())
    if "actual_cost" not in df.columns:
        raise ValueError("actual cost CSV must contain an actual_cost column.")
    run_level = df[df.columns.intersection(["gateway_name", "model_name", "sequence_id", "run_id", "actual_cost"])].copy()
    aggregate_level = df[df.columns.intersection(["gateway_name", "model_name", "sequence_id", "actual_cost"])].copy()
    return run_level, aggregate_level


def collect_run_rows(input_path: Path, catalog: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for json_path in list_json_files(input_path):
        payload = read_json(json_path)
        turn_records = payload.get("turn_records")
        if not isinstance(turn_records, list):
            continue
        model_name = None if payload.get("model_name") is None else str(payload.get("model_name"))
        gateway_name = infer_gateway_name(json_path, payload, model_name)
        sequence_id = payload.get("sequence_id")

        input_tokens = 0.0
        cached_tokens = 0.0
        output_tokens = 0.0
        for turn_record in turn_records:
            if not isinstance(turn_record, dict):
                continue
            input_tokens += float(turn_record.get("tokens_prompt") or 0.0)
            cached_tokens += float(turn_record.get("tokens_cached") or 0.0)
            output_tokens += float(turn_record.get("tokens_completion") or 0.0)

        pricing = resolve_pricing_entry(catalog, gateway_name, model_name)
        if pricing is not None:
            expected_cost, apply_cache_discount = expected_cost_from_tokens(
                input_tokens=input_tokens,
                cached_tokens=cached_tokens,
                output_tokens=output_tokens,
                pricing=pricing,
            )
        else:
            expected_cost = None
            apply_cache_discount = False

        rows.append(
            {
                "run_id": json_path.stem,
                "file_name": json_path.name,
                "gateway_name": gateway_name,
                "model_name": model_name,
                "sequence_id": sequence_id,
                "input_tokens": int(input_tokens),
                "cached_tokens": int(cached_tokens),
                "output_tokens": int(output_tokens),
                "expected_cost": expected_cost,
                "apply_cache_discount": apply_cache_discount,
            }
        )
    return rows


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Analyze billing",
        "paper-aligned implementation",
        "Aggregate 25-turn conversation runs, compute C_expected from published rates, and compare against actual cost when available.",
    )

    input_path = args.input_path.resolve()
    pricing_catalog = normalize_pricing_catalog(read_json(args.pricing_json.resolve()))
    run_rows = collect_run_rows(input_path, pricing_catalog)
    if not run_rows:
        print(f"No supported conversation logs found under {input_path}")
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df = pd.DataFrame(run_rows)
    actual_run_df, actual_group_df = load_actual_cost_rows(args.actual_cost_csv)
    if not actual_run_df.empty:
        run_df = run_df.merge(
            actual_run_df,
            on=[column for column in ["gateway_name", "model_name", "sequence_id", "run_id"] if column in actual_run_df.columns],
            how="left",
        )
        run_df = run_df.rename(columns={"actual_cost": "actual_cost"})
    else:
        run_df["actual_cost"] = pd.NA

    run_df["billing_gap"] = pd.to_numeric(run_df["actual_cost"], errors="coerce") - pd.to_numeric(run_df["expected_cost"], errors="coerce")
    run_df["billing_gap_percent"] = (
        run_df["billing_gap"] / pd.to_numeric(run_df["expected_cost"], errors="coerce").replace(0, pd.NA) * 100.0
    )
    run_df = run_df.sort_values(["gateway_name", "model_name", "sequence_id", "run_id"], na_position="last")
    runs_path = output_dir / "billing_runs.csv"
    run_df.to_csv(runs_path, index=False)

    aggregate_rows: list[dict[str, object]] = []
    for group_key, group_df in run_df.groupby(["gateway_name", "model_name", "sequence_id"], dropna=False, sort=True):
        expected_total = pd.to_numeric(group_df["expected_cost"], errors="coerce").sum(min_count=1)
        aggregate_row = {
            "gateway_name": group_key[0],
            "model_name": group_key[1],
            "sequence_id": group_key[2],
            "runs": int(len(group_df)),
            "input_tokens": int(group_df["input_tokens"].sum()),
            "cached_tokens": int(group_df["cached_tokens"].sum()),
            "output_tokens": int(group_df["output_tokens"].sum()),
            "C_expected": float(expected_total) if pd.notna(expected_total) else None,
            "cache_discount_applied": bool(group_df["apply_cache_discount"].fillna(False).any()),
        }
        aggregate_rows.append(aggregate_row)

    aggregate_df = pd.DataFrame(aggregate_rows)
    if not actual_group_df.empty:
        aggregate_df = aggregate_df.merge(
            actual_group_df.rename(columns={"actual_cost": "C_actual"}),
            on=[column for column in ["gateway_name", "model_name", "sequence_id"] if column in actual_group_df.columns],
            how="left",
        )
    elif run_df["actual_cost"].notna().any():
        actual_aggregate = (
            run_df.groupby(["gateway_name", "model_name", "sequence_id"], dropna=False)["actual_cost"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"actual_cost": "C_actual"})
        )
        aggregate_df = aggregate_df.merge(actual_aggregate, on=["gateway_name", "model_name", "sequence_id"], how="left")
    else:
        aggregate_df["C_actual"] = pd.NA

    aggregate_df["Gap"] = pd.to_numeric(aggregate_df["C_actual"], errors="coerce") - pd.to_numeric(aggregate_df["C_expected"], errors="coerce")
    aggregate_df["Gap_percent"] = (
        aggregate_df["Gap"] / pd.to_numeric(aggregate_df["C_expected"], errors="coerce").replace(0, pd.NA) * 100.0
    )
    aggregate_df = aggregate_df.sort_values(["gateway_name", "model_name", "sequence_id"], na_position="last")
    aggregate_path = output_dir / "billing_aggregate.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    overview = {
        "protocol": "paper_billing_25_turn_conversation",
        "input_path": str(input_path),
        "run_level_csv": str(runs_path),
        "aggregate_csv": str(aggregate_path),
        "runs": int(len(run_df)),
        "groups": int(len(aggregate_df)),
        "pricing_json": str(args.pricing_json.resolve()),
        "actual_cost_csv": str(args.actual_cost_csv.resolve()) if args.actual_cost_csv else None,
    }
    write_json(output_dir / "billing_overview.json", overview)

    print(f"Run CSV       : {runs_path}")
    print(f"Aggregate CSV : {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
