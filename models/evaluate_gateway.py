#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import print_stage_banner, read_json, write_json
from models.binary_protocol import load_model_registry, slugify_model_name
from tabular_utils import build_target_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GateScope one-vs-rest binary classifiers on gateway response records."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Gateway CSV to score.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/models/binary_protocol",
        help="Directory containing trained binary models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/evaluation/gateway_binary",
        help="Directory for gateway audit outputs.",
    )
    parser.add_argument("--claimed-model-column", default="model_name", help="Claimed model column.")
    parser.add_argument("--gateway-column", default="gateway_name", help="Gateway name column.")
    parser.add_argument("--test-id-column", default="test_id", help="Prompt id column.")
    parser.add_argument(
        "--include-all-scores",
        action="store_true",
        help="Include a JSON dictionary with all model scores per row in the detailed CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Evaluate gateway responses",
        "paper-aligned implementation",
        "Apply the paper binary classifiers and summarize how often gateway responses are classified as the claimed model.",
    )

    input_csv = args.input_csv.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}")
        return 1

    gateway_df = pd.read_csv(input_csv)

    summary = load_model_registry(model_dir)
    model_scores: dict[str, np.ndarray] = {}
    model_thresholds: dict[str, float] = {}
    model_difficulties: dict[str, str] = {}
    prepared_df = gateway_df.copy()

    for entry in summary["models"]:
        target_model = str(entry["target_model"])
        model_slug = slugify_model_name(target_model)
        metadata = read_json(model_dir / f"{model_slug}.metadata.json")
        prepared_target_df, feature_matrix, feature_columns = build_target_feature_matrix(
            gateway_df,
            target_model=target_model,
            model_column=args.claimed_model_column,
            test_id_column=args.test_id_column,
        )
        dmatrix = xgb.DMatrix(feature_matrix, feature_names=feature_columns)
        if len(prepared_target_df) != len(prepared_df):
            raise ValueError("Target feature preparation changed the gateway row count.")
        prepared_df = prepared_target_df

        booster = xgb.Booster()
        booster.load_model(str(model_dir / f"{model_slug}.json"))
        best_iteration = int(metadata["best_iteration"])
        model_scores[target_model] = booster.predict(dmatrix, iteration_range=(0, best_iteration + 1))
        model_thresholds[target_model] = float(metadata["selected_threshold"])
        model_difficulties[target_model] = str(metadata["difficulty_policy"]["difficulty"])

    target_models = sorted(model_scores)
    detailed_rows: list[dict[str, object]] = []
    for row_index, row in prepared_df.iterrows():
        claimed_model = row.get(args.claimed_model_column)
        claimed_model_str = None if pd.isna(claimed_model) else str(claimed_model)

        score_items = [(model_name, float(model_scores[model_name][row_index])) for model_name in target_models]
        score_items.sort(key=lambda item: item[1], reverse=True)
        positive_predictions = [
            model_name
            for model_name, score in score_items
            if score >= model_thresholds[model_name]
        ]
        top_model, top_score = score_items[0]

        if claimed_model_str is None:
            status = "unlabeled"
            claimed_model_supported = False
            claimed_model_score = None
            claimed_model_threshold = None
            classified_as_claimed = None
        elif claimed_model_str not in model_scores:
            status = "unsupported_claim"
            claimed_model_supported = False
            claimed_model_score = None
            claimed_model_threshold = None
            classified_as_claimed = None
        else:
            claimed_model_supported = True
            claimed_model_score = float(model_scores[claimed_model_str][row_index])
            claimed_model_threshold = float(model_thresholds[claimed_model_str])
            classified_as_claimed = bool(claimed_model_score >= claimed_model_threshold)
            status = "classified_as_claimed" if classified_as_claimed else "not_classified_as_claimed"

        detail = {
            "claimed_model": claimed_model_str,
            "gateway_name": row.get(args.gateway_column) if args.gateway_column in prepared_df.columns else None,
            "test_id": row.get(args.test_id_column) if args.test_id_column in prepared_df.columns else None,
            "top_model": top_model,
            "top_score": float(top_score),
            "positive_predictions": json.dumps(positive_predictions, ensure_ascii=False),
            "positive_prediction_count": int(len(positive_predictions)),
            "claimed_model_supported": claimed_model_supported,
            "claimed_model_score": claimed_model_score,
            "claimed_model_threshold": claimed_model_threshold,
            "classified_as_claimed": classified_as_claimed,
            "status": status,
        }
        if args.include_all_scores:
            detail["all_scores"] = json.dumps({name: score for name, score in score_items}, ensure_ascii=False)
        detailed_rows.append(detail)

    detailed_df = pd.DataFrame(detailed_rows)
    predictions_path = output_dir / "gateway_binary_predictions.csv"
    detailed_df.to_csv(predictions_path, index=False)

    summary_payload: dict[str, object] = {
        "protocol": "paper_binary_one_vs_rest",
        "feature_protocol": "paper_full_signature",
        "input_csv": str(input_csv),
        "records": int(len(detailed_df)),
        "predictions_csv": str(predictions_path),
        "supported_records": int(detailed_df["claimed_model_supported"].fillna(False).sum()),
    }

    supported_df = detailed_df[detailed_df["claimed_model_supported"] == True].copy()
    if not supported_df.empty:
        supported_df["classified_as_claimed_numeric"] = supported_df["classified_as_claimed"].astype(int)
        summary_payload["classified_as_claimed_rate"] = float(supported_df["classified_as_claimed_numeric"].mean())

        per_model = (
            supported_df.groupby("claimed_model", dropna=False)["classified_as_claimed_numeric"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "records", "mean": "classified_as_claimed_rate"})
            .sort_values("claimed_model")
        )
        per_model["difficulty"] = per_model["claimed_model"].map(model_difficulties)
        per_model_path = output_dir / "claimed_model_summary.csv"
        per_model.to_csv(per_model_path, index=False)
        summary_payload["claimed_model_summary_csv"] = str(per_model_path)

        if args.gateway_column in supported_df.columns:
            per_gateway = (
                supported_df.groupby(args.gateway_column, dropna=False)["classified_as_claimed_numeric"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "records", "mean": "classified_as_claimed_rate"})
                .sort_values(args.gateway_column)
            )
            per_gateway_path = output_dir / "gateway_summary.csv"
            per_gateway.to_csv(per_gateway_path, index=False)
            summary_payload["gateway_summary_csv"] = str(per_gateway_path)

            matrix = (
                supported_df.pivot_table(
                    index=args.claimed_model_column if args.claimed_model_column in supported_df.columns else "claimed_model",
                    columns=args.gateway_column,
                    values="classified_as_claimed_numeric",
                    aggfunc="mean",
                )
                .sort_index()
                .sort_index(axis=1)
            )
            matrix_path = output_dir / "gateway_claim_matrix.csv"
            matrix.to_csv(matrix_path)
            summary_payload["gateway_claim_matrix_csv"] = str(matrix_path)

    unsupported_rows = detailed_df[detailed_df["status"] == "unsupported_claim"]
    if not unsupported_rows.empty:
        unsupported_models = sorted({value for value in unsupported_rows["claimed_model"].dropna().astype(str)})
        summary_payload["unsupported_claims"] = unsupported_models

    write_json(output_dir / "gateway_audit_summary.json", summary_payload)

    print(f"Predictions : {predictions_path}")
    if "classified_as_claimed_rate" in summary_payload:
        print(f"Claim rate  : {summary_payload['classified_as_claimed_rate']:.4f}")
    print(f"Records     : {summary_payload['records']}")
    print(f"Supported   : {summary_payload['supported_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
