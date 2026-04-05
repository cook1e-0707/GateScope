#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import print_stage_banner, read_json, write_json
from models.binary_protocol import compute_binary_metrics, load_model_registry, slugify_model_name
from tabular_utils import build_target_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paper-aligned GateScope binary classifiers on the official test split."
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/processed/binary_protocol/binary_test.csv",
        help="Official test CSV created by split_data.py.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/models/binary_protocol",
        help="Directory containing trained binary models and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/evaluation/official_binary",
        help="Directory for official test metrics.",
    )
    parser.add_argument("--model-column", default="model_name", help="Model identity column.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Evaluate official binary models",
        "paper-aligned implementation",
        "Score each one-vs-rest classifier on the held-out official test rows and report per-model metrics.",
    )

    test_csv = args.test_csv.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_csv.exists():
        print(f"Test CSV not found: {test_csv}")
        return 1

    test_df = pd.read_csv(test_csv)
    if args.model_column not in test_df.columns:
        print(f"Missing model column: {args.model_column}")
        return 1

    summary = load_model_registry(model_dir)

    per_model_rows: list[dict[str, object]] = []
    for entry in summary["models"]:
        target_model = str(entry["target_model"])
        model_slug = slugify_model_name(target_model)
        metadata = read_json(model_dir / f"{model_slug}.metadata.json")
        prepared_df, feature_matrix, feature_columns = build_target_feature_matrix(
            test_df,
            target_model=target_model,
            model_column=args.model_column,
        )
        dtest = xgb.DMatrix(feature_matrix, feature_names=feature_columns)

        booster = xgb.Booster()
        booster.load_model(str(model_dir / f"{model_slug}.json"))
        best_iteration = int(metadata["best_iteration"])
        y_score = booster.predict(dtest, iteration_range=(0, best_iteration + 1))
        y_true = (prepared_df[args.model_column].astype(str) == target_model).astype(int).to_numpy()

        metrics = compute_binary_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=float(metadata["selected_threshold"]),
        )
        per_model_rows.append(
            {
                "target_model": target_model,
                "difficulty": metadata["difficulty_policy"]["difficulty"],
                "selected_threshold": float(metadata["selected_threshold"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "auroc": float(metrics["auroc"]),
                "average_precision": float(metrics["average_precision"]),
                "predicted_positive": int(metrics["predicted_positive"]),
                "true_positive": int(metrics["true_positive"]),
                "false_positive": int(metrics["false_positive"]),
                "true_negative": int(metrics["true_negative"]),
                "false_negative": int(metrics["false_negative"]),
            }
        )

    per_model_df = pd.DataFrame(per_model_rows).sort_values("target_model").reset_index(drop=True)
    per_model_path = output_dir / "per_model_metrics.csv"
    per_model_df.to_csv(per_model_path, index=False)

    aggregate = {
        "protocol": "paper_binary_one_vs_rest",
        "test_csv": str(test_csv),
        "records": int(len(test_df)),
        "feature_protocol": "paper_full_signature",
        "mean_precision": float(per_model_df["precision"].mean()),
        "mean_recall": float(per_model_df["recall"].mean()),
        "mean_f1": float(per_model_df["f1"].mean()),
        "mean_auroc": float(per_model_df["auroc"].mean()),
        "std_f1": float(per_model_df["f1"].std(ddof=0)),
        "per_model_metrics_csv": str(per_model_path),
        "per_model_metrics": per_model_rows,
    }
    write_json(output_dir / "official_test_summary.json", aggregate)

    print(f"Metrics CSV : {per_model_path}")
    print(f"Mean F1     : {aggregate['mean_f1']:.4f}")
    print(f"Mean P/R    : {aggregate['mean_precision']:.4f} / {aggregate['mean_recall']:.4f}")
    print(f"Mean AUROC  : {aggregate['mean_auroc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
