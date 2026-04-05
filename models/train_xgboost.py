#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import time
import sys

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from artifact_utils import print_stage_banner, write_json
from models.binary_protocol import (
    calculate_scale_pos_weight,
    difficulty_policy_for_model,
    load_difficulty_overrides,
    select_operating_threshold,
    slugify_model_name,
)
from tabular_utils import FEATURE_COLUMNS, build_target_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one-vs-rest GateScope binary XGBoost classifiers using the paper protocol."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/processed/binary_protocol/binary_train.csv",
        help="Official training CSV created by split_data.py.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ARTIFACT_ROOT / "outputs/models/binary_protocol",
        help="Directory for trained binary models and metadata.",
    )
    parser.add_argument("--model-column", default="model_name", help="Target model column.")
    parser.add_argument("--validation-size", type=float, default=0.1, help="Held-out validation share.")
    parser.add_argument("--oversample-factor", type=int, default=20, help="Positive-class oversampling factor.")
    parser.add_argument("--num-rounds", type=int, default=500, help="Maximum boosting rounds.")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping patience on validation logloss.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--difficulty-json",
        type=Path,
        default=None,
        help="Optional JSON mapping model_name to easy|medium|hard.",
    )
    return parser.parse_args()


def build_binary_targets(df: pd.DataFrame, model_column: str, target_model: str) -> np.ndarray:
    return (df[model_column].astype(str) == target_model).astype(int).to_numpy()


def train_single_model(
    *,
    df: pd.DataFrame,
    target_model: str,
    model_column: str,
    model_dir: Path,
    validation_size: float,
    oversample_factor: int,
    num_rounds: int,
    early_stopping_rounds: int,
    seed: int,
    difficulty_overrides: dict[str, str],
) -> dict[str, object]:
    labels_full = build_binary_targets(df, model_column=model_column, target_model=target_model)
    row_indices = np.arange(len(df))

    train_indices, val_indices = train_test_split(
        row_indices,
        test_size=validation_size,
        random_state=seed,
        stratify=labels_full,
    )

    fit_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    prepared_fit_df, x_train, feature_columns = build_target_feature_matrix(
        fit_df,
        target_model=target_model,
        model_column=model_column,
    )
    prepared_val_df, x_val, _ = build_target_feature_matrix(
        val_df,
        target_model=target_model,
        model_column=model_column,
    )

    y_train = build_binary_targets(prepared_fit_df, model_column=model_column, target_model=target_model)
    y_val = build_binary_targets(prepared_val_df, model_column=model_column, target_model=target_model)

    n_positive_train = int(y_train.sum())
    n_negative_train = int(len(y_train) - n_positive_train)
    if n_positive_train == 0 or n_negative_train == 0:
        raise ValueError(f"Target model {target_model} does not have both positive and negative training examples.")

    oversampler = RandomOverSampler(
        sampling_strategy={1: n_positive_train * oversample_factor},
        random_state=seed,
    )
    x_train_resampled, y_train_resampled = oversampler.fit_resample(x_train, y_train)

    policy = difficulty_policy_for_model(target_model, overrides=difficulty_overrides)
    scale_pos_weight, post_ratio = calculate_scale_pos_weight(
        n_positive=n_positive_train,
        n_negative=n_negative_train,
        oversample_factor=oversample_factor,
        policy=policy,
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "tree_method": "hist",
    }

    evals_result: dict[str, dict[str, list[float]]] = {}
    dtrain = xgb.DMatrix(x_train_resampled, label=y_train_resampled, feature_names=feature_columns)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=feature_columns)

    start_time = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        evals_result=evals_result,
    )
    training_time_seconds = time.time() - start_time

    best_iteration = int(booster.best_iteration)
    y_val_score = booster.predict(dval, iteration_range=(0, best_iteration + 1))
    threshold_search = select_operating_threshold(y_true=y_val, y_score=y_val_score, policy=policy)
    selected = threshold_search["selected"]

    model_slug = slugify_model_name(target_model)
    model_path = model_dir / f"{model_slug}.json"
    booster.save_model(str(model_path))

    metadata = {
        "target_model": target_model,
        "model_slug": model_slug,
        "model_path": str(model_path),
        "feature_columns": feature_columns,
        "feature_protocol": "paper_full_signature",
        "difficulty_policy": threshold_search["policy"],
        "validation_size": float(validation_size),
        "oversample_factor": int(oversample_factor),
        "counts": {
            "train_rows_total": int(len(df)),
            "train_rows_positive": int(labels_full.sum()),
            "train_rows_negative": int(len(labels_full) - labels_full.sum()),
            "fit_rows": int(len(prepared_fit_df)),
            "fit_rows_positive": n_positive_train,
            "fit_rows_negative": int(len(prepared_fit_df) - n_positive_train),
            "validation_rows": int(len(y_val)),
            "validation_rows_positive": int(y_val.sum()),
            "validation_rows_negative": int(len(y_val) - y_val.sum()),
            "resampled_rows": int(len(y_train_resampled)),
            "resampled_positive": int(y_train_resampled.sum()),
            "resampled_negative": int(len(y_train_resampled) - y_train_resampled.sum()),
        },
        "post_oversampling_negative_to_positive_ratio": float(post_ratio),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": best_iteration,
        "best_val_logloss": float(booster.best_score),
        "training_time_seconds": float(training_time_seconds),
        "threshold_search": threshold_search,
        "selected_threshold": float(selected["threshold"]),
        "validation_metrics_at_selected_threshold": selected["metrics"],
        "params": params,
        "train_logloss": [float(value) for value in evals_result["train"]["logloss"]],
        "val_logloss": [float(value) for value in evals_result["val"]["logloss"]],
    }
    write_json(model_dir / f"{model_slug}.metadata.json", metadata)
    return metadata


def main() -> int:
    args = parse_args()

    print_stage_banner(
        "Train binary XGBoost models",
        "paper-aligned implementation",
        "Train one one-vs-rest classifier per official model with oversampling, held-out validation, and threshold search.",
    )

    train_csv = args.train_csv.resolve()
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists():
        print(f"Training CSV not found: {train_csv}")
        return 1

    train_df = pd.read_csv(train_csv)
    if args.model_column not in train_df.columns:
        print(f"Missing model column: {args.model_column}")
        return 1

    difficulty_overrides = load_difficulty_overrides(args.difficulty_json.resolve() if args.difficulty_json else None)
    target_models = sorted(train_df[args.model_column].dropna().astype(str).unique().tolist())

    training_logs: list[dict[str, object]] = []
    for index, target_model in enumerate(target_models):
        print(f"[{index + 1}/{len(target_models)}] Training {target_model}")
        training_logs.append(
            train_single_model(
                df=train_df,
                target_model=target_model,
                model_column=args.model_column,
                model_dir=model_dir,
                validation_size=args.validation_size,
                oversample_factor=args.oversample_factor,
                num_rounds=args.num_rounds,
                early_stopping_rounds=args.early_stopping_rounds,
                seed=args.seed + index,
                difficulty_overrides=difficulty_overrides,
            )
        )

    summary = {
        "protocol": "paper_binary_one_vs_rest",
        "train_csv": str(train_csv),
        "feature_protocol": "paper_full_signature",
        "feature_columns": FEATURE_COLUMNS,
        "target_models": [entry["target_model"] for entry in training_logs],
        "validation_size": float(args.validation_size),
        "oversample_factor": int(args.oversample_factor),
        "num_rounds": int(args.num_rounds),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "mean_validation_f1": float(np.mean([entry["validation_metrics_at_selected_threshold"]["f1"] for entry in training_logs])),
        "mean_validation_precision": float(np.mean([entry["validation_metrics_at_selected_threshold"]["precision"] for entry in training_logs])),
        "mean_validation_recall": float(np.mean([entry["validation_metrics_at_selected_threshold"]["recall"] for entry in training_logs])),
        "models": training_logs,
    }
    write_json(model_dir / "training_summary.json", summary)

    print()
    print(f"Models trained: {len(target_models)}")
    print(f"Model dir     : {model_dir}")
    print(f"Mean val F1   : {summary['mean_validation_f1']:.4f}")
    print(f"Mean val P/R  : {summary['mean_validation_precision']:.4f} / {summary['mean_validation_recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
