from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from artifact_utils import read_json


HARD_MODELS = {
    "o3",
    "gpt-5-mini",
    "o4-mini",
    "gemini-2.5-flash",
}

MEDIUM_MODELS = {
    "o3-mini",
    "gemini-2.0-flash-lite",
    "gpt-5-nano",
}

THRESHOLD_GRID = np.round(np.arange(0.10, 1.00, 0.05), 2)


@dataclass(frozen=True)
class DifficultyPolicy:
    name: str
    exponent: float
    objective: str
    precision_floor: float


def slugify_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip())
    return cleaned.strip("_").lower()


def load_difficulty_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Difficulty override file must be a JSON object: {path}")
    overrides: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Difficulty override must map strings to strings: {path}")
        normalized = value.strip().lower()
        if normalized not in {"easy", "medium", "hard"}:
            raise ValueError(f"Unsupported difficulty '{value}' for model '{key}'")
        overrides[key] = normalized
    return overrides


def difficulty_name_for_model(model_name: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and model_name in overrides:
        return overrides[model_name]
    if model_name in HARD_MODELS:
        return "hard"
    if model_name in MEDIUM_MODELS:
        return "medium"
    return "easy"


def difficulty_policy_for_model(
    model_name: str,
    overrides: dict[str, str] | None = None,
) -> DifficultyPolicy:
    difficulty = difficulty_name_for_model(model_name, overrides=overrides)
    if difficulty == "easy":
        return DifficultyPolicy(
            name="easy",
            exponent=0.5,
            objective="f1",
            precision_floor=0.5,
        )
    if difficulty == "medium":
        return DifficultyPolicy(
            name="medium",
            exponent=0.7,
            objective="f1",
            precision_floor=0.5,
        )
    return DifficultyPolicy(
        name="hard",
        exponent=0.9,
        objective="recall",
        precision_floor=0.35,
    )


def calculate_scale_pos_weight(
    *,
    n_positive: int,
    n_negative: int,
    oversample_factor: int,
    policy: DifficultyPolicy,
) -> tuple[float, float]:
    if n_positive <= 0:
        raise ValueError("Positive count must be greater than zero.")
    target_positive_count = n_positive * oversample_factor
    post_oversampling_ratio = n_negative / max(target_positive_count, 1)
    scale_pos_weight = float(post_oversampling_ratio ** policy.exponent)
    return scale_pos_weight, float(post_oversampling_ratio)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")
    try:
        average_precision = average_precision_score(y_true, y_score)
    except ValueError:
        average_precision = float("nan")

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "average_precision": float(average_precision),
        "predicted_positive": int(y_pred.sum()),
        "true_positive": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_positive": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_negative": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_negative": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def select_operating_threshold(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    policy: DifficultyPolicy,
) -> dict[str, Any]:
    scored_candidates: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []

    for threshold in THRESHOLD_GRID:
        metrics = compute_binary_metrics(y_true, y_score, float(threshold))
        objective_value = metrics["recall"] if policy.objective == "recall" else metrics["f1"]
        candidate = {
            "threshold": float(threshold),
            "objective_value": float(objective_value),
            "meets_precision_floor": bool(metrics["precision"] >= policy.precision_floor),
            "metrics": metrics,
        }
        scored_candidates.append(candidate)
        if candidate["meets_precision_floor"]:
            valid_candidates.append(candidate)

    search_space = valid_candidates if valid_candidates else scored_candidates
    constraint_relaxed = not bool(valid_candidates)

    def ranking_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
        metrics = candidate["metrics"]
        return (
            float(candidate["objective_value"]),
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(candidate["threshold"]),
        )

    best = max(search_space, key=ranking_key)
    return {
        "policy": {
            "difficulty": policy.name,
            "objective": policy.objective,
            "precision_floor": float(policy.precision_floor),
        },
        "constraint_relaxed": constraint_relaxed,
        "selected": best,
        "candidates": scored_candidates,
    }


def load_model_registry(model_dir: Path) -> dict[str, Any]:
    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary not found: {summary_path}")
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"Training summary must be a JSON object: {summary_path}")
    return summary
