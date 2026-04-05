from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd


PAPER_BASE_FEATURE_COLUMNS = [
    "answer_match",
    "answer_position",
    "depth",
    "mean_step_length",
    "step_length_var",
    "response_length",
    "density",
    "has_numeric",
    "has_latex",
    "parse_success",
    "parse_degree",
]

CONTRASTIVE_SUFFIXES = [
    "mean_diff",
    "relative_diff",
    "cohens_d",
    "std_ratio",
    "overlap_ratio",
    "target_rank",
]

FEATURE_COLUMNS = PAPER_BASE_FEATURE_COLUMNS + [
    f"{feature}_{suffix}"
    for feature in PAPER_BASE_FEATURE_COLUMNS
    for suffix in CONTRASTIVE_SUFFIXES
]

NUMERIC_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\d+)(?:[%/]\d+)?")
LATEX_PATTERN = re.compile(r"(?:\\[A-Za-z]+|\\\(|\\\)|\\\[|\\\]|\$\$?.+?\$\$?)", re.DOTALL)


def _existing_text_series(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    pieces = []
    for column in columns:
        if column in df.columns:
            pieces.append(df[column].fillna("").astype(str))
    if not pieces:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined + "\n" + piece
    return combined


def _to_numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _to_boolish_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return (
        df[column]
        .map(
            {
                True: 1.0,
                False: 0.0,
                "True": 1.0,
                "False": 0.0,
                "true": 1.0,
                "false": 0.0,
                1: 1.0,
                0: 0.0,
            }
        )
        .fillna(default)
        .astype(float)
    )


def calculate_step_variance(raw_value: object) -> float:
    if pd.isna(raw_value) or raw_value == "":
        return 0.0
    try:
        lengths = [float(part.strip()) for part in str(raw_value).split(",") if part.strip()]
    except ValueError:
        return 0.0
    if len(lengths) < 2:
        return 0.0
    return float(np.var(lengths))


def calculate_mean_step_length(raw_value: object) -> float:
    if pd.isna(raw_value) or raw_value == "":
        return 0.0
    try:
        lengths = [float(part.strip()) for part in str(raw_value).split(",") if part.strip()]
    except ValueError:
        return 0.0
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def _parse_degree_series(df: pd.DataFrame) -> pd.Series:
    if "parsing_successful" in df.columns:
        return (
            df["parsing_successful"]
            .map(
                {
                    True: 1.0,
                    False: 0.0,
                    "True": 1.0,
                    "False": 0.0,
                    "true": 1.0,
                    "false": 0.0,
                    1: 1.0,
                    0: 0.0,
                    "partial": 0.5,
                    "PARTIAL": 0.5,
                    "Partial": 0.5,
                }
            )
            .fillna(0.0)
            .astype(float)
        )
    if "parsing_quality" in df.columns:
        return pd.to_numeric(df["parsing_quality"], errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _answer_match_series(df: pd.DataFrame) -> pd.Series:
    exact_match = _to_boolish_series(df, "answer_exact_match")
    normalized_match = _to_boolish_series(df, "answer_normalized_match")
    flexible_match = _to_boolish_series(df, "answer_flexible_match")
    accuracy_score = _to_numeric_series(df, "answer_accuracy_score")
    return (
        (exact_match > 0.0)
        | (normalized_match > 0.0)
        | (flexible_match > 0.0)
        | (accuracy_score >= 0.6)
    ).astype(float)


def _answer_position_series(df: pd.DataFrame, parse_degree: pd.Series) -> pd.Series:
    has_final_answer = pd.Series(False, index=df.index)
    if "model_final_answer" in df.columns:
        has_final_answer = df["model_final_answer"].fillna("").astype(str).str.strip().ne("")
    answer_in_reasoning = _to_boolish_series(df, "answer_in_reasoning")
    return ((has_final_answer) | (parse_degree >= 1.0) | ((parse_degree >= 0.5) & (answer_in_reasoning <= 0.0))).astype(float)


def _contains_pattern(series: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    return series.fillna("").astype(str).str.contains(pattern, regex=True, na=False).astype(float)


def ensure_paper_signature_features(df: pd.DataFrame) -> pd.DataFrame:
    enhanced = df.copy()

    parse_degree = _parse_degree_series(enhanced)
    reasoning_text = _existing_text_series(enhanced, ["reasoning_steps", "response_text"])
    response_text = _existing_text_series(enhanced, ["model_final_answer", "reasoning_steps", "response_text"])

    enhanced["answer_match"] = _answer_match_series(enhanced)
    enhanced["answer_position"] = _answer_position_series(enhanced, parse_degree=parse_degree)
    enhanced["depth"] = _to_numeric_series(enhanced, "reasoning_depth")

    if "avg_step_length" in enhanced.columns:
        enhanced["mean_step_length"] = _to_numeric_series(enhanced, "avg_step_length")
    else:
        step_lengths = enhanced["step_lengths"] if "step_lengths" in enhanced.columns else pd.Series("", index=enhanced.index)
        enhanced["mean_step_length"] = step_lengths.apply(calculate_mean_step_length).astype(float)

    if "step_length_variance" in enhanced.columns:
        enhanced["step_length_var"] = _to_numeric_series(enhanced, "step_length_variance")
    else:
        step_lengths = enhanced["step_lengths"] if "step_lengths" in enhanced.columns else pd.Series("", index=enhanced.index)
        enhanced["step_length_var"] = step_lengths.apply(calculate_step_variance).astype(float)

    enhanced["response_length"] = _to_numeric_series(enhanced, "response_length")
    depth_nonzero = enhanced["depth"].replace(0.0, np.nan)
    enhanced["density"] = (enhanced["response_length"] / depth_nonzero).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if any(column in enhanced.columns for column in ["model_final_answer", "reasoning_steps", "response_text"]):
        enhanced["has_numeric"] = _contains_pattern(response_text, NUMERIC_PATTERN)
        if "has_latex_content" not in enhanced.columns:
            enhanced["has_latex"] = _contains_pattern(response_text, LATEX_PATTERN)
        else:
            enhanced["has_latex"] = np.maximum(
                _to_boolish_series(enhanced, "has_latex_content"),
                _contains_pattern(response_text, LATEX_PATTERN),
            )
    else:
        enhanced["has_numeric"] = np.maximum(
            _to_boolish_series(enhanced, "has_latex_numeric"),
            _contains_pattern(reasoning_text, NUMERIC_PATTERN),
        )
        enhanced["has_latex"] = _to_boolish_series(enhanced, "has_latex_content")

    enhanced["parse_degree"] = parse_degree.astype(float)
    enhanced["parse_success"] = (parse_degree >= 1.0).astype(float)

    for column in PAPER_BASE_FEATURE_COLUMNS:
        enhanced[column] = pd.to_numeric(enhanced[column], errors="coerce").fillna(0.0).astype(float)

    return enhanced


def _compute_contrastive_statistics(target_values: pd.Series, other_values: pd.Series) -> dict[str, float]:
    target_values = pd.to_numeric(target_values, errors="coerce").dropna().astype(float)
    other_values = pd.to_numeric(other_values, errors="coerce").dropna().astype(float)
    if target_values.empty or other_values.empty:
        return {
            "mean_diff": 0.0,
            "relative_diff": 0.0,
            "cohens_d": 0.0,
            "std_ratio": 0.0,
            "overlap_ratio": 0.0,
        }

    target_mean = float(target_values.mean())
    other_mean = float(other_values.mean())
    target_std = float(target_values.std(ddof=0))
    other_std = float(other_values.std(ddof=0))
    mean_diff = target_mean - other_mean
    relative_diff = mean_diff / (abs(other_mean) + 1e-6)
    pooled_std = float(np.sqrt((target_std**2 + other_std**2) / 2.0 + 1e-6))
    cohens_d = mean_diff / pooled_std
    std_ratio = target_std / (other_std + 1e-6)

    target_min = float(target_values.min())
    target_max = float(target_values.max())
    other_min = float(other_values.min())
    other_max = float(other_values.max())
    overlap_min = max(target_min, other_min)
    overlap_max = min(target_max, other_max)
    if overlap_min < overlap_max:
        overlap_length = overlap_max - overlap_min
        total_length = max(target_max, other_max) - min(target_min, other_min)
        overlap_ratio = overlap_length / (total_length + 1e-6)
    else:
        overlap_ratio = 0.0

    return {
        "mean_diff": float(mean_diff),
        "relative_diff": float(relative_diff),
        "cohens_d": float(cohens_d),
        "std_ratio": float(std_ratio),
        "overlap_ratio": float(overlap_ratio),
    }


def _compute_normalized_rank(group: pd.DataFrame, feature: str, target_model: str, model_column: str) -> float:
    grouped = group.groupby(model_column, dropna=False)[feature].mean()
    if target_model not in grouped.index:
        return 0.0
    n_models = len(grouped)
    if n_models == 1:
        return 0.5
    rank = grouped.rank(ascending=False, method="average").get(target_model)
    if pd.isna(rank):
        return 0.0
    return float((n_models - float(rank) + 1.0) / n_models)


def build_target_feature_frame(
    df: pd.DataFrame,
    *,
    target_model: str,
    model_column: str = "model_name",
    test_id_column: str = "test_id",
) -> pd.DataFrame:
    if model_column not in df.columns:
        raise KeyError(f"Missing model column: {model_column}")
    if test_id_column not in df.columns:
        raise KeyError(f"Missing test id column: {test_id_column}")

    prepared = ensure_paper_signature_features(df)
    prepared[model_column] = prepared[model_column].fillna("").astype(str)
    prepared[test_id_column] = prepared[test_id_column].fillna("").astype(str)

    feature_rows: list[dict[str, float | str]] = []
    for test_id, group in prepared.groupby(test_id_column, sort=False, dropna=False):
        target_group = group[group[model_column] == target_model]
        other_group = group[group[model_column] != target_model]
        row: dict[str, float | str] = {test_id_column: str(test_id)}
        for feature in PAPER_BASE_FEATURE_COLUMNS:
            stats = _compute_contrastive_statistics(target_group[feature], other_group[feature])
            for suffix in [
                "mean_diff",
                "relative_diff",
                "cohens_d",
                "std_ratio",
                "overlap_ratio",
            ]:
                row[f"{feature}_{suffix}"] = stats[suffix]
            row[f"{feature}_target_rank"] = _compute_normalized_rank(
                group,
                feature=feature,
                target_model=target_model,
                model_column=model_column,
            )
        feature_rows.append(row)

    contrastive_df = pd.DataFrame(feature_rows)
    merged = prepared.merge(contrastive_df, on=test_id_column, how="left")
    for column in FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).astype(float)
    return merged


def build_target_feature_matrix(
    df: pd.DataFrame,
    *,
    target_model: str,
    model_column: str = "model_name",
    test_id_column: str = "test_id",
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    prepared = build_target_feature_frame(
        df,
        target_model=target_model,
        model_column=model_column,
        test_id_column=test_id_column,
    )
    return prepared, prepared[FEATURE_COLUMNS].to_numpy(dtype=float), list(FEATURE_COLUMNS)


def ensure_engineered_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    enhanced = ensure_paper_signature_features(df)
    for feature in feature_columns:
        if feature not in enhanced.columns:
            enhanced[feature] = 0.0
        enhanced[feature] = pd.to_numeric(enhanced[feature], errors="coerce").fillna(0.0).astype(float)
    return enhanced


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, np.ndarray]:
    prepared = ensure_engineered_features(df, feature_columns=feature_columns)
    return prepared, prepared[list(feature_columns)].to_numpy(dtype=float)
