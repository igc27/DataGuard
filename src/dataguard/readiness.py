"""Problem inference, conservative leakage clues, and measurable readiness factors."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from dataguard.config import MAX_CLASSES, MIN_MODEL_ROWS
from dataguard.profiling import Profile, column_kind
from dataguard.quality import Score


@dataclass
class Problem:
    task: str
    reason: str
    valid_rows: int
    classes: int
    minority_rate: float | None = None


@dataclass
class Readiness:
    problem: Problem
    score: Score
    status: str
    recommendations: list[str]
    leakage: list[dict[str, str]]


def valid_target(series: pd.Series) -> pd.Series:
    mask = series.notna()
    if is_numeric_dtype(series):
        mask &= np.isfinite(series.astype(float))
    return mask


def infer_problem(frame: pd.DataFrame, target: str, override: str = "auto") -> Problem:
    if target not in frame:
        return Problem("unsupported", "Choose an existing target column.", 0, 0)
    values = frame.loc[valid_target(frame[target]), target]
    n, unique = len(values), values.nunique()
    if len(frame.columns) < 2:
        return Problem("unsupported", "At least one predictor column is required.", n, unique)
    if n < MIN_MODEL_ROWS:
        return Problem("unsupported", f"At least {MIN_MODEL_ROWS} usable target rows are required.", n, unique)
    if unique < 2:
        return Problem("unsupported", "Target has fewer than two distinct usable values.", n, unique)
    numeric = is_numeric_dtype(values) and not is_bool_dtype(values)
    if override == "regression":
        task = "regression" if numeric else "unsupported"
        reason = "Regression explicitly selected." if numeric else "Regression requires a numeric target."
    elif override == "classification":
        task = "classification" if unique <= MAX_CLASSES else "unsupported"
        reason = "Classification explicitly selected." if task == "classification" else f"Classification is limited to {MAX_CLASSES} classes."
    elif column_kind(values) == "datetime":
        task, reason = "unsupported", "Datetime targets need an explicit forecasting design."
    elif not numeric:
        task = "classification" if unique <= MAX_CLASSES else "unsupported"
        reason = "Categorical target." if task == "classification" else "High-cardinality target is unsupported."
    elif unique <= MAX_CLASSES and (values.astype(float) % 1 == 0).all():
        task, reason = "classification", "Low-cardinality integer target; confirm this is a class label."
    elif unique <= MAX_CLASSES:
        task, reason = "ambiguous", "Low-cardinality fractional target. Select classification or regression explicitly."
    else:
        task, reason = "regression", "Numeric target with more than 20 distinct values; confirm regression intent."
    minority = None
    if task == "classification":
        counts = values.value_counts()
        minority = float(counts.min() / counts.sum())
        if counts.min() < 2:
            task, reason = "unsupported", "Each class needs at least two observations for a holdout split."
    return Problem(task, reason, n, unique, minority)


def leakage_clues(frame: pd.DataFrame, target: str) -> list[dict[str, str]]:
    """Full-data diagnostic clues; never a claim of causal leakage or model selection input."""
    clues = []
    if target not in frame:
        return clues
    mask = valid_target(frame[target])
    y = frame.loc[mask, target]
    for name in frame.columns:
        if name == target:
            continue
        x = frame.loc[mask, name]
        if x.astype(str).equals(y.astype(str)):
            clues.append({"column": str(name), "evidence": "Equals the selected target on all usable target rows",
                          "action": "Investigate provenance; exact target copies are excluded by the training audit."})
        elif is_numeric_dtype(x) and is_numeric_dtype(y):
            pairs = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pairs) >= 20 and pairs.x.nunique() > 1:
                correlation = pairs.x.corr(pairs.y)
                if pd.notna(correlation) and abs(correlation) >= 0.995:
                    clues.append({"column": str(name), "evidence": f"Near-perfect target correlation (r={correlation:.4f}); heuristic",
                                  "action": "Inspect timing and provenance; this may be legitimate signal. Review feature exclusions."})
        tokens = str(name).lower()
        if any(token in tokens for token in ("post_outcome", "target_copy", "label_copy")) and not any(c['column'] == name for c in clues):
            clues.append({"column": str(name), "evidence": "Outcome-like column name; heuristic",
                          "action": "Check whether this feature is available at prediction time."})
    return clues


def assess_readiness(frame: pd.DataFrame, profile: Profile, target: str, override: str = "auto") -> Readiness:
    problem = infer_problem(frame, target, override)
    features = profile.column_stats.drop(index=target, errors="ignore")
    clues = leakage_clues(frame, target)
    n_features = max(1, len(features))
    missing = features.missing.sum() / max(1, profile.rows * len(features))
    invalid = features.invalid.sum() / max(1, profile.rows * len(features))
    unusable = (features.constant | features.id_like | (features.kind == "datetime")).sum() / n_features
    target_loss = 1 - problem.valid_rows / profile.rows
    imbalance = 0.0
    if problem.minority_rate is not None and problem.classes:
        imbalance = max(0, 1 - problem.minority_rate * problem.classes)
    factors = [
        ("Feature missingness", 20, missing), ("Infinite feature values", 10, invalid),
        ("Unusable / identifier / datetime features", 15, unusable),
        ("High-cardinality features", 10, features.high_cardinality.sum() / n_features),
        ("Unusable target rows", 15, target_loss), ("Class imbalance", 15, imbalance),
        ("Target leakage clues (heuristic)", 15, min(1, len(clues) / n_features)),
    ]
    breakdown = pd.DataFrame([{"factor": name, "max_penalty": weight, "rate": float(rate),
                              "penalty": float(weight * rate)} for name, weight, rate in factors])
    supported = problem.task in {"classification", "regression"}
    score = Score(round(max(0, 100 - breakdown.penalty.sum()), 1) if supported else 0.0, breakdown)
    recommendations = ["Use a train/test split appropriate to your deployment setting; grouped and temporal data need domain review."]
    if missing or invalid:
        recommendations.append("Impute numerical/categorical gaps and replace infinities within the training pipeline.")
    if (features.kind.isin(["categorical", "boolean"])).any():
        recommendations.append("Encode categories with training-only vocabulary and handle unseen categories.")
    if (features.kind == "numeric").any():
        recommendations.append("Scale numerical predictors for linear models using training-only statistics.")
    if unusable:
        recommendations.append("Review identifiers and constant features. Datetime features require domain feature engineering.")
    if features.high_cardinality.any():
        recommendations.append("Review high-cardinality text/categories; automatic baselines exclude these features.")
    if imbalance > 0.5:
        recommendations.append("Investigate class imbalance; compare macro F1 and balanced accuracy alongside accuracy.")
    if clues:
        recommendations.append("Investigate possible leakage. Correlation alone cannot establish whether a feature is available at prediction time.")
    if profile.duplicate_rows:
        recommendations.append("Prevent identical predictor rows appearing in both training and holdout sets.")
    if not supported:
        recommendations.append(problem.reason)
    status = "Unsupported / needs a target decision" if not supported else (
        "Good, preprocessing recommended" if score.value >= 75 else "Review issues before relying on baselines")
    return Readiness(problem, score, status, recommendations, clues)
