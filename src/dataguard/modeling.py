"""Reproducible holdout baselines with group protection and training-only preprocessing."""

import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from dataguard.config import MAX_MODEL_ROWS, MIN_MODEL_ROWS, SEED
from dataguard.evaluation import classification_metrics, regression_metrics
from dataguard.ingestion import DatasetError
from dataguard.preprocessing import make_preprocessor, normalize_features, select_features
from dataguard.readiness import infer_problem, valid_target

logger = logging.getLogger(__name__)


@dataclass
class ModelRun:
    task: str
    target: str
    results: pd.DataFrame
    winner: str
    ranking_metric: str
    train_rows: int
    test_rows: int
    included_features: list[str]
    excluded_features: dict[str, str]
    confusion: pd.DataFrame | None
    importance: pd.DataFrame
    notes: list[str]
    errors: dict[str, str] = field(default_factory=dict)


def split_data(x: pd.DataFrame, y: pd.Series, task: str):
    """Group identical predictor records to prevent exact-feature overlap across partitions."""
    hashes = pd.util.hash_pandas_object(x, index=False)
    if hashes.duplicated().any():
        if hashes.nunique() < 2:
            raise DatasetError(
                "At least two independent predictor groups are required for a holdout."
            )
        splitter = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=SEED)
        for train_idx, test_idx in splitter.split(x, y, groups=hashes):
            if task != "classification" or (
                set(y.iloc[train_idx]) == set(y) == set(y.iloc[test_idx])
            ):
                return (
                    x.iloc[train_idx],
                    x.iloc[test_idx],
                    y.iloc[train_idx],
                    y.iloc[test_idx],
                    "Grouped identical predictors to prevent train/test overlap.",
                )
        raise DatasetError(
            "Could not create a grouped holdout containing every class. Add independent observations."
        )
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=SEED,
            stratify=y if task == "classification" else None,
        )
    except ValueError as exc:
        raise DatasetError(f"Unable to create a valid 75/25 holdout: {exc}") from exc
    return x_train, x_test, y_train, y_test, "Seeded 75/25 holdout; stratified for classification."


def train_baselines(
    frame: pd.DataFrame,
    target: str,
    *,
    override: str = "auto",
    excluded: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ModelRun:
    """Compare three learned models plus a dummy control; preserve uploaded data."""
    problem = infer_problem(frame, target, override)
    if problem.task not in {"classification", "regression"}:
        raise DatasetError(problem.reason)
    work = frame.loc[valid_target(frame[target])].copy().reset_index(drop=True)
    notes = [f"Removed {len(frame) - len(work)} rows with missing/infinite targets."]
    if len(work) > MAX_MODEL_ROWS:
        # Classification sample is stratified whenever feasible; fail clearly on rare classes.
        try:
            work, _ = train_test_split(
                work,
                train_size=MAX_MODEL_ROWS,
                random_state=SEED,
                stratify=work[target] if problem.task == "classification" else None,
            )
        except ValueError as exc:
            raise DatasetError(
                f"Cannot make a representative bounded baseline sample: {exc}"
            ) from exc
        notes.append(f"Baselines use a seeded sample of {MAX_MODEL_ROWS:,} rows.")
    y = work[target].astype(str) if problem.task == "classification" else work[target].astype(float)
    raw_x = work.drop(columns=target)
    # Stateless normalization before splitting learns no summary statistics or vocabulary.
    x = normalize_features(raw_x)
    x_train, x_test, y_train, y_test, split_note = split_data(x, y, problem.task)
    notes.append(split_note)
    if len(x_train) < MIN_MODEL_ROWS // 2 or len(x_test) < 2:
        raise DatasetError("Too few independent rows remain for meaningful holdout evaluation.")
    # Audit raw values so booleans and datetimes retain semantic type information.
    kept, dropped = select_features(raw_x.loc[x_train.index], y_train, excluded)
    x_train, x_test = x_train[kept], x_test[kept]
    preprocessor = make_preprocessor(x_train)
    if problem.task == "classification":
        models = {
            "Dummy prior": DummyClassifier(strategy="prior"),
            "Logistic regression": LogisticRegression(
                max_iter=1500, class_weight="balanced", random_state=SEED
            ),
            "Random forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=SEED,
            ),
            "Extra trees": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=SEED,
            ),
        }
        metric, ascending = "f1_macro", False
        scoring = "f1_macro"
    else:
        models = {
            "Dummy mean": DummyRegressor(strategy="mean"),
            "Ridge regression": Ridge(alpha=1.0),
            "Random forest": RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=SEED
            ),
            "Extra trees": ExtraTreesRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=SEED
            ),
        }
        metric, ascending, scoring = "rmse", True, "neg_root_mean_squared_error"
    rows, fitted, errors = [], {}, {}
    for name, model in models.items():
        if progress:
            progress(name)
        pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(x_train, y_train)
            for warning in caught:
                notes.append(f"{name}: {warning.message}")
            predictions = pipeline.predict(x_test)
            metrics = (
                classification_metrics(
                    y_test, predictions, pipeline.predict_proba(x_test), pipeline.classes_
                )
                if problem.task == "classification"
                else regression_metrics(y_test, predictions)
            )
            rows.append(
                {"model": name, **metrics, "fit_seconds": round(time.perf_counter() - started, 3)}
            )
            fitted[name] = pipeline
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("Baseline %s failed (%s)", name, type(exc).__name__)
            errors[name] = f"{type(exc).__name__}: {exc}"
    if not rows:
        raise DatasetError("All baselines failed. Inspect numerical ranges and feature types.")
    results = (
        pd.DataFrame(rows)
        .sort_values(metric, ascending=ascending, kind="stable")
        .reset_index(drop=True)
    )
    winner = str(results.iloc[0].model)
    best = fitted[winner]
    notes.append(
        f"Top baseline on this holdout: {winner}, ranked by {'lowest RMSE' if ascending else 'highest macro F1'}. This is a single exploratory holdout, not production validation."
    )
    confusion = None
    if problem.task == "classification":
        confusion = pd.DataFrame(
            confusion_matrix(y_test, best.predict(x_test), labels=best.classes_),
            index=best.classes_,
            columns=best.classes_,
        )
    importance = pd.DataFrame(columns=["feature", "mean", "std"])
    if not winner.startswith("Dummy") and len(x_test) >= 20:
        if progress:
            progress("Holdout permutation importance")
        sample = x_test.sample(min(250, len(x_test)), random_state=SEED)
        sample_y = y_test.loc[sample.index]
        try:
            permutation = permutation_importance(
                best, sample, sample_y, n_repeats=3, random_state=SEED, scoring=scoring, n_jobs=1
            )
            importance = pd.DataFrame(
                {
                    "feature": kept,
                    "mean": permutation.importances_mean,
                    "std": permutation.importances_std,
                }
            ).sort_values("mean", ascending=False)
            notes.append(
                "Permutation importance: 3 repeats, at most 250 seeded holdout rows; original-column score decrease. Negative values and correlated-feature effects are possible. Exploratory reuse of holdout; not causal importance."
            )
        except (ValueError, TypeError) as exc:
            notes.append(f"Permutation importance unavailable: {exc}")
    else:
        notes.append("Permutation importance skipped: dummy winner or fewer than 20 holdout rows.")
    return ModelRun(
        problem.task,
        target,
        results,
        winner,
        metric,
        len(x_train),
        len(x_test),
        kept,
        dropped,
        confusion,
        importance,
        list(dict.fromkeys(notes)),
        errors,
    )
