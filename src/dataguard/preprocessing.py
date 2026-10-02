"""Train-fitted mixed-type preprocessing and defensive feature selection."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataguard.config import MAX_ENCODED_CATEGORIES, MAX_MODEL_FEATURES
from dataguard.ingestion import DatasetError
from dataguard.profiling import column_kind, identifier_like


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """A stateless copy: infinities become NaN; categorical cells become strings."""
    result = frame.copy(deep=True)
    for name in result:
        if is_numeric_dtype(result[name]) and not is_bool_dtype(result[name]):
            result[name] = result[name].astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            result[name] = result[name].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return result


def select_features(train: pd.DataFrame, y_train: pd.Series, excluded: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    """Audit training predictors only; no holdout statistics determine feature removal."""
    kept, dropped = [], {}
    for name in train:
        values = train[name]
        kind = column_kind(values)
        reason = None
        if name in (excluded or []):
            reason = "User excluded"
        elif values.nunique() <= 1:
            reason = "Constant / all-missing in training partition"
        elif identifier_like(str(name), values, kind):
            reason = "Possible identifier (training-only heuristic)"
        elif kind == "datetime":
            reason = "Datetime requires domain-specific feature engineering"
        elif kind == "categorical" and values.nunique() > 50 and values.nunique() / max(1, values.count()) > 0.5:
            reason = "High-cardinality category / possible free text"
        elif values.astype(str).reset_index(drop=True).equals(y_train.astype(str).reset_index(drop=True)):
            reason = "Exact target copy on training rows; inspect provenance"
        if reason:
            dropped[str(name)] = reason
        else:
            kept.append(name)
    if len(kept) > MAX_MODEL_FEATURES:
        for name in kept[MAX_MODEL_FEATURES:]:
            dropped[str(name)] = f"Baseline feature limit ({MAX_MODEL_FEATURES})"
        kept = kept[:MAX_MODEL_FEATURES]
    if not kept:
        raise DatasetError("No usable predictors remain after the training feature audit.")
    return kept, dropped


def make_preprocessor(train: pd.DataFrame, *, scale: bool = True) -> ColumnTransformer:
    numeric = [name for name in train if is_numeric_dtype(train[name]) and not is_bool_dtype(train[name])]
    categorical = [name for name in train if name not in numeric]
    numerical_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numerical_steps.append(("scale", StandardScaler()))
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2,
                                 max_categories=MAX_ENCODED_CATEGORIES, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numerical_steps), numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop", sparse_threshold=0)
