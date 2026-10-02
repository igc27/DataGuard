"""Full-frame quality statistics, with explicitly bounded pairwise calculations."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from dataguard.config import MAX_CORRELATION_COLUMNS, MAX_CORRELATION_ROWS, SEED
from dataguard.ingestion import DatasetError


@dataclass
class Profile:
    rows: int
    columns: int
    memory_bytes: int
    duplicate_rows: int
    column_stats: pd.DataFrame
    correlation: pd.DataFrame
    correlated_pairs: list[dict]
    duplicate_columns: list[tuple[str, str]]
    notes: list[str]


def column_kind(series: pd.Series) -> str:
    """Infer semantic types without converting the source data."""
    if is_bool_dtype(series):
        return "boolean"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "numeric"
    values = series.dropna().astype(str)
    if len(values) and set(values.str.lower().unique()) <= {"true", "false"}:
        return "boolean"
    sample = values.iloc[:200]
    # Date punctuation is required: ordinary numeric strings must not become dates.
    if len(sample) and sample.str.contains(r"[-/:]", regex=True).mean() >= 0.9:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            return "datetime"
    return "categorical"


def identifier_like(name: str, series: pd.Series, kind: str) -> bool:
    """Flag named IDs or mostly unique monotonic integer sequences, never unique floats alone."""
    values = series.dropna()
    if len(values) < 10 or values.nunique() / len(values) < 0.98:
        return False
    normalized = name.lower().replace("-", "_")
    named = normalized in {"id", "uuid", "index"} or normalized.endswith(("_id", "_uuid"))
    if named:
        return True
    if kind == "numeric":
        finite = values[np.isfinite(values)]
        return bool(len(finite) == len(values) and (finite % 1 == 0).all()
                    and (finite.is_monotonic_increasing or finite.is_monotonic_decreasing))
    return False


def profile_dataset(frame: pd.DataFrame) -> Profile:
    """Profile missingness, types, finite statistics, IQR outliers, and Pearson correlation."""
    if frame.empty or not frame.columns.is_unique:
        raise DatasetError("Profiling needs data rows and unique column names.")
    frame = frame.rename(columns=str)
    if not frame.columns.is_unique:
        raise DatasetError("Column labels must be unique when represented as text.")
    records = []
    for name in frame.columns:
        series = frame[name]
        valid = series.dropna()
        kind = column_kind(series)
        count = int(series.count())
        unique = int(series.nunique())
        common = float(valid.value_counts(normalize=True).iloc[0]) if count else 0.0
        stats = {
            "column": str(name), "dtype": str(series.dtype), "kind": kind,
            "missing": int(series.isna().sum()), "missing_pct": float(series.isna().mean() * 100),
            "unique": unique, "cardinality": unique / count if count else 0.0,
            "constant": unique <= 1, "near_constant": unique > 1 and common >= 0.98,
            "id_like": identifier_like(str(name), series, kind),
            "high_cardinality": kind == "categorical" and unique > 50 and unique / max(count, 1) > 0.5,
            "invalid": 0, "outliers": 0, "outlier_pct": 0.0, "zero_pct": 0.0,
            "mixed_numeric": False, "mean": None, "std": None, "min": None,
            "q25": None, "median": None, "q75": None, "max": None, "skew": None,
            "iqr_lower": None, "iqr_upper": None,
        }
        if kind == "numeric":
            nums = pd.to_numeric(series, errors="coerce").astype(float)
            stats["invalid"] = int(np.isinf(nums).sum())
            finite = nums[np.isfinite(nums)]
            if len(finite):
                q1, q3 = finite.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                # A zero IQR has no stable fence; report no IQR detections, transparently.
                outliers = int(((finite < lower) | (finite > upper)).sum()) if iqr > 0 else 0
                stats.update({
                    "mean": float(finite.mean()), "std": float(finite.std()) if len(finite) > 1 else None,
                    "min": float(finite.min()), "q25": float(q1), "median": float(finite.median()),
                    "q75": float(q3), "max": float(finite.max()),
                    "skew": float(finite.skew()) if len(finite) >= 3 and finite.nunique() > 1 else None,
                    "zero_pct": float((finite == 0).mean() * 100), "outliers": outliers,
                    "outlier_pct": outliers / len(finite) * 100,
                    "iqr_lower": float(lower) if iqr > 0 else None,
                    "iqr_upper": float(upper) if iqr > 0 else None,
                })
        elif kind == "categorical" and count:
            ratio = pd.to_numeric(valid, errors="coerce").notna().mean()
            stats["mixed_numeric"] = bool(0.8 <= ratio < 1)
        records.append(stats)
    table = pd.DataFrame(records).set_index("column")
    numeric = table.index[table.kind == "numeric"].tolist()
    notes = []
    if len(numeric) > MAX_CORRELATION_COLUMNS:
        notes.append(f"Correlation uses the first {MAX_CORRELATION_COLUMNS} numeric columns.")
    numeric = numeric[:MAX_CORRELATION_COLUMNS]
    corr_frame = frame[numeric].replace([np.inf, -np.inf], np.nan)
    if len(corr_frame) > MAX_CORRELATION_ROWS:
        corr_frame = corr_frame.sample(MAX_CORRELATION_ROWS, random_state=SEED)
        notes.append(f"Correlation uses a seeded sample of {MAX_CORRELATION_ROWS:,} rows.")
    corr = corr_frame.corr(min_periods=3)
    pairs = [
        {"feature_a": a, "feature_b": b, "correlation": float(corr.loc[a, b])}
        for i, a in enumerate(numeric) for b in numeric[i + 1:]
        if pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) >= 0.9
    ]
    # Hash candidates in linear time, then confirm equality to guard hash collisions.
    buckets: dict[int, list[str]] = {}
    duplicates = []
    for name in frame:
        key = int(pd.util.hash_pandas_object(frame[name], index=False).sum())
        for previous in buckets.get(key, []):
            if frame[name].equals(frame[previous]):
                duplicates.append((str(previous), str(name)))
                break
        buckets.setdefault(key, []).append(name)
    return Profile(len(frame), len(frame.columns), int(frame.memory_usage(deep=True).sum()),
                   int(frame.duplicated().sum()), table, corr, pairs, duplicates, notes)
