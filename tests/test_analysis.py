import numpy as np
import pandas as pd
import pytest

from dataguard.demo import demo_dataset
from dataguard.ingestion import DatasetError, read_csv
from dataguard.profiling import profile_dataset
from dataguard.quality import health_score, quality_findings
from dataguard.readiness import assess_readiness, infer_problem


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_csv_delimiters_and_quoted_headers(delimiter):
    data = f'name{delimiter}value\n"Alan, A"{delimiter}3\nBob{delimiter}4\n'.encode()
    frame = read_csv(data)
    assert frame.shape == (2, 2)
    assert frame.iloc[0, 0] == "Alan, A"


def test_bom_latin1_and_explicit_encoding():
    assert read_csv(b"\xef\xbb\xbfname,x\nA,2\n").columns[0] == "name"
    assert read_csv("name,x\ncaf\xe9,2\n".encode("latin-1")).iloc[0, 0] == "caf\xe9"
    assert "Latin-1" in read_csv(b"name,x\ncaf\xe9,2\n").attrs["ingestion"]["encoding"]


@pytest.mark.parametrize(
    "payload", [b"", b"a,b\n", b"a,a\n1,2\n", b",b\n1,2\n", b"\x00bad", b"a,b\n1,2\n3,4,5\n"]
)
def test_reject_invalid_csv(payload):
    with pytest.raises(DatasetError):
        read_csv(payload)


def test_ingestion_limits(monkeypatch):
    import dataguard.ingestion as ingestion

    monkeypatch.setattr(ingestion, "MAX_ROWS", 2)
    with pytest.raises(DatasetError, match="rows"):
        read_csv(b"x\n1\n2\n3\n")
    monkeypatch.setattr(ingestion, "MAX_COLUMNS", 1)
    with pytest.raises(DatasetError, match="column"):
        read_csv(b"x,y\n1,2\n")
    monkeypatch.setattr(ingestion, "MAX_UPLOAD_BYTES", 2)
    with pytest.raises(DatasetError, match="upload"):
        read_csv(b"x\n1\n")


def test_profile_measured_issues_and_preserves_input():
    frame = pd.DataFrame(
        {
            "x": [0.0, 1, 2, 3, 4, 100, np.inf, np.nan],
            "constant": ["a"] * 8,
            "date": ["2024-01-01"] * 8,
        }
    )
    frame["x_copy"] = frame.x
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    original = frame.copy(deep=True)
    profile = profile_dataset(frame)
    assert profile.rows == 9 and profile.columns == 4
    assert profile.duplicate_rows == 1
    assert profile.column_stats.loc["x", "missing"] == 1
    assert profile.column_stats.loc["x", "invalid"] == 1
    assert profile.column_stats.loc["x", "outliers"] == 1
    assert profile.column_stats.loc["date", "kind"] == "datetime"
    assert ("x", "x_copy") in profile.duplicate_columns
    pd.testing.assert_frame_equal(frame, original)
    assert any(f["issue"] == "Infinite numeric values" for f in quality_findings(profile))


def test_zero_iqr_all_missing_and_zero_heavy():
    profile = profile_dataset(pd.DataFrame({"x": [0.0] * 99 + [100.0], "empty": [np.nan] * 100}))
    assert profile.column_stats.loc["x", "outliers"] == 0
    assert profile.column_stats.loc["x", "zero_pct"] == 99
    assert profile.column_stats.loc["empty", "constant"]
    assert profile.column_stats.loc["empty", "missing_pct"] == 100


def test_mixed_types_cardinality_and_identifiers():
    frame = pd.DataFrame(
        {
            "account_id": [f"id-{i}" for i in range(100)],
            "text": [f"text-{i}" for i in range(100)],
            "mixed": ["1"] * 95 + ["error"] * 5,
            "flag": [True, False] * 50,
        }
    )
    stats = profile_dataset(frame).column_stats
    assert stats.loc["account_id", "id_like"]
    assert stats.loc["text", "high_cardinality"]
    assert stats.loc["mixed", "mixed_numeric"]
    assert stats.loc["flag", "kind"] == "boolean"


def test_score_formula_and_monotonic_missingness():
    frame = pd.DataFrame({"x": [1.0, 3, 2, 4], "y": ["a", "b", "a", "b"]})
    clean = health_score(profile_dataset(frame))
    assert clean.value == 100
    frame.loc[0, "x"] = np.nan
    scored = health_score(profile_dataset(frame))
    assert scored.value == round(100 - 35 / 8, 1)
    assert scored.value < clean.value
    assert scored.breakdown.max_penalty.sum() == 100
    assert scored.breakdown.penalty.ge(0).all()


@pytest.mark.parametrize(
    "task,target", [("classification", "churn"), ("regression", "annual_value")]
)
def test_demo_target_readiness(task, target):
    frame = demo_dataset(task)
    profile = profile_dataset(frame)
    readiness = assess_readiness(frame, profile, target)
    assert readiness.problem.task == task
    assert 0 <= readiness.score.value <= 100
    assert readiness.score.breakdown.max_penalty.sum() == 100
    assert any(item["column"] == "target_copy_review" for item in readiness.leakage)


def test_unsupported_and_ambiguous_targets():
    frame = pd.DataFrame({"y": [0] * 40, "x": range(40)})
    assert infer_problem(frame, "missing").task == "unsupported"
    assert infer_problem(frame, "y").task == "unsupported"
    assert infer_problem(frame.head(5), "x").task == "unsupported"
    frame.y = [0] * 39 + [1]
    assert infer_problem(frame, "y").task == "unsupported"
    frame.y = [0.2, 0.7] * 20
    assert infer_problem(frame, "y").task == "ambiguous"
    assert infer_problem(frame, "y", "regression").task == "regression"
    frame.y = ["a", "b"] * 20
    assert infer_problem(frame, "y", "regression").task == "unsupported"


def test_correlation_caps(monkeypatch):
    import dataguard.profiling as profiling

    monkeypatch.setattr(profiling, "MAX_CORRELATION_COLUMNS", 2)
    monkeypatch.setattr(profiling, "MAX_CORRELATION_ROWS", 20)
    profile = profile_dataset(pd.DataFrame(np.random.default_rng(2).normal(size=(100, 4))))
    assert profile.correlation.shape == (2, 2)
    assert len(profile.notes) == 2
