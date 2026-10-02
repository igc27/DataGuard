import numpy as np
import pandas as pd
import pytest

from dataguard.demo import demo_dataset
from dataguard.ingestion import DatasetError
from dataguard.modeling import split_data, train_baselines
from dataguard.preprocessing import make_preprocessor, normalize_features, select_features


def test_train_only_imputation_and_unknown_categories():
    train = pd.DataFrame({"x": [1.0, 3.0, np.nan], "category": ["a", "b", None]})
    test = pd.DataFrame({"x": [np.nan, 1000.0, np.inf], "category": ["new", None, "a"]})
    preprocessor = make_preprocessor(normalize_features(train))
    preprocessor.fit(normalize_features(train))
    imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == 2
    transformed = preprocessor.transform(normalize_features(test))
    assert transformed.shape[0] == 3
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize(
    "task,target", [("classification", "churn"), ("regression", "annual_value")]
)
def test_baseline_pipeline_and_input_preservation(task, target):
    frame = demo_dataset(task)
    original = frame.copy(deep=True)
    result = train_baselines(frame, target)
    assert result.task == task
    assert len(result.results) == 4
    assert not result.errors
    assert "target_copy_review" in result.excluded_features
    assert "account_id" in result.excluded_features
    assert target not in result.included_features
    assert result.train_rows + result.test_rows == len(frame)
    assert not result.importance.empty
    if task == "classification":
        assert result.confusion.values.sum() == result.test_rows
        assert result.results.f1_macro.between(0, 1).all()
        assert result.results.roc_auc_ovr_macro.notna().all()
    else:
        assert result.results.rmse.ge(0).all()
        assert result.results.iloc[0].rmse == result.results.rmse.min()
    pd.testing.assert_frame_equal(frame, original)


def test_multiclass_and_imbalance():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=150),
            "category": ["a", "b", "c"] * 50,
            "target": [0] * 120 + [1] * 20 + [2] * 10,
        }
    )
    result = train_baselines(frame, "target")
    assert len(result.results) == 4
    assert result.confusion.shape == (3, 3)
    assert "balanced_accuracy" in result.results
    assert result.results.roc_auc_ovr_macro.notna().all()


def test_duplicate_predictor_groups_never_overlap():
    x = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6] * 20})
    y = pd.Series([0, 0, 1, 1, 0, 1] * 20)
    train, test, _, _, note = split_data(x, y, "classification")
    assert not set(train.x) & set(test.x)
    assert "Grouped" in note


def test_leakage_audit_uses_training_rows_only():
    train = pd.DataFrame({"copy": [0, 1] * 20, "x": [1, 3, 2, 4] * 10})
    y = pd.Series([0, 1] * 20)
    kept, dropped = select_features(train, y)
    assert kept == ["x"] and "copy" in dropped


@pytest.mark.parametrize("target", ["absent", "one_class"])
def test_invalid_targets_fail_clearly(target):
    frame = pd.DataFrame({"x": range(40), "one_class": [1] * 40})
    with pytest.raises(DatasetError):
        train_baselines(frame, target)


def test_no_usable_predictors():
    frame = pd.DataFrame(
        {"constant": [1] * 50, "account_id": [f"ACC-{i}" for i in range(50)], "target": [0, 1] * 25}
    )
    with pytest.raises(DatasetError, match="No usable"):
        train_baselines(frame, "target")


def test_stratified_model_sampling(monkeypatch):
    import dataguard.modeling as modeling

    monkeypatch.setattr(modeling, "MAX_MODEL_ROWS", 100)
    result = train_baselines(demo_dataset(), "churn")
    assert result.train_rows + result.test_rows == 100
    assert any("sample" in note for note in result.notes)
