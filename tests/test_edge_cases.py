import numpy as np
import pandas as pd
import pytest

from dataguard.evaluation import classification_metrics
from dataguard.ingestion import DatasetError
from dataguard.modeling import split_data, train_baselines
from dataguard.preprocessing import make_preprocessor, normalize_features
from dataguard.profiling import profile_dataset
from dataguard.readiness import assess_readiness, infer_problem


def test_all_missing_target_and_no_predictors():
    frame = pd.DataFrame({"x": range(40), "y": [np.nan] * 40})
    assessment = assess_readiness(frame, profile_dataset(frame), "y")
    assert assessment.score.value == 0
    assert assessment.problem.task == "unsupported"
    assert infer_problem(pd.DataFrame({"y": [0, 1] * 20}), "y").task == "unsupported"


def test_single_independent_group_has_actionable_error():
    with pytest.raises(DatasetError, match="independent predictor groups"):
        split_data(pd.DataFrame({"x": [1] * 40}), pd.Series([0, 1] * 20), "classification")


def test_unseen_categories_with_no_rare_bucket():
    train = pd.DataFrame({"label": ["a", "b"] * 10})
    test = pd.DataFrame({"label": ["unseen", None]})
    transform = make_preprocessor(normalize_features(train))
    transform.fit(normalize_features(train))
    values = transform.transform(normalize_features(test))
    assert values.shape == (2, 2)
    assert np.isfinite(values).all()


def test_all_missing_numeric_column_preprocessing_is_finite():
    train = pd.DataFrame({"empty": [np.nan] * 4, "x": [1.0, 2, 3, 4]})
    transform = make_preprocessor(train)
    assert np.isfinite(transform.fit_transform(train)).all()


def test_auc_with_missing_holdout_class_is_unavailable():
    with pytest.warns(UserWarning, match="single label"):
        metrics = classification_metrics(
            np.array(["a", "a"]),
            np.array(["a", "a"]),
            np.array([[0.7, 0.3], [0.8, 0.2]]),
            np.array(["a", "b"]),
        )
    assert metrics["roc_auc_ovr_macro"] is None


def test_training_omits_invalid_targets_without_mutation():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)})
    frame.loc[0, "y"] = np.inf
    frame.loc[1, "y"] = np.nan
    run = train_baselines(frame, "y")
    assert run.train_rows + run.test_rows == 58
    assert np.isinf(frame.loc[0, "y"])


def test_empty_frame_profile_fails():
    with pytest.raises(DatasetError):
        profile_dataset(pd.DataFrame())
