"""Metric definitions shared by model comparison and report generation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def classification_metrics(y_true, predictions, probabilities, classes) -> dict[str, float | None]:
    """Macro metrics give minority classes equal weight; AUC uses estimator class order."""
    auc = None
    if len(np.unique(y_true)) == len(classes):
        if len(classes) == 2:
            auc = float(roc_auc_score(np.asarray(y_true) == classes[1], probabilities[:, 1]))
        else:
            auc = float(
                roc_auc_score(
                    y_true, probabilities, labels=classes, multi_class="ovr", average="macro"
                )
            )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_macro": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": auc,
    }


def regression_metrics(y_true, predictions) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }
