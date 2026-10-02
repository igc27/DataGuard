# Explainable scoring, version 1.0.0

These scores prioritize review. They are not probabilities, model quality
estimates, compliance certificates, or universal indicators of dataset value.
Weights are documented product choices, not empirically calibrated thresholds.
The application shows every measured rate, point budget, and resulting penalty.

## Dataset health

`score = round(max(0, 100 − Σ(weight × rate)), 1)`

| Factor | Maximum penalty | Rate |
|---|---:|---|
| Missing cells | 35 | Missing cells / all cells |
| Duplicate rows | 20 | Rows beyond first occurrence / rows |
| Infinite cells | 15 | Infinite numeric cells / all cells |
| Constant or empty columns | 10 | Affected columns / all columns |
| Near-constant columns | 5 | Affected columns / all columns |
| Possible identifiers | 5 | Flagged columns / all columns |
| High-cardinality categories | 5 | Flagged columns / all columns |
| Possible mixed numeric types | 5 | Flagged columns / all columns |

Each rate lies in [0, 1] and maximum penalties sum to 100. For example, 10%
missing cells contribute 3.5 points; 5% duplicated rows contribute 1 point.
Different feature issues may overlap and receive separate penalties because
they describe separate preparation burdens. Adding irrelevant columns can
dilute column/cell rates; the score should not be optimized as an objective.

Outlier candidates, skewness, zeros, duplicate columns, and strong correlation
are displayed but carry no direct penalty. They can be legitimate properties.
Class imbalance is assessed in target-dependent readiness rather than giving
an untargeted dataset a speculative imbalance penalty.

## ML readiness

For a supported target: the same weighted-rate formula. The target is excluded
from feature-factor denominators. For unsupported or ambiguous problems, score
is **0** regardless of the displayed raw factor penalties; the validity gate and
its reason are shown explicitly.

| Factor | Maximum penalty | Rate |
|---|---:|---|
| Feature missingness | 20 | Missing feature cells / all feature cells |
| Infinite feature values | 10 | Infinite feature cells / all feature cells |
| Unusable / ID / datetime features | 15 | Union of constant, ID-like, datetime features / feature count |
| High-cardinality features | 10 | Flagged features / feature count |
| Unusable target rows | 15 | Missing or infinite target rows / all rows |
| Class imbalance | 15 | `max(0, 1 − K × smallest_class_fraction)` |
| Target leakage clues | 15 | `min(1, flagged_feature_count / feature_count)` |

Here K is the observed class count; imbalance is zero for regression. Uniform
class frequencies have no imbalance penalty. A 99:1 binary target contributes
14.7 points. Leakage clues are heuristics; their penalties represent review
effort, not a claim that leakage has been established. Empty feature denominators
use 1 to avoid undefined arithmetic; the training audit still refuses a model
when no usable predictors remain.

Scores ≥75 show “Good, preprocessing recommended”; lower scores show “Review
issues before relying on baselines.” Even a high score may have no predictive
signal or a failed split. For example, a dataset with many rows but one unique
predictor group cannot support independent holdout evaluation. Readiness does
not replace Model Lab validation or domain assessment.
