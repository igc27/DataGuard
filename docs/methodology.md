# Analysis and modeling methodology

## Profiling

Missingness uses pandas null detection; its default CSV null tokens apply.
Infinities are counted separately and excluded from summary statistics,
correlations, and IQR calculation. Numeric statistics include mean, sample
standard deviation, min/max, quartiles, median, skewness (when at least three
finite values and nonconstant), and zero proportion among finite values.

Cardinality is non-null distinct values / non-null rows. A constant feature has
at most one non-null distinct value, including all-missing columns. Near-constant
means at least 98% of non-null observations share one value, with more than one
distinct value. High-cardinality categories have >50 distinct values and a
cardinality ratio >0.5. Columns with 80–<100% numerically parseable non-null cells
receive a mixed-numeric-type heuristic warning; they are not silently converted.

ID candidates require at least 10 non-null observations and cardinality ≥0.98,
plus an ID-like name (`id`, `uuid`, `index`, `_id`, `_uuid`) or a monotonic integer
sequence. Unique continuous measurements alone are not flagged. This heuristic
can flag a legitimate monotonically sorted integer measurement; review it.

Datetime-like strings require date/time punctuation in at least 90% of the
first 200 non-null values and ≥90% successful date parsing. Boolean text recognizes
`true`/`false`. The UI exposes both pandas dtype and semantic type. Sampled type
inference can miss anomalies later in a column.

## Outliers and association

IQR fences are Q1 − 1.5×IQR and Q3 + 1.5×IQR using finite values. Counts and
percentages use finite observations as their denominator. For zero IQR, no
stable fence is reported and the IQR candidate count is zero. This avoids labeling
every rare value in a discrete or sparse feature an extreme outlier. It can
miss important extremes in zero-heavy features, which are reported separately.
No candidate is automatically deleted. IQR candidates are not necessarily errors.

Pearson correlation uses pairwise complete finite observations with at least
three valid pairs. Pairs with |r| ≥0.90 are listed. At most 40 numeric columns and
10,000 seeded rows are used; UI and reports expose those limits. This is not a
test of causality and can miss nonlinear association.

## Task inference and validity gates

At least 30 usable target rows and two distinct values are required. Missing and
infinite target rows are omitted from training and counted explicitly.
Categorical and boolean targets with ≤20 classes suggest classification.
Integer-valued numeric targets with ≤20 distinct values also suggest
classification, with a prompt to confirm their meaning. Fractional numeric
targets with ≤20 values are ambiguous. Numeric targets with >20 values suggest
regression. Users can explicitly select either task; classification is capped
at 20 classes and regression requires a numerical target.

One-class, singleton-class, excessive-cardinality categorical, and datetime
targets are unsupported. Each class needs two observations before attempting a
split. Actual group/split validation may impose a stronger restriction. Numerical
ordinal outcomes and counts require the user's domain decision.

## Leakage review

Exact target equality, numerical |r| ≥0.995 (at least 20 paired finite rows), and
outcome-like names are diagnostic clues. They do not prove leakage. A valid
predictor can be highly correlated with the target. Full-data diagnostic checks
inform review but are not used to choose baseline features automatically.

The training audit excludes exact target copies using training rows alone.
Near-perfect correlations are left for user review through explicit feature
exclusions. Domain knowledge must establish timing, provenance, and whether
features exist at prediction time. Nonlinear leakage and proxy outcomes can
escape these conservative heuristics.

## Split, features, and preprocessing

Stateless normalization creates a copy, converts infinities to null, and casts
categorical/boolean values to string labels. It learns no statistics. A seeded
75/25 split follows, stratified for classification. Identical raw predictor rows
are hash-grouped and kept in one partition. When groups exist, up to 30 seeded
group splits are attempted to put every class into both partitions. If no valid
split exists, training fails clearly. Hash collisions can group unrelated rows
conservatively; they cannot create cross-partition identical-row overlap.

Feature auditing uses training rows only: exclude constants, all-missing
features, ID candidates, datetime features, high-cardinality text/categories,
exact target copies, and user exclusions. At most 80 retained features are used.
Grouping is based on original predictors; after feature exclusions, distinct
records can still map to identical feature vectors. Domain identifiers,
repeated entities, and temporal dependence need a custom validation scheme.

Each model is a fresh scikit-learn `Pipeline` containing a `ColumnTransformer`.
Numerical predictors use training medians and `StandardScaler`. Categorical
predictors use training modes and `OneHotEncoder`, grouping rare values and
limiting each vocabulary to 30 output categories. Unseen categories are handled
by the fitted encoder. Forests do not require scaling, but affine scaling is
harmless and a shared preprocessing configuration keeps comparisons consistent.

## Baselines and metrics

Classification compares a dummy prior, class-weighted logistic regression,
class-weighted random forest, and class-weighted extra trees. Displayed metrics
are accuracy, balanced accuracy, macro precision/recall/F1, and ROC-AUC (binary
or multiclass one-vs-rest macro as appropriate). Binary AUC treats the estimator's
second sorted class as positive and uses its matching probability column. Macro
metrics equally weight classes. Undefined precision/recall contributions become
zero; AUC is unavailable when the holdout cannot represent every class.

Regression compares a dummy mean, Ridge, random forest, and extra trees, using
MAE, RMSE, and R². Classification ranks by highest macro F1; regression by lowest
RMSE. Dummy controls are included in ranking, so DataGuard can honestly show
when learned models fail to beat a naive reference. Per-model failures are shown
rather than replaced by fabricated results.

There is no hyperparameter tuning or cross-validation in v1.0.0. Ranking and
importance reuse one holdout, so it is exploratory model comparison. Reserve an
independent test set before making deployment decisions. There are no claimed
benchmark or real-world performance numbers.

## Explainability and reports

Permutation importance evaluates the highest-ranked learned baseline on at most
250 seeded holdout rows, three repeats, using macro F1 or negative RMSE. It
reports original input-column score decrease and repeat standard deviation.
Negative importance is retained. Correlated predictors can suppress one
another's importance. These are predictive associations, not causal effects.
Importance is skipped for a dummy winner or fewer than 20 holdout rows; errors
are described instead of inventing values.

Reports embed Plotly JavaScript and charts for offline use. Tables and textual
labels are HTML-escaped. No raw records are exported in HTML reports. Reports
include version, UTC generation time, scores, issues, actions, EDA highlights,
readiness, and available model results. Aggregate values and class names may
still disclose information; review before sharing.
