# DataGuard v1.0.0 — Initial Public Release

DataGuard is an automated Dataset Health & Machine Learning Readiness Analyzer
by Mohammed Alanazi. This release provides a complete local Streamlit workflow
from CSV upload to a standalone analysis report.

## Included

- CSV ingestion with encoding/delimiter overrides, integrity checks, and resource limits.
- Profiling for structure, types, missingness, duplicates, infinities, cardinality,
  constants, identifiers, mixed numeric tokens, distribution statistics, IQR
  candidates, duplicate columns, and bounded Pearson correlation.
- Explainable health and target-dependent readiness scores with displayed factor breakdowns.
- Classification/regression inference, conservative leakage clues, and preparation recommendations.
- Train-only feature audits and scikit-learn preprocessing pipelines for numerical
  and categorical features, including unseen categories.
- Three learned baselines plus dummy controls, macro classification metrics and
  valid ROC-AUC, regression MAE/RMSE/R², confusion matrices, and model comparison.
- Grouped identical original predictors to reduce duplicate train/test contamination.
- Exploratory holdout permutation importance with explicit validity limits.
- Six-view Streamlit dashboard, interactive Plotly charts, and self-contained HTML reports.
- Two MIT-licensed, wholly synthetic demonstrations with documented injected issues.
- Tests, reproducible dependency pins, GitHub Actions configuration, and methodology documentation.

## Validation

49 automated tests pass on Windows with Python 3.12.10. Core-package statement
coverage is 94.36%. Ruff lint and formatting checks pass; wheel and source
distributions build. CSV upload and dataset profiling were also exercised through
the running browser UI. Classification and regression demo workflows generate
four evaluated baselines each and offline HTML reports.

Remote GitHub Actions results must be verified after publication; local test
results are not represented as a remote CI run.

## Scope and privacy

This is dataset triage and exploratory baseline comparison, not production
model validation. Scores are documented rubrics rather than calibrated measures.
Leakage clues require domain/provenance review. Domain grouping and temporal
splits need custom validation. Input is bounded to 25 MiB / 100,000 rows / 200
columns; baseline training and correlation use documented limits.

Analysis uses application memory without external AI calls or dataset telemetry.
Hosted deployments process uploads on their server. Reports omit raw records
but contain labels and aggregate statistics.

MIT License — Copyright (c) 2026 Mohammed Alanazi.
