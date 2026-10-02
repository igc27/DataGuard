# v1.0.0 validation record

Reference system: Windows, CPython 3.12.10. Dependencies are recorded in
`requirements-lock.txt`. This record reports completed local checks and the successful hosted GitHub
Actions run. A live hosted application remains a separate deployment step.

| Check | Verified result |
|---|---|
| Complete automated suite | 49 passed |
| Core package statement coverage | 94.36% |
| Second clean virtual environment | Installed dependency pins and editable package; 49 tests passed |
| Dependency integrity | `pip check`: no broken requirements |
| Ruff lint | Pass |
| Ruff formatting | Pass |
| Package build | Isolated wheel and source build succeeded |
| Application startup | HTTP server running on 127.0.0.1:8501 |
| Browser CSV upload | Uploaded bundled CSV; 612 rows / 12 columns profiled |
| Demo health score | 97.2 / 100, with factor breakdown |
| Classification | Four baseline results, confusion matrix, and permutation importance |
| Regression | Four baseline results, MAE/RMSE/R², and permutation importance |
| Streamlit navigation and target controls | All six views exercised; target changes invalidate results |
| Report generation | Classification and regression standalone HTML reports generated |
| Report safety | HTML escaping, omitted raw records, and embedded Plotly verified by tests |
| Charts | Browser rendering and serializable Plotly data checked |
| README preview | Actual application screenshot, not a mockup |
| GitHub Actions | All jobs passed on Linux with Python 3.11, 3.12, and 3.13 |
| License and attribution | MIT; Copyright (c) 2026 Mohammed Alanazi |

Demo outputs can be regenerated with `python scripts/validate_demo.py`. The
generated reports are intentionally ignored by Git because they embed several
megabytes of third-party Plotly JavaScript and are reproducible.

Hosted validation completed successfully on Python 3.11, 3.12, and 3.13:
[GitHub Actions run](https://github.com/igc27/DataGuard/actions/runs/37065945467).
Each job installed the pinned dependencies, checked Ruff lint and formatting, ran
the 49-test suite with the 80% coverage gate, built distributions, and passed
`pip check`. Local Windows validation used Python 3.12.10.
