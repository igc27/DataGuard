# v1.0.0 validation record

Reference system: Windows, CPython 3.12.10. Dependencies are recorded in
`requirements-lock.txt`. This record reports checks actually performed locally;
it does not claim a remote CI run or a live hosted deployment.

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
| GitHub Actions references | Official checkout/setup-python v7 refs verified |
| License and attribution | MIT; Copyright (c) 2026 Mohammed Alanazi |

Demo outputs can be regenerated with `python scripts/validate_demo.py`. The
generated reports are intentionally ignored by Git because they embed several
megabytes of third-party Plotly JavaScript and are reproducible.

Python 3.11 and 3.13 are configured in remote CI and all pinned packages declare
compatible Python ranges. Only Python 3.12 was executed on this machine; the
additional CI environments must be verified once the repository is published.
