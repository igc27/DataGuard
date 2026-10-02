"""Escaped, standalone HTML reports with embedded Plotly and no external resources."""

from datetime import UTC, datetime
from html import escape

import pandas as pd
import plotly.io as pio

from dataguard import __version__
from dataguard.modeling import ModelRun
from dataguard.profiling import Profile
from dataguard.quality import Score, quality_findings
from dataguard.readiness import Readiness
from dataguard.visualization import correlation_chart, importance_chart, missing_chart, outlier_chart


def _table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, escape=True, border=0, float_format=lambda value: f"{value:.3f}")


def build_report(profile: Profile, health: Score, *, dataset_name: str = "Dataset",
                 readiness: Readiness | None = None, model_run: ModelRun | None = None,
                 include_charts: bool = True) -> str:
    """Summarize aggregate analysis; no raw records are included. Names and labels are escaped."""
    title = escape(dataset_name)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        f"<h1>DataGuard <span>Dataset health report</span></h1><p>{title} · {generated} · v{__version__}</p>",
        f"<div class='metrics'><strong>{profile.rows:,}<small>Rows</small></strong><strong>{profile.columns}<small>Columns</small></strong><strong>{health.value:.1f}/100<small>Health score</small></strong><strong>{profile.duplicate_rows:,}<small>Duplicate rows</small></strong></div>",
        "<h2>Dataset overview</h2>" + _table(profile.column_stats.reset_index()),
        "<h2>Health score breakdown</h2><p>100 minus weighted measured rates. A quality triage indicator, not a fitness guarantee. Outliers and correlation are diagnostic only.</p>" + _table(health.breakdown),
        "<h2>Detected issues and actions</h2>" + (_table(pd.DataFrame(quality_findings(profile))) if quality_findings(profile) else "<p>No configured rules were triggered.</p>"),
        "<h2>EDA highlights</h2><p>Finite numeric values use 1.5×IQR fences; zero IQR is not scored as an outlier. Correlation is Pearson, pairwise complete, with at least three paired rows.</p>",
        _table(pd.DataFrame(profile.correlated_pairs)) if profile.correlated_pairs else "<p>No numeric pairs exceeded |r| ≥ 0.90 within the analyzed limits.</p>",
    ]
    if profile.notes:
        sections.append("<ul>" + "".join(f"<li>{escape(note)}</li>" for note in profile.notes) + "</ul>")
    if readiness:
        sections.extend([
            f"<h2>ML readiness · {readiness.score.value:.1f}/100</h2><p>{escape(readiness.status)} · {escape(readiness.problem.task)}</p><p>{escape(readiness.problem.reason)}</p>",
            _table(readiness.score.breakdown),
            "<h3>Recommendations</h3><ul>" + "".join(f"<li>{escape(item)}</li>" for item in readiness.recommendations) + "</ul>",
        ])
        if readiness.leakage:
            sections.append("<h3>Possible leakage · requires provenance review</h3>" + _table(pd.DataFrame(readiness.leakage)))
    else:
        sections.append("<h2>ML readiness</h2><p>No target selected. Select a target to assess supervised-learning readiness.</p>")
    if model_run:
        sections.extend([
            f"<h2>Baseline comparison</h2><p>Target: {escape(model_run.target)} · {model_run.train_rows:,} training / {model_run.test_rows:,} holdout rows. Ranked by {escape(model_run.ranking_metric)}; top baseline: {escape(model_run.winner)}.</p>",
            _table(model_run.results),
            "<h3>Feature audit</h3>" + _table(pd.DataFrame([{"feature": name, "reason": reason} for name, reason in model_run.excluded_features.items()])),
            "<ul>" + "".join(f"<li>{escape(note)}</li>" for note in model_run.notes) + "</ul>",
        ])
        if model_run.confusion is not None:
            sections.append("<h3>Confusion matrix · rows actual, columns predicted</h3>" + model_run.confusion.to_html(escape=True, border=0))
        if not model_run.importance.empty:
            sections.append("<h3>Permutation importance</h3>" + _table(model_run.importance))
        if model_run.errors:
            sections.append("<h3>Models that could not be evaluated</h3>" + _table(pd.DataFrame([{"model": name, "error": error} for name, error in model_run.errors.items()])))
    else:
        sections.append("<h2>Model results</h2><p>Training was not performed for this report.</p>")
    if include_charts:
        charts = []
        if profile.column_stats.missing.any():
            charts.append(missing_chart(profile))
        if not profile.correlation.empty:
            charts.append(correlation_chart(profile))
        if profile.column_stats.outliers.any():
            charts.append(outlier_chart(profile))
        if model_run and not model_run.importance.empty:
            charts.append(importance_chart(model_run.importance))
        for i, fig in enumerate(charts):
            sections.append(pio.to_html(fig, full_html=False, include_plotlyjs=True if i == 0 else False,
                                        config={"displaylogo": False, "responsive": True}))
    sections.append(f"<footer>DataGuard {__version__} · Mohammed Alanazi · MIT. Analysis runs in application memory without external AI APIs. Reports contain aggregate statistics and column/class labels that may still be sensitive. Review before sharing. Heuristics cannot establish leakage or deployment suitability. Baselines are exploratory; temporal and domain group splitting require further work.</footer>")
    css = """
    :root{color-scheme:light}body{font:15px/1.65 Arial,sans-serif;color:#182b3a;background:#f5f7f9;margin:0;padding:40px}
    main{max-width:1200px;margin:auto;background:white;padding:40px;border:1px solid #dbe3e8;border-radius:12px}
    h1{color:#0f766e;font-size:30px}h1 span{display:block;color:#64748b;font-size:17px;font-weight:400}
    h2{margin-top:36px;font-size:21px;border-bottom:1px solid #e2e8f0;padding-bottom:10px}h3{font-size:17px}
    .metrics{display:flex;flex-wrap:wrap;gap:24px;padding:24px;background:#f0fdfa;border-radius:8px}
    .metrics strong{font-size:26px;flex:1;min-width:140px}.metrics small{display:block;font-size:12px;color:#64748b;font-weight:400}
    .table-scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;width:100%;margin:15px 0}th,td{padding:9px;text-align:left;border-bottom:1px solid #e2e8f0}th{background:#f1f5f9}footer{font-size:12px;color:#64748b;margin-top:40px}
    @media(max-width:700px){body{padding:8px}main{padding:18px}}@media print{body{padding:0;background:white}main{border:0}}
    """
    body = "".join(sections).replace('<table ', '<div class="table-scroll"><table ').replace('</table>', '</table></div>')
    return f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>DataGuard · {title}</title><style>{css}</style></head><body><main>{body}</main></body></html>"
