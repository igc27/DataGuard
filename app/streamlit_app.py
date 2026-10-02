"""DataGuard dashboard. UI orchestration only; analysis lives in the package."""

import hashlib
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from dataguard import __version__
from dataguard.demo import demo_dataset
from dataguard.ingestion import DatasetError, read_csv
from dataguard.modeling import train_baselines
from dataguard.profiling import profile_dataset
from dataguard.quality import health_score, quality_findings
from dataguard.readiness import assess_readiness
from dataguard.reporting import build_report
from dataguard.visualization import (
    correlation_chart,
    distribution_chart,
    importance_chart,
    missing_chart,
    outlier_chart,
    style,
)

st.set_page_config(page_title="DataGuard · Dataset intelligence", page_icon="◈", layout="wide")
st.markdown(
    """<style>
    .block-container{padding-top:4rem;max-width:1440px}
    [data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);border-radius:9px;padding:16px 20px}
    [data-testid="stMetricLabel"]{font-size:.8rem;letter-spacing:.02em}
    .eyebrow{font-size:.73rem;letter-spacing:.13em;color:#0d9488;font-weight:700;margin-bottom:6px}
    .subtitle{color:#8292a3;font-size:1rem;margin-top:-9px;margin-bottom:25px}
    h1{font-size:2.2rem!important;letter-spacing:-.035em}
    h2{font-size:1.4rem!important;letter-spacing:-.02em}
    .context{border-left:3px solid #14b8a6;padding:10px 16px;background:rgba(20,184,166,.06);margin-bottom:20px}
    </style>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ◈ DataGuard")
    st.caption("Dataset health & ML readiness")
    page = st.radio(
        "Workspace",
        ["Overview", "Data Quality", "EDA", "ML Readiness", "Model Lab", "Report"],
        label_visibility="collapsed",
    )
    st.divider()
    source = st.radio("Dataset source", ["Built-in demo", "Upload CSV"])
    task_demo = "classification"
    uploaded = None
    if source == "Built-in demo":
        demo_choice = st.selectbox(
            "Demo task", ["Classification · churn", "Regression · annual value"]
        )
        task_demo = "classification" if demo_choice.startswith("Classification") else "regression"
        inject = st.toggle("Include synthetic quality issues", value=True)
    else:
        uploaded = st.file_uploader(
            "Upload a CSV", type=["csv"], help="25 MiB · up to 100,000 rows and 200 columns"
        )
        with st.expander("CSV parsing options"):
            encoding = st.selectbox("Encoding", ["auto", "utf-8-sig", "latin-1", "cp1252"])
            separator = st.selectbox("Delimiter", ["auto", ",", ";", "\t", "|"])
    st.divider()
    st.caption(
        "Data stays in application memory. No external AI calls. On a hosted instance, processing runs on that server."
    )
    st.caption(f"v{__version__} · Mohammed Alanazi")

st.markdown('<div class="eyebrow">DATASET INTELLIGENCE / WORKSPACE</div>', unsafe_allow_html=True)
st.title(page)
st.markdown(
    '<div class="subtitle">Understand the data. Inspect the risks. Establish a baseline.</div>',
    unsafe_allow_html=True,
)

if source == "Upload CSV" and uploaded is None:
    st.info("Upload a CSV from the sidebar to begin, or explore a built-in demo.")
    st.markdown(
        "DataGuard profiles your dataset, explains quality issues, and builds supervised-learning baselines when you select a target."
    )
    st.stop()

try:
    if source == "Built-in demo":
        frame = demo_dataset(task_demo, issues=inject)
        name = f"Synthetic subscription accounts · {task_demo}"
        fingerprint = f"demo-{task_demo}-{inject}"
    else:
        payload = uploaded.getvalue()
        frame = read_csv(payload, encoding=encoding, delimiter=separator)
        name = uploaded.name
        fingerprint = hashlib.sha256(payload + encoding.encode() + separator.encode()).hexdigest()
except DatasetError as exc:
    st.error(str(exc))
    st.stop()

# Session-local analysis avoids a global shared cache of potentially sensitive datasets.
if st.session_state.get("dataset_key") != fingerprint:
    with st.spinner("Profiling dataset…"):
        st.session_state.profile = profile_dataset(frame)
        st.session_state.health = health_score(st.session_state.profile)
    st.session_state.dataset_key = fingerprint
    st.session_state.pop("model_run", None)
    st.session_state.pop("model_key", None)
    st.session_state.pop("target_column", None)
    st.session_state.pop("feature_exclusions", None)

profile, health = st.session_state.profile, st.session_state.health
findings = quality_findings(profile)
with st.sidebar:
    no_target = "No target selected"
    while no_target in frame.columns:
        no_target += " "
    options = [no_target, *frame.columns.tolist()]
    default_target = "churn" if task_demo == "classification" else "annual_value"
    default_idx = options.index(default_target) if source == "Built-in demo" else 0
    selected_target = st.selectbox(
        "Target column (optional)",
        options,
        index=default_idx,
        key="target_column",
    )
    target = None if selected_target == no_target else selected_target
    override = st.selectbox("Problem type", ["auto", "classification", "regression"])

readiness = assess_readiness(frame, profile, target, override) if target is not None else None
run_key_base = (fingerprint, target, override)
run = st.session_state.get("model_run")
if run and st.session_state.get("model_key", ())[:3] != run_key_base:
    run = None
    st.session_state.pop("model_run", None)

st.markdown(
    f'<div class="context"><strong>{escape(name)}</strong> &nbsp;·&nbsp; {profile.rows:,} rows &nbsp;·&nbsp; {profile.columns} columns</div>',
    unsafe_allow_html=True,
)
if source == "Built-in demo":
    st.caption(
        "Fictional accounts generated with seed 42. Missingness, duplicates, extremes, constants, and a target-copy column are deliberately injected when the issue toggle is enabled."
    )

if page == "Overview":
    cards = st.columns(4)
    cards[0].metric("Dataset health", f"{health.value:.1f} / 100")
    cards[1].metric(
        "Missing cells",
        f"{profile.column_stats.missing.sum():,}",
        f"{profile.column_stats.missing.sum() / (profile.rows * profile.columns):.1%} of all cells",
        delta_color="off",
    )
    cards[2].metric("Duplicate rows", f"{profile.duplicate_rows:,}")
    cards[3].metric("Memory footprint", f"{profile.memory_bytes / 1024**2:.2f} MiB")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Data preview")
        st.dataframe(frame.head(50), width="stretch", hide_index=True)
        st.caption("First 50 records. Uploaded data is never edited in place.")
    with right:
        st.subheader("Column types")
        kinds = (
            profile.column_stats.kind.value_counts().rename_axis("type").reset_index(name="columns")
        )
        st.plotly_chart(
            style(
                px.bar(kinds, x="type", y="columns", color_discrete_sequence=["#14b8a6"]),
                "Inferred semantic types",
            ),
            use_container_width=True,
        )
    st.subheader("What needs attention")
    priority = [finding for finding in findings if finding["severity"] == "warning"]
    if priority:
        st.dataframe(
            pd.DataFrame(priority).head(8)[["column", "issue", "evidence", "action"]],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Showing {min(8, len(priority))} of {len(priority)} warning findings. Review the complete audit under Data Quality."
        )
    else:
        st.success(
            "No configured warning rules were triggered. Review EDA and domain constraints before using the data."
        )
    with st.expander("Dataset structure and ingestion details"):
        st.dataframe(
            profile.column_stats[["dtype", "kind", "unique", "cardinality"]], width="stretch"
        )
        st.write(frame.attrs.get("ingestion", {"source": "MIT-licensed synthetic demo"}))

elif page == "Data Quality":
    left, right = st.columns(2)
    with left:
        st.subheader("Missingness")
        if profile.column_stats.missing.any():
            st.plotly_chart(missing_chart(profile), use_container_width=True)
        else:
            st.success("No missing cells detected.")
    with right:
        st.subheader("Outlier candidates")
        if profile.column_stats.outliers.any():
            st.plotly_chart(outlier_chart(profile), use_container_width=True)
        else:
            st.info("No values outside nonzero IQR fences.")
        st.caption(
            "1.5×IQR on finite values. Zero IQR has no stable fence. Candidates are never automatically deleted."
        )
    st.subheader("Quality audit")
    if findings:
        st.dataframe(pd.DataFrame(findings), width="stretch", hide_index=True)
    else:
        st.success("No configured issues detected.")
    with st.expander("Measured column statistics", expanded=False):
        st.dataframe(profile.column_stats, width="stretch")
    with st.expander("How the health score is calculated"):
        st.write(
            "100 minus point budget × measured rate for each factor. Outliers and correlation carry no penalty because their validity depends on the domain."
        )
        st.dataframe(health.breakdown, width="stretch", hide_index=True)

elif page == "EDA":
    column = st.selectbox("Explore a feature", frame.columns.tolist())
    st.plotly_chart(
        distribution_chart(frame, column, profile.column_stats.loc[column, "kind"]),
        use_container_width=True,
    )
    st.caption(
        "Numeric plots use at most 10,000 seeded finite observations. Categorical plots show the top 20 values; omitted categories are excluded from this chart only."
    )
    if not profile.correlation.empty:
        st.plotly_chart(correlation_chart(profile), use_container_width=True)
        if profile.correlated_pairs:
            st.dataframe(pd.DataFrame(profile.correlated_pairs), width="stretch", hide_index=True)
    else:
        st.info("No numeric columns for correlation analysis.")
    for note in profile.notes:
        st.info(note)
    st.caption(
        "Pearson correlation uses pairwise complete finite values. Strong association can reflect valid signal, redundancy, or leakage; provenance determines the interpretation."
    )

elif page == "ML Readiness":
    if readiness is None:
        st.info("Select a target column in the sidebar to assess supervised-learning readiness.")
    else:
        cards = st.columns(3)
        cards[0].metric("ML readiness", f"{readiness.score.value:.1f} / 100")
        cards[1].metric("Inferred task", readiness.problem.task.title())
        cards[2].metric("Usable target rows", f"{readiness.problem.valid_rows:,}")
        st.info(readiness.status + ". " + readiness.problem.reason)
        st.subheader("Recommended preparation")
        for recommendation in readiness.recommendations:
            st.write("• " + recommendation)
        st.plotly_chart(
            distribution_chart(
                frame,
                target,
                "numeric" if readiness.problem.task == "regression" else "categorical",
            ),
            use_container_width=True,
        )
        if readiness.leakage:
            st.warning(
                "Potential leakage clues require review. These checks do not establish that leakage exists."
            )
            st.dataframe(pd.DataFrame(readiness.leakage), width="stretch", hide_index=True)
        with st.expander("Readiness score breakdown"):
            st.dataframe(readiness.score.breakdown, width="stretch", hide_index=True)
            st.caption(
                "Unsupported or ambiguous targets receive 0 regardless of factor totals. A high score does not guarantee a valid split, predictive signal, or deployment suitability."
            )

elif page == "Model Lab":
    if readiness is None:
        st.info("Select a target column in the sidebar to run baseline models.")
    elif readiness.problem.task not in {"classification", "regression"}:
        st.warning(readiness.problem.reason)
    else:
        st.subheader(f"{readiness.problem.task.title()} baselines")
        st.write(
            "Compare a linear model, random forest, extra trees, and a dummy control on one holdout. Preprocessing statistics and category vocabularies are learned from training rows."
        )
        exclusions = st.multiselect(
            "Additional features to exclude",
            [name for name in frame if name != target],
            key="feature_exclusions",
            help="Review timestamps, outcome proxies, and features unavailable at prediction time.",
        )
        desired_key = (*run_key_base, tuple(exclusions))
        if run and st.session_state.get("model_key") != desired_key:
            run = None
            st.session_state.pop("model_run", None)
        st.caption(
            "Up to 5,000 seeded rows and 80 usable features. Identical predictor records are grouped across the split. ID candidates, constants, datetime, high-cardinality categories, and exact target copies are excluded using the training audit."
        )
        if st.button("Run baseline comparison", type="primary"):
            try:
                with st.status("Running baseline comparison…", expanded=True) as status:
                    run = train_baselines(
                        frame,
                        target,
                        override=override,
                        excluded=exclusions,
                        progress=lambda model: st.write(f"Evaluating {model}"),
                    )
                    st.session_state.model_run = run
                    st.session_state.model_key = desired_key
                    status.update(
                        label="Baseline comparison complete", state="complete", expanded=False
                    )
            except DatasetError as exc:
                st.error(str(exc))
        if run:
            st.success(
                f"Top baseline: {run.winner} · ranked by {'highest macro F1' if run.task == 'classification' else 'lowest RMSE'} on this holdout."
            )
            st.dataframe(run.results, width="stretch", hide_index=True)
            st.plotly_chart(
                style(
                    px.bar(
                        run.results,
                        x="model",
                        y=run.ranking_metric,
                        color_discrete_sequence=["#14b8a6"],
                    ),
                    f"Baseline comparison · {run.ranking_metric}",
                ),
                use_container_width=True,
            )
            if run.confusion is not None:
                st.plotly_chart(
                    style(
                        px.imshow(
                            run.confusion,
                            text_auto=True,
                            color_continuous_scale="Teal",
                            labels={"x": "Predicted", "y": "Actual"},
                            aspect="auto",
                        ),
                        "Top baseline confusion matrix",
                    ),
                    use_container_width=True,
                )
            if not run.importance.empty:
                st.plotly_chart(importance_chart(run.importance), use_container_width=True)
            with st.expander("Training audit, limits, and evaluation notes", expanded=False):
                st.write(f"{run.train_rows:,} training / {run.test_rows:,} holdout rows")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"feature": name, "reason": reason}
                            for name, reason in run.excluded_features.items()
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                for note in run.notes:
                    st.write("• " + note)
            if run.errors:
                st.warning("Some baselines could not be evaluated.")
                st.write(run.errors)
        else:
            st.info(
                "Run a comparison to see metrics, the feature audit, and holdout permutation importance."
            )

elif page == "Report":
    st.subheader("Export the analysis")
    st.write(
        "Download a standalone HTML report with quality findings, score breakdowns, interactive charts, recommendations, and model results when available."
    )
    st.caption(
        "Includes aggregate statistics, column names, and class labels; excludes raw records. Review potentially sensitive labels before sharing. Reports open offline."
    )
    cards = st.columns(3)
    cards[0].metric("Health score", f"{health.value:.1f}")
    cards[1].metric("Readiness", f"{readiness.score.value:.1f}" if readiness else "No target")
    cards[2].metric("Baselines evaluated", len(run.results) if run else 0)
    with st.spinner("Preparing report…"):
        report = build_report(
            profile, health, dataset_name=name, readiness=readiness, model_run=run
        )
    st.download_button(
        "Download HTML report",
        report,
        file_name="dataguard-report.html",
        mime="text/html",
        type="primary",
    )
    st.download_button(
        "Download column profile (CSV)",
        profile.column_stats.to_csv().encode(),
        file_name="dataguard-column-profile.csv",
        mime="text/csv",
    )
    st.caption(
        "Methodology v1.0.0 · Scores are transparent triage indicators. Baseline comparisons use an exploratory holdout, not cross-validated production performance."
    )
