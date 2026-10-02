import pandas as pd

from dataguard.demo import demo_dataset
from dataguard.profiling import profile_dataset
from dataguard.quality import health_score
from dataguard.readiness import assess_readiness
from dataguard.reporting import build_report
from dataguard.visualization import (
    correlation_chart,
    distribution_chart,
    missing_chart,
    outlier_chart,
)


def test_report_escapes_labels_excludes_raw_records_and_is_offline():
    frame = demo_dataset()
    frame = frame.rename(columns={"plan": "<script>alert(1)</script>"})
    profile = profile_dataset(frame)
    report = build_report(
        profile,
        health_score(profile),
        dataset_name="<script>bad()</script>",
        readiness=assess_readiness(frame, profile, "churn"),
    )
    assert "<script>alert(1)</script>" not in report
    assert "<script>bad()</script>" not in report
    assert "&lt;script&gt;bad()&lt;/script&gt;" in report
    assert "ACC-00001" not in report
    assert "<script src=" not in report
    assert "plotly.js" in report
    assert "Mohammed Alanazi" in report
    assert "Health score breakdown" in report
    assert "ML readiness" in report


def test_chart_data_encodes_validly():
    frame = demo_dataset()
    profile = profile_dataset(frame)
    charts = [
        missing_chart(profile),
        correlation_chart(profile),
        outlier_chart(profile),
        distribution_chart(frame, "monthly_charge", "numeric"),
        distribution_chart(frame, "plan", "categorical"),
    ]
    for chart in charts:
        assert len(chart.data) > 0
        assert len(chart.to_json()) > 100


def test_report_without_target_or_training():
    profile = profile_dataset(demo_dataset())
    report = build_report(profile, health_score(profile), include_charts=False)
    assert "No target selected" in report
    assert "Training was not performed" in report


def test_nullable_boolean_distribution():
    frame = pd.DataFrame({"flag": pd.Series([True, False, None], dtype="boolean")})
    chart = distribution_chart(frame, "flag", "boolean")
    assert sum(chart.data[0].y) == 3
