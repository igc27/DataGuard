"""Small, focused Plotly figures used by the UI and self-contained report."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dataguard.config import MAX_CORRELATION_ROWS, SEED
from dataguard.profiling import Profile

TEAL = "#14b8a6"


def style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=380,
        margin={"l": 20, "r": 20, "t": 55, "b": 25},
        font={"family": "Arial, sans-serif", "size": 12},
    )
    return fig


def missing_chart(profile: Profile) -> go.Figure:
    data = (
        profile.column_stats[profile.column_stats.missing > 0]
        .nlargest(20, "missing_pct")
        .reset_index()
    )
    return style(
        px.bar(
            data,
            x="missing_pct",
            y="column",
            orientation="h",
            color_discrete_sequence=[TEAL],
            labels={"missing_pct": "Missing (%)"},
        ),
        "Missing values · top 20",
    )


def correlation_chart(profile: Profile) -> go.Figure:
    return style(
        px.imshow(
            profile.correlation,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            labels={"color": "Pearson r"},
            aspect="auto",
        ),
        "Numeric correlations",
    )


def distribution_chart(frame: pd.DataFrame, column: str, kind: str) -> go.Figure:
    series = frame[column]
    if kind == "numeric":
        sample = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(sample) > MAX_CORRELATION_ROWS:
            sample = sample.sample(MAX_CORRELATION_ROWS, random_state=SEED)
        fig = px.histogram(x=sample, nbins=35, color_discrete_sequence=[TEAL], labels={"x": column})
    else:
        labels = series.map(lambda value: str(value) if pd.notna(value) else "(missing)")
        counts = labels.value_counts().head(20)
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color_discrete_sequence=[TEAL],
            labels={"x": column, "y": "Count"},
        )
    return style(fig, f"Distribution · {column}")


def outlier_chart(profile: Profile) -> go.Figure:
    data = (
        profile.column_stats[profile.column_stats.outliers > 0]
        .nlargest(20, "outlier_pct")
        .reset_index()
    )
    return style(
        px.bar(
            data,
            x="column",
            y="outlier_pct",
            color_discrete_sequence=["#f59e0b"],
            labels={"outlier_pct": "Finite values outside IQR fences (%)"},
        ),
        "IQR outlier candidates",
    )


def importance_chart(importance: pd.DataFrame) -> go.Figure:
    data = importance.head(15).sort_values("mean")
    return style(
        px.bar(
            data,
            x="mean",
            y="feature",
            error_x="std",
            orientation="h",
            color_discrete_sequence=[TEAL],
            labels={"mean": "Holdout score decrease"},
        ),
        "Permutation importance · top 15",
    )
