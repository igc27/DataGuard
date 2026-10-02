"""Measured quality findings and an additive, auditable health score."""

from dataclasses import dataclass

import pandas as pd

from dataguard.profiling import Profile


@dataclass
class Score:
    value: float
    breakdown: pd.DataFrame


def quality_findings(profile: Profile) -> list[dict[str, str]]:
    """Return specific evidence and an action for every triggered finding."""
    findings = []
    if profile.duplicate_rows:
        findings.append({"severity": "warning", "column": "Dataset", "issue": "Duplicate rows",
                         "evidence": f"{profile.duplicate_rows:,} repeated records",
                         "action": "Review repeated records; prevent train/test overlap."})
    rules = [
        ("missing", "Missing values", "Impute from training data or investigate collection gaps."),
        ("invalid", "Infinite numeric values", "Investigate and replace infinities before modeling."),
        ("constant", "Constant or empty feature", "Exclude features without usable variation."),
        ("near_constant", "Near-constant feature", "Review usefulness; rare values may still matter."),
        ("id_like", "Possible identifier (heuristic)", "Review and exclude identifiers from baselines."),
        ("high_cardinality", "High-cardinality category", "Review IDs/text; consider domain-aware encoding."),
        ("mixed_numeric", "Possible mixed numeric types (heuristic)", "Check non-numeric tokens before conversion."),
        ("outliers", "IQR outlier candidates", "Inspect unusual observations; no rows are deleted."),
    ]
    for name, row in profile.column_stats.iterrows():
        for field, issue, action in rules:
            if row[field]:
                evidence = str(row[field])
                if field == "missing":
                    evidence = f"{row['missing']} cells ({row['missing_pct']:.1f}%)"
                elif field == "outliers":
                    evidence = f"{row['outliers']} finite values ({row['outlier_pct']:.1f}%)"
                elif field in {"high_cardinality", "id_like"}:
                    evidence = f"{row['unique']} unique values; ratio {row['cardinality']:.1%}"
                findings.append({"severity": "info" if field == "outliers" else "warning",
                                 "column": str(name), "issue": issue, "evidence": evidence, "action": action})
        if row.zero_pct >= 80:
            findings.append({"severity": "info", "column": str(name), "issue": "Zero-heavy feature",
                             "evidence": f"{row.zero_pct:.1f}% of finite values are zero",
                             "action": "Check whether zeros are structural or missing-value placeholders."})
    for a, b in profile.duplicate_columns:
        findings.append({"severity": "warning", "column": b, "issue": "Duplicate column",
                         "evidence": f"Exactly equal to {a}", "action": "Review redundant features."})
    for pair in profile.correlated_pairs:
        findings.append({"severity": "info", "column": pair['feature_a'],
                         "issue": "Strong numerical correlation", "evidence": f"{pair['feature_b']}: r={pair['correlation']:.3f}",
                         "action": "Review collinearity; correlation alone does not establish leakage."})
    return findings


def health_score(profile: Profile) -> Score:
    """100 minus weighted rates; each factor stays within its stated point budget."""
    table = profile.column_stats
    cells = profile.rows * profile.columns
    factors = [
        ("Missing cells", 35, table.missing.sum() / cells),
        ("Duplicate rows", 20, profile.duplicate_rows / profile.rows),
        ("Infinite cells", 15, table.invalid.sum() / cells),
        ("Constant / empty columns", 10, table.constant.mean()),
        ("Near-constant columns", 5, table.near_constant.mean()),
        ("Possible identifiers", 5, table.id_like.mean()),
        ("High-cardinality categories", 5, table.high_cardinality.mean()),
        ("Possible mixed numeric types", 5, table.mixed_numeric.mean()),
    ]
    breakdown = pd.DataFrame([{"factor": name, "max_penalty": budget,
                              "rate": float(rate), "penalty": float(budget * rate)}
                             for name, budget, rate in factors])
    return Score(round(max(0, 100 - breakdown.penalty.sum()), 1), breakdown)
