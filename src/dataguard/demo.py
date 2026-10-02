"""Deterministic, wholly synthetic demos released under the project's MIT license."""

import numpy as np
import pandas as pd

from dataguard.config import SEED


def demo_dataset(task: str = "classification", *, issues: bool = True) -> pd.DataFrame:
    """Generate fictional subscription accounts; never represents actual customers."""
    rng = np.random.default_rng(SEED)
    n = 600
    tenure = rng.integers(1, 73, n)
    tickets = rng.poisson(1.6, n)
    monthly = rng.normal(64, 17, n).clip(15, 160)
    plan = rng.choice(["Starter", "Pro", "Business"], n, p=[0.45, 0.4, 0.15])
    risk = -1.4 - 0.025 * tenure + 0.45 * tickets + 0.013 * (monthly - 64)
    churn = rng.binomial(1, 1 / (1 + np.exp(-risk)))
    frame = pd.DataFrame(
        {
            "account_id": [f"ACC-{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charge": monthly.round(2),
            "support_tickets": tickets,
            "plan": plan,
            "autopay": rng.choice([True, False], n),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "joined_date": pd.date_range("2022-01-01", periods=n).strftime("%Y-%m-%d"),
        }
    )
    target = "churn" if task == "classification" else "annual_value"
    frame[target] = (
        churn
        if task == "classification"
        else (monthly * 12 + tenure * 2 - tickets * 30 + rng.normal(0, 90, n)).round(2)
    )
    if issues:
        frame.loc[rng.choice(n, 48, replace=False), "monthly_charge"] = np.nan
        frame.loc[rng.choice(n, 25, replace=False), "plan"] = None
        frame.loc[[10, 20, 30], "monthly_charge"] = [850, 1100, 950]
        frame["currency"] = "USD"
        frame["legacy_flag"] = ["current"] * (n - 3) + ["legacy"] * 3
        frame["target_copy_review"] = frame[target]
        frame = pd.concat([frame, frame.iloc[:12]], ignore_index=True)
    return frame
