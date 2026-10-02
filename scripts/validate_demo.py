"""Exercise ingestion, profiling, scoring, models, and export outside Streamlit."""

import argparse
from pathlib import Path

from dataguard.demo import demo_dataset
from dataguard.ingestion import read_csv
from dataguard.modeling import train_baselines
from dataguard.profiling import profile_dataset
from dataguard.quality import health_score
from dataguard.readiness import assess_readiness
from dataguard.reporting import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for task, target in [("classification", "churn"), ("regression", "annual_value")]:
        # CSV round trip verifies ingestion rather than bypassing it.
        frame = read_csv(demo_dataset(task).to_csv(index=False).encode("utf-8"))
        profile = profile_dataset(frame)
        health = health_score(profile)
        readiness = assess_readiness(frame, profile, target)
        run = train_baselines(frame, target)
        report = build_report(
            profile, health, dataset_name=f"Synthetic {task}", readiness=readiness, model_run=run
        )
        (args.output / f"demo-{task}.html").write_text(report, encoding="utf-8")
        print(
            f"{task}: {profile.rows} rows; health={health.value}; readiness={readiness.score.value}; {len(run.results)} baselines; top={run.winner}"
        )
        print(run.results.to_string(index=False))


if __name__ == "__main__":
    main()
