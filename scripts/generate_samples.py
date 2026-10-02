"""Regenerate MIT-licensed synthetic CSV samples deterministically."""

from pathlib import Path

from dataguard.demo import demo_dataset

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "sample_data"
    destination.mkdir(exist_ok=True)
    for task in ["classification", "regression"]:
        demo_dataset(task).to_csv(destination / f"synthetic_{task}.csv", index=False)
    print("Generated both synthetic demo CSVs.")
