#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.evaluation.plots import plot_roc_pr, plot_score_distribution, plot_score_timeline


def _threshold(run_dir: Path) -> float | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics.get("threshold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AM01 result plots from saved score CSV files.")
    parser.add_argument("--run-dir", required=True, help="Directory containing scores_val.csv and scores_test.csv.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to <run-dir>/figures.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.output) if args.output else run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    threshold = _threshold(run_dir)

    for split in ["val", "test"]:
        scores_path = run_dir / f"scores_{split}.csv"
        if not scores_path.exists():
            continue
        prefix = split.capitalize()
        plot_score_distribution(scores_path, out / f"{split}_score_distribution.png", title=f"{prefix} score distribution")
        plot_score_timeline(scores_path, out / f"{split}_score_timeline.png", threshold=threshold)
        try:
            plot_roc_pr(
                scores_path,
                out / f"{split}_roc_curve.png",
                out / f"{split}_precision_recall_curve.png",
                title_prefix=prefix,
            )
        except ValueError as exc:
            print(f"Skipping supervised curve for {split}: {exc}")

    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
