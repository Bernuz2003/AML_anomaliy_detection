#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.evaluation.metrics import all_metrics, select_threshold
from am01.pipeline import save_json


def _load_scores(path: Path):
    frame = pd.read_csv(path)
    required = {"run_id", "start", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    labels = frame["label"].to_numpy() if "label" in frame.columns else None
    return (
        frame["score"].to_numpy(),
        labels,
        frame["run_id"].astype(str).to_numpy(),
        frame["start"].to_numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved AM01 anomaly scores.")
    parser.add_argument("--run-dir", required=True, help="Directory containing scores_val.csv and scores_test.csv.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional fixed threshold.")
    parser.add_argument("--method", default="best_f1", choices=["best_f1", "percentile", "normal_percentile"])
    parser.add_argument("--fallback-percentile", type=float, default=99.0)
    parser.add_argument("--output", default=None, help="Metrics JSON path. Defaults to <run-dir>/metrics.json.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    val_scores, val_labels, val_run_ids, val_starts = _load_scores(run_dir / "scores_val.csv")
    test_scores, test_labels, test_run_ids, test_starts = _load_scores(run_dir / "scores_test.csv")

    threshold = args.threshold
    if threshold is None:
        threshold = select_threshold(
            val_labels,
            val_scores,
            method=args.method,
            fallback_percentile=args.fallback_percentile,
        )

    result = {
        "threshold": float(threshold),
        "validation_metrics": all_metrics(val_labels, val_scores, threshold, val_run_ids, val_starts),
        "test_metrics": all_metrics(test_labels, test_scores, threshold, test_run_ids, test_starts),
    }
    output_path = Path(args.output) if args.output else run_dir / "metrics.json"
    save_json(result, output_path)
    print("Evaluation completed.")
    print(f"Threshold: {threshold:.6f}")
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
