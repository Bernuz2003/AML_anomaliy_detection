#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.pipeline import run_experiment, save_json
from am01.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate an AM01 anomaly-detection experiment.")
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument("--data", required=True, help="CSV file or directory with raw time-series data.")
    parser.add_argument("--output", required=True, help="Output directory for results.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_experiment(cfg, data_path=args.data, output_dir=args.output)
    save_json(result, Path(args.output) / "metrics.json")
    print("Experiment completed.")
    print(f"Model: {result['model_type']}")
    print(f"Threshold: {result['threshold']:.6f}")
    print("Test metrics:")
    for key, value in result["test_metrics"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
