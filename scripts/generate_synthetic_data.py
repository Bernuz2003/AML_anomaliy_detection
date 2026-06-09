#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.data.synthetic import save_synthetic_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Kuka-like dataset for pipeline testing.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--runs", type=int, default=18)
    parser.add_argument("--length", type=int, default=220)
    parser.add_argument("--joints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anomaly-run-fraction", type=float, default=0.35)
    args = parser.parse_args()
    path = save_synthetic_csv(
        args.output,
        runs=args.runs,
        length=args.length,
        joints=args.joints,
        seed=args.seed,
        anomaly_run_fraction=args.anomaly_run_fraction,
    )
    print(f"Synthetic dataset written to {path}")


if __name__ == "__main__":
    main()
