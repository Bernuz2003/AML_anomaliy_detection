#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.pipeline import prepare_data, save_json
from am01.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data loading, split, scaling and windowing checks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/data_audit")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepared = prepare_data(args.data, cfg, output_dir=args.output)
    save_json(prepared.split_summary, Path(args.output) / "split_summary.json")
    print("Data preparation completed.")
    print(prepared.split_summary)


if __name__ == "__main__":
    main()
