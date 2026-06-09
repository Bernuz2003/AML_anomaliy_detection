#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.data.audit import save_dataset_summary
from am01.data.io import infer_feature_columns, load_timeseries_data, save_dataframe_summary, validate_no_missing_feature_columns
from am01.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw AM01/Kuka time-series data.")
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument("--data", required=True, help="CSV data or KukaVelocityDataset directory.")
    parser.add_argument("--output", default="results/data_audit", help="Output directory.")
    parser.add_argument(
        "--no-feature-stats",
        action="store_true",
        help="Skip per-run per-feature mean/std columns in dataset_summary.csv.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    run_col = data_cfg.get("run_col", "run_id")
    time_col = data_cfg.get("time_col", "t")
    label_col = data_cfg.get("label_col", "label")
    normal_label = data_cfg.get("normal_label", 0)

    df = load_timeseries_data(
        args.data,
        run_col=run_col,
        time_col=time_col,
        label_col=label_col,
        data_format=data_cfg.get("format", "auto"),
        anomaly_col=data_cfg.get("anomaly_col", "anomaly"),
        action_col=data_cfg.get("action_col", "action"),
        kuka_run_strategy=data_cfg.get("kuka_run_strategy", "action_segments"),
        kuka_fixed_run_length=int(data_cfg.get("kuka_fixed_run_length", 512)),
    )
    feature_cols = infer_feature_columns(
        df,
        feature_cols=data_cfg.get("feature_cols"),
        run_col=run_col,
        time_col=time_col,
        label_col=label_col,
        split_col=data_cfg.get("split_col"),
    )
    validate_no_missing_feature_columns(df, feature_cols)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_dataset_summary(
        df,
        feature_cols,
        output / "dataset_summary.csv",
        run_col=run_col,
        label_col=label_col,
        normal_label=normal_label,
        include_feature_stats=not args.no_feature_stats,
    )
    save_dataframe_summary(df, feature_cols, output / "feature_summary.csv")
    print("Data audit completed.")
    print(f"Rows: {len(df)}")
    print(f"Runs: {df[run_col].nunique()}")
    print(f"Features: {len(feature_cols)}")
    if label_col in df.columns:
        print(f"Anomalous rows: {int((df[label_col] != normal_label).sum())}")
    print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()
