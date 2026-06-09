#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.data.io import infer_feature_columns, load_timeseries_data
from am01.utils.config import load_config


def _pick_run(df, run_col: str, label_col: str, normal_label):
    run_labels = df.groupby(run_col)[label_col].apply(lambda values: int((values.to_numpy() != normal_label).any()))
    normal_runs = run_labels[run_labels == 0].index
    anomalous_runs = run_labels[run_labels == 1].index
    normal = normal_runs[0] if len(normal_runs) else None
    anomalous = anomalous_runs[0] if len(anomalous_runs) else None
    return normal, anomalous


def _plot_run(
    part,
    *,
    time_col: str | None,
    label_col: str,
    normal_label,
    features: list[str],
    title: str,
    output_path: Path,
) -> None:
    x = part[time_col] if time_col and time_col in part.columns else range(len(part))
    fig, axes = plt.subplots(len(features), 1, figsize=(10, max(3, 1.8 * len(features))), sharex=True)
    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        ax.plot(x, part[feature], linewidth=1.1)
        if label_col in part.columns and (part[label_col] != normal_label).any():
            ymin, ymax = ax.get_ylim()
            ax.fill_between(
                x,
                ymin,
                ymax,
                where=(part[label_col].to_numpy() != normal_label),
                color="tab:orange",
                alpha=0.18,
                step="mid",
            )
        ax.set_ylabel(feature)
    axes[0].set_title(title)
    axes[-1].set_xlabel(time_col or "sample")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot normal/anomalous signal examples for AM01 data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/figures/data_examples")
    parser.add_argument("--features", nargs="*", default=None, help="Feature names to plot. Defaults to first six inferred features.")
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
    features = args.features or feature_cols[:6]
    missing = [feature for feature in features if feature not in feature_cols]
    if missing:
        raise ValueError(f"Unknown feature(s): {missing}")

    normal_run, anomalous_run = _pick_run(df, run_col, label_col, normal_label)
    output = Path(args.output)
    if normal_run is not None:
        _plot_run(
            df[df[run_col] == normal_run],
            time_col=time_col,
            label_col=label_col,
            normal_label=normal_label,
            features=features,
            title=f"Normal run: {normal_run}",
            output_path=output / "normal_run_signals.png",
        )
    if anomalous_run is not None:
        _plot_run(
            df[df[run_col] == anomalous_run],
            time_col=time_col,
            label_col=label_col,
            normal_label=normal_label,
            features=features,
            title=f"Anomalous run: {anomalous_run}",
            output_path=output / "anomalous_run_signals.png",
        )
    print(f"Signal figures written to {output}")


if __name__ == "__main__":
    main()
