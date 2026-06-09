from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_dataset_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    run_col: str = "run_id",
    label_col: str | None = "label",
    normal_label: int | float | str = 0,
    include_feature_stats: bool = True,
) -> pd.DataFrame:
    """Build a per-run audit table for the time-series dataset."""
    rows: list[dict[str, object]] = []
    has_labels = label_col is not None and label_col in df.columns
    for run_id, part in df.groupby(run_col, sort=False):
        row: dict[str, object] = {
            "run_id": run_id,
            "n_rows": int(len(part)),
            "n_features": int(len(feature_cols)),
            "missing_feature_values": int(part[feature_cols].isna().sum().sum()),
            "missing_feature_pct": float(part[feature_cols].isna().mean().mean()),
        }
        if has_labels:
            anomalous = (part[label_col].to_numpy() != normal_label)
            row["n_anomalous_rows"] = int(anomalous.sum())
            row["anomaly_fraction"] = float(anomalous.mean())
            row["run_label"] = "anomalous" if anomalous.any() else "normal"
        else:
            row["n_anomalous_rows"] = None
            row["anomaly_fraction"] = None
            row["run_label"] = "unlabeled"

        if "source_file" in part.columns:
            row["source_file"] = str(part["source_file"].iloc[0])
        if "action" in part.columns:
            row["action"] = part["action"].iloc[0]

        if include_feature_stats:
            means = part[feature_cols].mean(numeric_only=True)
            stds = part[feature_cols].std(numeric_only=True)
            for feature in feature_cols:
                row[f"mean__{feature}"] = float(means[feature])
                row[f"std__{feature}"] = float(stds[feature])
        rows.append(row)
    return pd.DataFrame(rows)


def save_dataset_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: str | Path,
    *,
    run_col: str = "run_id",
    label_col: str | None = "label",
    normal_label: int | float | str = 0,
    include_feature_stats: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_dataset_summary(
        df,
        feature_cols,
        run_col=run_col,
        label_col=label_col,
        normal_label=normal_label,
        include_feature_stats=include_feature_stats,
    )
    summary.to_csv(output_path, index=False)
