from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from am01.data.io import infer_feature_columns, load_timeseries_data
from am01.data.preprocessing import clean_missing_values
from am01.data.windowing import make_windows, split_by_run
from am01.evaluation.metrics import all_metrics, select_threshold
from am01.models.aae import LatentDiscriminator
from am01.pipeline import build_autoencoder
from am01.training.losses import per_window_reconstruction_error


MODEL_LABELS = {
    "pca": "PCA",
    "isolation_forest": "Isolation Forest",
    "ae_mlp": "AE MLP",
    "aae_mlp": "AAE MLP",
    "ae_conv1d": "AE Conv1D",
}


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_markdown(path: str | Path, lines: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def savefig(path: str | Path, *, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def model_key_from_run(run_dir: Path, cfg: dict[str, Any] | None = None) -> str:
    if cfg is not None:
        model_type = cfg.get("model", {}).get("type")
        if model_type:
            return str(model_type)
    name = run_dir.name
    for key in sorted(MODEL_LABELS, key=len, reverse=True):
        if key in name:
            return key
    return name


def load_project_dataframe(config: dict[str, Any], data_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    data_cfg = config.get("data", {})
    run_col = data_cfg.get("run_col", "run_id")
    time_col = data_cfg.get("time_col", "t")
    label_col = data_cfg.get("label_col", "label")
    df = load_timeseries_data(
        data_path,
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
    return df, feature_cols


def dataset_report_artifacts(
    config: dict[str, Any],
    data_path: str | Path,
    *,
    tables_dir: str | Path,
    figures_dir: str | Path,
    window_lengths: list[int] | tuple[int, ...] = (32, 64),
    primary_window_length: int = 64,
) -> dict[str, pd.DataFrame]:
    """Create report-ready dataset and windowing tables/figures."""
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    prep_cfg = config.get("preprocessing", {})
    win_cfg = config.get("windowing", {})
    run_col = data_cfg.get("run_col", "run_id")
    time_col = data_cfg.get("time_col", "t")
    label_col = data_cfg.get("label_col", "label")
    normal_label = data_cfg.get("normal_label", 0)

    df, feature_cols = load_project_dataframe(config, data_path)
    df = clean_missing_values(
        df,
        feature_cols,
        run_col=run_col,
        strategy=prep_cfg.get("missing_strategy", "interpolate"),
    )

    source_col = "source_file" if "source_file" in df.columns else label_col
    composition = (
        df.assign(class_label=np.where(df[label_col].to_numpy() == normal_label, "normal", "slow/anomalous"))
        .groupby([source_col, "class_label"], dropna=False)
        .agg(rows=(run_col, "size"), segments=(run_col, "nunique"))
        .reset_index()
        .rename(columns={source_col: "source"})
    )
    composition.to_csv(tables_dir / "dataset_composition.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=composition, x="source", y="rows", hue="class_label", ax=axes[0])
    axes[0].set_title("Dataset rows")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=composition, x="source", y="segments", hue="class_label", ax=axes[1])
    axes[1].set_title("Dataset segments")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    savefig(figures_dir / "fig01_dataset_composition.png")
    plt.close(fig)

    segment_lengths = (
        df.groupby(run_col, sort=False)
        .agg(
            length=(run_col, "size"),
            label=(label_col, lambda values: int((values.to_numpy() != normal_label).any())),
            action=("action", "first") if "action" in df.columns else (run_col, "first"),
            source_file=("source_file", "first") if "source_file" in df.columns else (run_col, "first"),
        )
        .reset_index()
    )
    segment_lengths.to_csv(tables_dir / "segment_lengths.csv", index=False)

    plt.figure(figsize=(9, 4.5))
    ax = sns.histplot(data=segment_lengths, x="length", hue="label", bins=40, multiple="stack")
    for length in window_lengths:
        ax.axvline(length, linestyle="--", linewidth=1.4, label=f"w={length}")
    ax.set_title("Segment length distribution")
    ax.set_xlabel("segment length")
    ax.legend()
    savefig(figures_dir / "fig02_segment_length_distribution.png")
    plt.close()

    split_frames = split_by_run(
        df,
        run_col=run_col,
        label_col=label_col,
        normal_label=normal_label,
        split_col=data_cfg.get("split_col"),
        train_ratio=float(data_cfg.get("train_ratio", 0.6)),
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        test_ratio=float(data_cfg.get("test_ratio", 0.2)),
        stratify_by_label=bool(data_cfg.get("stratify_by_label", True)),
        seed=int(config.get("seed", 42)),
    )
    window_rows = []
    for length in window_lengths:
        split_windows = {}
        for split_name, split_df in zip(["train", "val", "test"], split_frames):
            windows = make_windows(
                split_df,
                feature_cols=feature_cols,
                run_col=run_col,
                label_col=label_col,
                normal_label=normal_label,
                window_length=int(length),
                stride=int(win_cfg.get("stride", 16)),
                anomaly_fraction=float(win_cfg.get("anomaly_fraction", 0.10)),
            )
            split_windows[split_name] = windows
        test_w = split_windows["test"]
        window_rows.append(
            {
                "window_length": length,
                "train_windows": split_windows["train"].n_windows,
                "val_windows": split_windows["val"].n_windows,
                "test_windows": test_w.n_windows,
                "test_anomalous_windows": int(test_w.y.sum()) if test_w.y is not None else np.nan,
                "test_anomaly_prevalence": float(test_w.y.mean()) if test_w.y is not None else np.nan,
                "contributing_test_runs": int(len(np.unique(test_w.run_ids))),
            }
        )
    window_counts = pd.DataFrame(window_rows)
    window_counts.to_csv(tables_dir / "window_count_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=window_counts, x="window_length", y="test_windows", ax=axes[0], color="tab:blue")
    axes[0].set_title("Test windows by window length")
    axes[0].set_xlabel("window length")
    sns.barplot(data=window_counts, x="window_length", y="test_anomaly_prevalence", ax=axes[1], color="tab:orange")
    axes[1].set_title("Test anomaly prevalence")
    axes[1].set_xlabel("window length")
    savefig(figures_dir / "fig03_window_count_sensitivity.png")
    plt.close(fig)

    selected_features = _select_interesting_features(df, feature_cols, max_features=6)
    _plot_example_traces(
        df,
        run_col=run_col,
        time_col=time_col,
        label_col=label_col,
        normal_label=normal_label,
        features=selected_features[:4],
        output_path=figures_dir / "fig04_example_sensor_traces_normal_vs_slow.png",
    )
    _plot_feature_distributions(
        df,
        feature_cols=selected_features,
        label_col=label_col,
        normal_label=normal_label,
        output_path=figures_dir / "fig05_feature_distributions.png",
    )
    _plot_windowing_scheme(
        window_length=primary_window_length,
        stride=int(win_cfg.get("stride", 16)),
        output_path=figures_dir / "fig06_windowing_scheme.png",
    )

    return {
        "dataset_composition": composition,
        "segment_lengths": segment_lengths,
        "window_count_sensitivity": window_counts,
    }


def _select_interesting_features(df: pd.DataFrame, feature_cols: list[str], *, max_features: int = 6) -> list[str]:
    preferred_tokens = ("power", "gyro", "vel", "velocity", "q", "joint")
    preferred = [
        col for col in feature_cols
        if any(token in col.lower() for token in preferred_tokens)
    ]
    ranked = df[feature_cols].std(numeric_only=True).sort_values(ascending=False).index.tolist()
    out: list[str] = []
    for col in preferred + ranked:
        if col not in out:
            out.append(col)
        if len(out) >= max_features:
            break
    return out


def _plot_example_traces(
    df: pd.DataFrame,
    *,
    run_col: str,
    time_col: str | None,
    label_col: str,
    normal_label: Any,
    features: list[str],
    output_path: Path,
) -> None:
    run_labels = df.groupby(run_col)[label_col].apply(lambda values: int((values.to_numpy() != normal_label).any()))
    normal_runs = run_labels[run_labels == 0].index
    anomalous_runs = run_labels[run_labels == 1].index
    chosen = []
    if len(normal_runs):
        chosen.append(("normal", normal_runs[0]))
    if len(anomalous_runs):
        chosen.append(("slow/anomalous", anomalous_runs[0]))
    fig, axes = plt.subplots(len(chosen), 1, figsize=(11, max(3, 3 * len(chosen))), sharex=False)
    if len(chosen) == 1:
        axes = [axes]
    for ax, (label, run_id) in zip(axes, chosen):
        part = df[df[run_col] == run_id]
        x = part[time_col] if time_col and time_col in part.columns else np.arange(len(part))
        for feature in features:
            values = part[feature].to_numpy(dtype=float)
            denom = np.nanstd(values) or 1.0
            ax.plot(x, (values - np.nanmean(values)) / denom, linewidth=1.0, label=feature)
        if (part[label_col] != normal_label).any():
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
        ax.set_title(f"{label} segment: {run_id}")
        ax.set_ylabel("z-normalized value")
        ax.legend(loc="upper right", ncols=2, fontsize=8)
    axes[-1].set_xlabel(time_col or "sample")
    savefig(output_path)
    plt.close(fig)


def _plot_feature_distributions(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str,
    normal_label: Any,
    output_path: Path,
) -> None:
    sample = df[[label_col, *feature_cols]].copy()
    if len(sample) > 15000:
        sample = sample.sample(15000, random_state=0)
    sample["class"] = np.where(sample[label_col].to_numpy() == normal_label, "normal", "slow/anomalous")
    long = sample.melt(id_vars=["class"], value_vars=feature_cols, var_name="feature", value_name="value")
    long["value_z"] = long.groupby("feature")["value"].transform(
        lambda values: (values - values.mean()) / (values.std() if values.std() else 1.0)
    )
    plt.figure(figsize=(11, 5))
    ax = sns.boxplot(data=long, x="feature", y="value_z", hue="class", showfliers=False)
    ax.set_title("Feature distributions by class")
    ax.set_xlabel("")
    ax.set_ylabel("z-normalized value")
    ax.tick_params(axis="x", rotation=30)
    savefig(output_path)
    plt.close()


def _plot_windowing_scheme(*, window_length: int, stride: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    total = window_length + 2 * stride + 10
    ax.hlines(0, 0, total, color="black", linewidth=2)
    colors = ["tab:blue", "tab:green", "tab:purple"]
    for i, start in enumerate([0, stride, 2 * stride]):
        ax.broken_barh([(start, window_length)], (-0.25 + i * 0.25, 0.18), facecolors=colors[i], alpha=0.8)
        ax.text(start + window_length / 2, 0.05 + i * 0.25, f"window {i + 1}", ha="center", fontsize=9)
        ax.vlines([start, start + window_length - 1], -0.45, 0.8, colors=colors[i], linestyles="--", linewidth=0.8)
    ax.set_title(f"Sliding-window scheme: length={window_length}, stride={stride}")
    ax.set_xlabel("timestep")
    ax.set_yticks([])
    ax.set_xlim(-2, total + 2)
    savefig(output_path)
    plt.close(fig)


def preprocessing_summary_table(run_dir: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    summary = read_json(Path(run_dir) / "split_summary.json")
    rows = []
    for split in ["train", "val", "test"]:
        windows = int(summary.get("windows", {}).get(split, 0))
        anomalous = int(summary.get("anomalous_windows", {}).get(split, 0))
        rows.append(
            {
                "split": split,
                "rows": summary.get("rows", {}).get(split),
                "runs": summary.get("runs", {}).get(split),
                "windows": windows,
                "anomalous_windows": anomalous,
                "anomaly_percent": anomalous / max(windows, 1),
            }
        )
    frame = pd.DataFrame(rows)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return frame


def collect_run_metrics(runs_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(runs_root).rglob("metrics.json")):
        run_dir = metrics_path.parent
        cfg_path = run_dir / "config_used.json"
        cfg = read_json(cfg_path) if cfg_path.exists() else {}
        metrics = read_json(metrics_path)
        model_key = model_key_from_run(run_dir, cfg)
        row: dict[str, Any] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model_key": model_key,
            "model": MODEL_LABELS.get(model_key, model_key),
            "seed": cfg.get("seed"),
            "window_length": cfg.get("windowing", {}).get("window_length"),
            "stride": cfg.get("windowing", {}).get("stride"),
            "latent_dim": cfg.get("model", {}).get("latent_dim"),
            "lambda_adv": cfg.get("model", {}).get("lambda_adv"),
            "warmup_epochs": cfg.get("training", {}).get("warmup_epochs", cfg.get("model", {}).get("warmup_epochs")),
            "ramp_epochs": cfg.get("training", {}).get("ramp_epochs", cfg.get("model", {}).get("ramp_epochs")),
            "scaler": cfg.get("preprocessing", {}).get("scaler"),
            "loss": cfg.get("training", {}).get("loss"),
            "threshold": metrics.get("threshold"),
        }
        for split_key, prefix in [("validation_metrics", "val"), ("test_metrics", "test")]:
            for metric_name, value in metrics.get(split_key, {}).items():
                row[f"{prefix}_{metric_name}"] = value
        split_summary = metrics.get("split_summary", {})
        row["test_windows"] = split_summary.get("windows", {}).get("test")
        row["test_anomalous_windows"] = split_summary.get("anomalous_windows", {}).get("test")
        if row["test_windows"]:
            row["test_anomaly_prevalence"] = row["test_anomalous_windows"] / row["test_windows"]
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {runs_root}")
    return pd.DataFrame(rows)


def save_main_results(runs_root: str | Path, tables_dir: str | Path) -> pd.DataFrame:
    metrics = collect_run_metrics(runs_root)
    preferred = [
        "model",
        "run_name",
        "test_f1",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_event_recall",
        "test_false_alarms_per_run",
        "val_pr_auc",
        "val_f1",
        "test_windows",
        "test_anomaly_prevalence",
    ]
    out = metrics[[col for col in preferred if col in metrics.columns]].sort_values("test_pr_auc", ascending=False)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tables_dir / "main_results.csv", index=False)
    metrics.to_csv(tables_dir / "main_results_full.csv", index=False)
    return out


def plot_main_result_figures(runs_root: str | Path, figures_dir: str | Path) -> None:
    runs = collect_run_metrics(runs_root)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_supervised_curves(runs, figures_dir / "fig07_main_pr_curves.png", kind="pr")
    _plot_supervised_curves(runs, figures_dir / "fig08_main_roc_curves.png", kind="roc")
    _plot_score_distributions(runs, figures_dir / "fig09_score_distributions.png")
    _plot_confusion_matrices(runs, figures_dir / "fig10_confusion_matrices.png")


def _load_scores(run_dir: str | Path, split: str = "test") -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / f"scores_{split}.csv")


def _plot_supervised_curves(runs: pd.DataFrame, output_path: Path, *, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in runs.sort_values("model").iterrows():
        scores_path = Path(row["run_dir"]) / "scores_test.csv"
        if not scores_path.exists():
            continue
        scores = pd.read_csv(scores_path)
        if "label" not in scores or len(scores["label"].unique()) < 2:
            continue
        y = scores["label"].to_numpy().astype(int)
        s = scores["score"].to_numpy()
        label = row["model"]
        if kind == "pr":
            PrecisionRecallDisplay.from_predictions(y, s, ax=ax, name=label)
        else:
            RocCurveDisplay.from_predictions(y, s, ax=ax, name=label)
    ax.set_title("Main models precision-recall" if kind == "pr" else "Main models ROC")
    savefig(output_path)
    plt.close(fig)


def _plot_score_distributions(runs: pd.DataFrame, output_path: Path) -> None:
    frames = []
    for _, row in runs.iterrows():
        path = Path(row["run_dir"]) / "scores_test.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "label" not in frame:
            continue
        frame = frame[["score", "label"]].copy()
        frame["model"] = row["model"]
        frame["class"] = np.where(frame["label"].to_numpy() == 1, "anomalous", "normal")
        frame["log_score"] = np.log1p(np.maximum(frame["score"].to_numpy(), 0))
        frames.append(frame)
    if not frames:
        return
    long = pd.concat(frames, ignore_index=True)
    g = sns.FacetGrid(long, col="model", hue="class", col_wrap=3, sharex=False, sharey=False, height=3.0)
    g.map_dataframe(sns.histplot, x="log_score", stat="density", bins=35, alpha=0.45)
    g.add_legend()
    g.fig.suptitle("Test score distributions", y=1.03)
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def _plot_confusion_matrices(runs: pd.DataFrame, output_path: Path) -> None:
    n = len(runs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (_, row) in zip(axes, runs.iterrows()):
        scores = _load_scores(row["run_dir"], "test")
        y = scores["label"].to_numpy().astype(int)
        pred = (scores["score"].to_numpy() >= float(row["threshold"])).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["normal", "anomaly"]).plot(ax=ax, colorbar=False)
        ax.set_title(row["model"])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Main model confusion matrices", y=1.02)
    savefig(output_path)
    plt.close(fig)


def save_ae_vs_aae_comparison(main_results: pd.DataFrame, tables_dir: str | Path, figures_dir: str | Path) -> pd.DataFrame:
    ae = main_results[main_results["model"] == "AE MLP"].iloc[0]
    aae = main_results[main_results["model"] == "AAE MLP"].iloc[0]
    metrics = ["test_f1", "test_pr_auc", "test_roc_auc", "test_false_alarms_per_run"]
    rows = []
    for metric in metrics:
        rows.append({"metric": metric, "AE MLP": ae[metric], "AAE MLP": aae[metric], "AAE_minus_AE": aae[metric] - ae[metric]})
    comparison = pd.DataFrame(rows)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    comparison.to_csv(Path(tables_dir) / "ae_vs_aae_direct_comparison.csv", index=False)

    plt.figure(figsize=(7, 4.2))
    ax = sns.barplot(data=comparison, x="metric", y="AAE_minus_AE", color="tab:blue")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("AAE minus AE under main protocol")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    savefig(Path(figures_dir) / "fig11_ae_vs_aae_metric_delta.png")
    plt.close()
    return comparison


def save_ablation_tables_and_figures(
    *,
    window_runs_root: str | Path,
    aae_runs_root: str | Path,
    preprocessing_runs_root: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    window_df = collect_run_metrics(window_runs_root)
    window_cols = [
        "window_length",
        "model",
        "test_windows",
        "test_anomaly_prevalence",
        "test_pr_auc",
        "test_f1",
        "test_roc_auc",
        "val_pr_auc",
    ]
    window_table = window_df[[col for col in window_cols if col in window_df.columns]].sort_values(["window_length", "model"])
    window_table.to_csv(tables_dir / "window_length_sensitivity.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    ax = sns.lineplot(data=window_table, x="window_length", y="test_pr_auc", hue="model", marker="o")
    ax.set_title("Window length vs test PR-AUC")
    ax.set_xticks(sorted(window_table["window_length"].dropna().unique()))
    savefig(figures_dir / "fig12_window_length_vs_performance.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=window_table.drop_duplicates("window_length"), x="window_length", y="test_windows", color="tab:blue")
    ax.set_title("Window length vs test windows")
    savefig(figures_dir / "fig13_window_length_vs_test_windows.png")
    plt.close()

    aae_df = collect_run_metrics(aae_runs_root)
    aae_df.to_csv(tables_dir / "aae_ablation_results.csv", index=False)
    heat = aae_df.pivot_table(index="lambda_adv", columns="latent_dim", values="val_pr_auc", aggfunc="max")
    plt.figure(figsize=(6, 4.5))
    ax = sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis")
    ax.set_title("AAE ablation: validation PR-AUC")
    savefig(figures_dir / "fig14_aae_lambda_latent_heatmap_val_pr_auc.png")
    plt.close()

    prep_df = collect_run_metrics(preprocessing_runs_root)
    prep_df.to_csv(tables_dir / "preprocessing_loss_ablation.csv", index=False)
    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=prep_df, x="scaler", y="val_pr_auc", hue="loss")
    ax.set_title("Preprocessing/loss ablation: validation PR-AUC")
    savefig(figures_dir / "fig15_preprocessing_loss_ablation.png")
    plt.close()
    return {"window_length": window_table, "aae_ablation": aae_df, "preprocessing_loss": prep_df}


def select_best_run(runs_root: str | Path, model_key: str, *, metric: str = "val_pr_auc") -> Path:
    runs = collect_run_metrics(runs_root)
    part = runs[runs["model_key"] == model_key].dropna(subset=[metric])
    if part.empty:
        raise FileNotFoundError(f"No run for model={model_key!r} with metric={metric!r} under {runs_root}")
    return Path(part.sort_values(metric, ascending=False).iloc[0]["run_dir"])


def aae_diagnostics_artifacts(
    *,
    ae_run_dir: str | Path,
    aae_run_dir: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    extended_dir: str | Path,
    selection_metric: str = "val_pr_auc",
    batch_size: int = 512,
) -> dict[str, Any]:
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    extended_dir = Path(extended_dir)
    for path in [tables_dir, figures_dir, extended_dir]:
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_cfg, ae_model, _ = _load_autoencoder_run(Path(ae_run_dir), device=device)
    aae_cfg, aae_model, aae_disc = _load_autoencoder_run(Path(aae_run_dir), device=device, require_discriminator=True)
    ae_loss = ae_cfg.get("training", {}).get("loss", "mse")
    aae_loss = aae_cfg.get("training", {}).get("loss", "mse")
    ae_val = _forward_split(Path(ae_run_dir), ae_model, None, "val", device=device, batch_size=batch_size, error_mode=ae_loss)
    ae_test = _forward_split(Path(ae_run_dir), ae_model, None, "test", device=device, batch_size=batch_size, error_mode=ae_loss)
    aae_train = _forward_split(Path(aae_run_dir), aae_model, aae_disc, "train", device=device, batch_size=batch_size, error_mode=aae_loss)
    aae_val = _forward_split(Path(aae_run_dir), aae_model, aae_disc, "val", device=device, batch_size=batch_size, error_mode=aae_loss)
    aae_test = _forward_split(Path(aae_run_dir), aae_model, aae_disc, "test", device=device, batch_size=batch_size, error_mode=aae_loss)

    mahal = LedoitWolf().fit(aae_train["z"][aae_train["y"] == 0] if aae_train["y"] is not None else aae_train["z"])
    val_scores, norm_params = _build_aae_score_frame(aae_val, "val", mahal)
    test_scores, _ = _build_aae_score_frame(aae_test, "test", mahal, norm_params)
    val_scores.to_csv(extended_dir / "aae_extended_scores_val.csv", index=False)
    test_scores.to_csv(extended_dir / "aae_extended_scores_test.csv", index=False)
    diagnostics_validation = _validate_reconstruction_score_consistency(
        ae_run_dir=Path(ae_run_dir),
        aae_run_dir=Path(aae_run_dir),
        ae_loss=ae_loss,
        aae_loss=aae_loss,
        aae_test_scores=test_scores,
        output_path=tables_dir / "aae_diagnostics_validation.json",
    )

    score_cols = [
        "score_rec",
        "score_latent_norm",
        "score_latent_mahalanobis",
        "score_disc",
        *[c for c in test_scores.columns if c.startswith("score_combined_")],
    ]
    score_rows = []
    for score_col in score_cols:
        val_y = val_scores["label"].to_numpy().astype(int)
        test_y = test_scores["label"].to_numpy().astype(int)
        val_s = val_scores[score_col].to_numpy()
        test_s = test_scores[score_col].to_numpy()
        threshold = select_threshold(val_y, val_s, method="best_f1")
        metrics = all_metrics(test_y, test_s, threshold, test_scores["run_id"].to_numpy(), test_scores["start"].to_numpy())
        score_rows.append(
            {
                "aae_score": score_col,
                "threshold": threshold,
                "val_pr_auc": average_precision_score(val_y, val_s) if len(np.unique(val_y)) == 2 else np.nan,
                "val_roc_auc": roc_auc_score(val_y, val_s) if len(np.unique(val_y)) == 2 else np.nan,
                "test_pr_auc": average_precision_score(test_y, test_s) if len(np.unique(test_y)) == 2 else np.nan,
                "test_roc_auc": roc_auc_score(test_y, test_s) if len(np.unique(test_y)) == 2 else np.nan,
                "test_f1": metrics.get("f1"),
                "test_balanced_accuracy": metrics.get("balanced_accuracy"),
                "test_false_alarms_per_run": metrics.get("false_alarms_per_run"),
            }
        )
    score_table = pd.DataFrame(score_rows).sort_values(["val_pr_auc", "test_pr_auc"], ascending=False)
    score_table.to_csv(tables_dir / "aae_specific_scores.csv", index=False)
    best_score = str(score_table.iloc[0]["aae_score"])
    best_threshold = float(score_table.iloc[0]["threshold"])

    _plot_aae_score_bars(score_table, figures_dir / "fig16_aae_specific_scores_pr_auc.png")
    latent_frame = _plot_latent_diagnostics(test_scores, best_score, best_threshold, figures_dir, tables_dir, Path(aae_run_dir))
    _plot_latent_score_distributions(test_scores, best_score, figures_dir / "fig19_latent_discriminator_score_distribution.png")
    per_feature = _feature_reconstruction_artifacts(ae_test, aae_test, tables_dir, figures_dir)
    per_action = _per_action_artifacts(ae_val, ae_test, val_scores, test_scores, best_score, best_threshold, Path(ae_run_dir), Path(aae_run_dir), tables_dir, figures_dir)

    return {
        "ae_config": ae_cfg,
        "aae_config": aae_cfg,
        "score_table": score_table,
        "best_score": best_score,
        "diagnostics_validation": diagnostics_validation,
        "latent_frame": latent_frame,
        "per_feature": per_feature,
        "per_action": per_action,
    }


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_npz(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(run_dir / f"processed_{split}.npz", allow_pickle=True)
    return (
        data["X"].astype(np.float32),
        data["y"].astype(int) if bool(data["has_labels"]) else None,
        data["run_ids"].astype(str),
        data["starts"].astype(int),
        [str(v) for v in data["feature_cols"]],
    )


def _load_autoencoder_run(run_dir: Path, *, device: torch.device, require_discriminator: bool = False):
    cfg = read_json(run_dir / "config_used.json")
    X, _, _, _, _ = _load_npz(run_dir, "test")
    checkpoint = _safe_torch_load(run_dir / "model.pt")
    model = build_autoencoder(cfg.get("model", {}), window_length=X.shape[1], n_channels=X.shape[2])
    model.load_state_dict(checkpoint["autoencoder_state_dict"])
    model.to(device).eval()
    discriminator = None
    if "discriminator_state_dict" in checkpoint:
        model_cfg = cfg.get("model", {})
        discriminator = LatentDiscriminator(
            latent_dim=int(model_cfg.get("latent_dim", 16)),
            hidden_dims=list(model_cfg.get("discriminator_hidden_dims", [128, 64])),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        discriminator.to(device).eval()
    elif require_discriminator:
        raise ValueError(f"Missing discriminator checkpoint in {run_dir}")
    return cfg, model, discriminator


@torch.inference_mode()
def _forward_split(
    run_dir: Path,
    model: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    split: str,
    *,
    device: torch.device,
    batch_size: int,
    error_mode: str = "mse",
) -> dict[str, Any]:
    X, y, run_ids, starts, feature_cols = _load_npz(run_dir, split)
    tensor = torch.from_numpy(X).float()
    rec_scores, feature_errors, z_batches, disc_batches = [], [], [], []
    for start in range(0, len(tensor), batch_size):
        x = tensor[start:start + batch_size].to(device)
        z = model.encode(x)
        x_hat = model.decode(z)
        err = _pointwise_reconstruction_error(x, x_hat, mode=error_mode)
        rec_scores.append(per_window_reconstruction_error(x, x_hat, mode=error_mode).cpu().numpy())
        feature_errors.append(err.mean(dim=1).cpu().numpy())
        z_batches.append(z.cpu().numpy())
        if discriminator is not None:
            disc_batches.append(torch.sigmoid(discriminator(z)).cpu().numpy())
    out = {
        "X_shape": X.shape,
        "y": y,
        "run_ids": run_ids,
        "starts": starts,
        "feature_cols": feature_cols,
        "score_rec": np.concatenate(rec_scores),
        "feature_error": np.concatenate(feature_errors, axis=0),
        "error_mode": error_mode,
        "z": np.concatenate(z_batches, axis=0),
    }
    if disc_batches:
        out["disc_prob_prior"] = np.concatenate(disc_batches)
    return out


def _pointwise_reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "mse":
        return (x - x_hat) ** 2
    if mode == "mae":
        return torch.abs(x - x_hat)
    if mode == "huber":
        return torch.nn.functional.smooth_l1_loss(x_hat, x, reduction="none")
    raise ValueError(f"Unknown reconstruction error mode: {mode}")


def _validate_reconstruction_score_consistency(
    *,
    ae_run_dir: Path,
    aae_run_dir: Path,
    ae_loss: str,
    aae_loss: str,
    aae_test_scores: pd.DataFrame,
    output_path: Path,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    stored_metrics = read_json(aae_run_dir / "metrics.json")
    stored_pr_auc = float(stored_metrics["test_metrics"]["pr_auc"])
    y = aae_test_scores["label"].to_numpy().astype(int)
    recomputed_pr_auc = float(average_precision_score(y, aae_test_scores["score_rec"].to_numpy()))
    delta = abs(stored_pr_auc - recomputed_pr_auc)
    check_passed = bool(delta <= tolerance)
    payload = {
        "ae_run": str(ae_run_dir),
        "aae_run": str(aae_run_dir),
        "ae_loss": ae_loss,
        "aae_loss": aae_loss,
        "stored_aae_test_pr_auc": stored_pr_auc,
        "recomputed_aae_test_pr_auc": recomputed_pr_auc,
        "absolute_delta": delta,
        "tolerance": tolerance,
        "check_passed": check_passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if not check_passed:
        raise ValueError(
            "AAE diagnostics reconstruction score is inconsistent with stored run metrics. "
            f"Stored test PR-AUC={stored_pr_auc:.6f}, recomputed={recomputed_pr_auc:.6f}, "
            f"delta={delta:.6g}. Validation written to {output_path}."
        )
    return payload


def _norm_params(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return float(lo), float(hi)


def _apply_norm(values: np.ndarray, params: tuple[float, float]) -> np.ndarray:
    lo, hi = params
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _build_aae_score_frame(
    diag: dict[str, Any],
    split: str,
    mahal: LedoitWolf,
    norm_params: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    eps = 1e-9
    frame = pd.DataFrame(
        {
            "run_id": diag["run_ids"],
            "start": diag["starts"],
            "label": diag["y"],
            "split": split,
            "score_rec": diag["score_rec"],
            "score_latent_norm": np.sum(diag["z"] ** 2, axis=1),
            "score_latent_mahalanobis": mahal.mahalanobis(diag["z"]),
            "score_disc": -np.log(np.clip(diag["disc_prob_prior"], eps, 1.0)),
        }
    )
    z_frame = pd.DataFrame(diag["z"], columns=[f"z_{i}" for i in range(diag["z"].shape[1])])
    frame = pd.concat([frame, z_frame], axis=1)
    base = ["score_rec", "score_latent_norm", "score_latent_mahalanobis", "score_disc"]
    if norm_params is None:
        norm_params = {col: _norm_params(frame[col].to_numpy()) for col in base}
    for col in base:
        frame[f"{col}_normed"] = _apply_norm(frame[col].to_numpy(), norm_params[col])
    for alpha in [0.25, 0.5, 0.75]:
        suffix = str(alpha).replace(".", "p")
        frame[f"score_combined_rec_disc_a{suffix}"] = alpha * frame["score_rec_normed"] + (1 - alpha) * frame["score_disc_normed"]
        frame[f"score_combined_rec_latent_a{suffix}"] = alpha * frame["score_rec_normed"] + (1 - alpha) * frame["score_latent_mahalanobis_normed"]
    return frame, norm_params


def _plot_aae_score_bars(score_table: pd.DataFrame, output_path: Path) -> None:
    plot_df = score_table.head(12).melt(
        id_vars="aae_score",
        value_vars=["val_pr_auc", "test_pr_auc"],
        var_name="split_metric",
        value_name="pr_auc",
    )
    plt.figure(figsize=(11, 4.8))
    ax = sns.barplot(data=plot_df, x="aae_score", y="pr_auc", hue="split_metric")
    ax.set_title("AAE-specific scores: PR-AUC")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    savefig(output_path)
    plt.close()


def _action_map(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "dataset_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if not {"run_id", "action"}.issubset(frame.columns):
        return {}
    frame["run_id"] = frame["run_id"].astype(str)
    return frame.drop_duplicates("run_id").set_index("run_id")["action"].to_dict()


def _plot_latent_diagnostics(
    test_scores: pd.DataFrame,
    best_score: str,
    threshold: float,
    figures_dir: Path,
    tables_dir: Path,
    aae_run_dir: Path,
) -> pd.DataFrame:
    z_cols = [col for col in test_scores.columns if col.startswith("z_")]
    z = test_scores[z_cols].to_numpy()
    coords = PCA(n_components=2, random_state=0).fit_transform(z) if z.shape[1] >= 2 else np.column_stack([z[:, 0], np.zeros(len(z))])
    frame = test_scores[["run_id", "start", "label", best_score]].copy()
    frame["pc1"] = coords[:, 0]
    frame["pc2"] = coords[:, 1]
    frame["class"] = np.where(frame["label"].to_numpy() == 1, "anomalous", "normal")
    pred = (frame[best_score].to_numpy() >= threshold).astype(int)
    y = frame["label"].to_numpy().astype(int)
    frame["outcome"] = np.select(
        [(y == 0) & (pred == 0), (y == 0) & (pred == 1), (y == 1) & (pred == 0), (y == 1) & (pred == 1)],
        ["TN", "FP", "FN", "TP"],
        default="unknown",
    )
    actions = _action_map(aae_run_dir)
    frame["action"] = frame["run_id"].map(actions)
    frame.to_csv(tables_dir / "aae_latent_pca_coordinates.csv", index=False)

    plt.figure(figsize=(6.5, 5.2))
    ax = sns.scatterplot(data=frame, x="pc1", y="pc2", hue="class", alpha=0.75, s=24)
    ax.set_title("AAE latent PCA by label")
    savefig(figures_dir / "fig17_aae_latent_pca_label.png")
    plt.close()

    if frame["action"].notna().any():
        top_actions = frame["action"].value_counts().head(8).index
        plot_df = frame[frame["action"].isin(top_actions)]
        plt.figure(figsize=(7.5, 5.5))
        ax = sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="action", style="class", alpha=0.8, s=26)
        ax.set_title("AAE latent PCA by action")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        savefig(figures_dir / "fig18_aae_latent_pca_action.png")
        plt.close()
    return frame


def _plot_latent_score_distributions(test_scores: pd.DataFrame, best_score: str, output_path: Path) -> None:
    cols = [col for col in ["score_latent_norm", "score_latent_mahalanobis", "score_disc", best_score] if col in test_scores]
    plot_df = test_scores[["label", *cols]].copy()
    plot_df["class"] = np.where(plot_df["label"].to_numpy() == 1, "anomalous", "normal")
    long = plot_df.melt(id_vars=["label", "class"], value_vars=cols, var_name="score", value_name="value")
    long["log1p_value"] = np.log1p(np.maximum(long["value"].to_numpy(), 0))
    g = sns.FacetGrid(long, col="score", hue="class", col_wrap=2, sharex=False, sharey=False, height=3.1)
    g.map_dataframe(sns.histplot, x="log1p_value", stat="density", bins=35, alpha=0.45)
    g.add_legend()
    g.fig.suptitle("AAE latent/discriminator score distributions", y=1.03)
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def _feature_reconstruction_artifacts(ae_test: dict[str, Any], aae_test: dict[str, Any], tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    ae = _feature_summary(ae_test, "AE")
    aae = _feature_summary(aae_test, "AAE")
    merged = ae.merge(aae, on="feature", suffixes=("_ae", "_aae"))
    merged["delta"] = merged["separation_aae"] - merged["separation_ae"]
    merged.to_csv(tables_dir / "per_feature_reconstruction.csv", index=False)
    top = pd.concat([
        ae.assign(model="AE").nlargest(12, "separation"),
        aae.assign(model="AAE").nlargest(12, "separation"),
    ])
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=top, y="feature", x="separation", hue="model")
    ax.set_title("Top feature reconstruction-error separation")
    ax.set_xlabel("anomaly error - normal error")
    savefig(figures_dir / "fig20_top_feature_reconstruction_separation.png")
    plt.close()
    return merged


def _feature_summary(diag: dict[str, Any], label: str) -> pd.DataFrame:
    y = diag["y"].astype(int)
    feature_error = diag["feature_error"]
    rows = []
    for idx, feature in enumerate(diag["feature_cols"]):
        normal_error = float(feature_error[y == 0, idx].mean()) if (y == 0).any() else np.nan
        anomaly_error = float(feature_error[y == 1, idx].mean()) if (y == 1).any() else np.nan
        rows.append({
            "feature": feature,
            "error_mode": diag.get("error_mode", "mse"),
            f"{label.lower()}_normal_error": normal_error,
            f"{label.lower()}_anomaly_error": anomaly_error,
            "separation": anomaly_error - normal_error,
        })
    return pd.DataFrame(rows)


def _per_action_artifacts(
    ae_val: dict[str, Any],
    ae_test: dict[str, Any],
    aae_val_scores: pd.DataFrame,
    aae_test_scores: pd.DataFrame,
    best_score: str,
    best_threshold: float,
    ae_run_dir: Path,
    aae_run_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    actions = _action_map(aae_run_dir)
    if not actions:
        return pd.DataFrame()
    ae_val_frame = pd.DataFrame({"run_id": ae_val["run_ids"], "label": ae_val["y"], "score": ae_val["score_rec"]})
    ae_threshold = select_threshold(ae_val_frame["label"].to_numpy(), ae_val_frame["score"].to_numpy(), method="best_f1")
    ae_frame = pd.DataFrame({"run_id": ae_test["run_ids"], "label": ae_test["y"], "score": ae_test["score_rec"]})
    ae_frame["action"] = ae_frame["run_id"].map(actions)
    aae_frame = aae_test_scores[["run_id", "label", best_score]].rename(columns={best_score: "score"}).copy()
    aae_frame["action"] = aae_frame["run_id"].map(actions)
    rows = []
    rows.extend(_per_action_rows(ae_frame, ae_threshold, "AE"))
    rows.extend(_per_action_rows(aae_frame, best_threshold, "AAE best score"))
    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "per_action_metrics.csv", index=False)
    if out.empty:
        return out
    top_actions = out.groupby("action")["n_windows"].max().sort_values(ascending=False).head(12).index
    plot_df = out[out["action"].isin(top_actions)]
    plt.figure(figsize=(11, 4.8))
    ax = sns.barplot(data=plot_df, x="action", y="f1", hue="model")
    ax.set_title("Per-action F1")
    ax.tick_params(axis="x", rotation=30)
    savefig(figures_dir / "fig21_per_action_f1.png")
    plt.close()
    plt.figure(figsize=(11, 4.8))
    ax = sns.barplot(data=plot_df, x="action", y="fp", hue="model")
    ax.set_title("Per-action false positives")
    ax.tick_params(axis="x", rotation=30)
    savefig(figures_dir / "fig22_per_action_false_positives.png")
    plt.close()
    return out


def _per_action_rows(frame: pd.DataFrame, threshold: float, model: str) -> list[dict[str, Any]]:
    rows = []
    tmp = frame.dropna(subset=["action"]).copy()
    tmp["pred"] = (tmp["score"].to_numpy() >= threshold).astype(int)
    for action, part in tmp.groupby("action"):
        y = part["label"].to_numpy().astype(int)
        pred = part["pred"].to_numpy().astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(
            {
                "model": model,
                "action": action,
                "n_windows": len(part),
                "anomaly_prevalence": float(y.mean()),
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return rows


def write_official_summary(
    *,
    output_path: str | Path,
    main_results: pd.DataFrame,
    ae_vs_aae: pd.DataFrame,
    aae_specific: pd.DataFrame | None = None,
    window_sensitivity: pd.DataFrame | None = None,
) -> None:
    best = main_results.sort_values("test_pr_auc", ascending=False).iloc[0]
    delta_pr = ae_vs_aae[ae_vs_aae["metric"] == "test_pr_auc"]["AAE_minus_AE"].iloc[0]
    delta_f1 = ae_vs_aae[ae_vs_aae["metric"] == "test_f1"]["AAE_minus_AE"].iloc[0]
    lines = [
        "# AM01 Official Summary",
        "",
        f"- Best main model by test PR-AUC: **{best['model']}** (`PR-AUC={best['test_pr_auc']:.4f}`).",
        f"- AE vs AAE delta under the main protocol: `PR-AUC={delta_pr:.4f}`, `F1={delta_f1:.4f}`.",
        "- Main protocol uses `window_length=64`, `stride=16`, StandardScaler, MSE and seed 42.",
        "- Window sensitivity is limited to `w=32` and `w=64`; `w=128` is intentionally excluded because it leaves too few test windows.",
    ]
    if aae_specific is not None and not aae_specific.empty:
        score = aae_specific.sort_values("val_pr_auc", ascending=False).iloc[0]
        lines.append(
            f"- Best AAE-specific score by validation PR-AUC: `{score['aae_score']}` (`val PR-AUC={score['val_pr_auc']:.4f}`, `test PR-AUC={score['test_pr_auc']:.4f}`)."
        )
    if window_sensitivity is not None and not window_sensitivity.empty:
        compact = _markdown_table(window_sensitivity[["window_length", "model", "test_windows", "test_pr_auc"]])
        lines.extend(["", "## Window Sensitivity", "", compact])
    lines.extend(
        [
            "",
            "## Final Takeaways",
            "",
            "1. The dataset must be evaluated as temporal windows; window length materially changes the evaluation population.",
            "2. Under the official `w=64` protocol, AAE does not robustly improve the AE baseline.",
            "3. AAE-specific latent/discriminator scores are useful diagnostics, but they are treated as supporting analysis rather than the primary comparison.",
            "4. The final conclusion is critical: adversarial regularization is not automatically beneficial for this Kuka anomaly-detection setting.",
        ]
    )
    write_markdown(output_path, lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
