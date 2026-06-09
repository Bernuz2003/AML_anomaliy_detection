from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def _require_labels(frame: pd.DataFrame, path: str | Path) -> np.ndarray:
    if "label" not in frame.columns:
        raise ValueError(f"Cannot create supervised plots: {path} has no label column.")
    return frame["label"].to_numpy().astype(int)


def plot_score_distribution(scores_csv: str | Path, output_path: str | Path, *, title: str) -> None:
    frame = pd.read_csv(scores_csv)
    y = _require_labels(frame, scores_csv)
    scores = frame["score"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y == 0], bins=50, alpha=0.65, label="normal", density=True)
    ax.hist(scores[y == 1], bins=50, alpha=0.65, label="anomalous", density=True)
    ax.set_title(title)
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc_pr(scores_csv: str | Path, roc_path: str | Path, pr_path: str | Path, *, title_prefix: str) -> None:
    frame = pd.read_csv(scores_csv)
    y = _require_labels(frame, scores_csv)
    scores = frame["score"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"Cannot create ROC/PR curves from a single-class label vector: {scores_csv}")

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y, scores, ax=ax)
    ax.set_title(f"{title_prefix} ROC")
    fig.tight_layout()
    Path(roc_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y, scores, ax=ax)
    ax.set_title(f"{title_prefix} Precision-Recall")
    fig.tight_layout()
    Path(pr_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)


def plot_score_timeline(scores_csv: str | Path, output_path: str | Path, *, threshold: float | None = None) -> None:
    frame = pd.read_csv(scores_csv)
    y = frame["label"].to_numpy().astype(int) if "label" in frame.columns else None

    run_ids = frame["run_id"].astype(str).unique()
    n_runs = min(len(run_ids), 6)
    fig, axes = plt.subplots(n_runs, 1, figsize=(10, max(2.5, 2.0 * n_runs)), sharex=False)
    if n_runs == 1:
        axes = [axes]

    for ax, run_id in zip(axes, run_ids[:n_runs]):
        part = frame[frame["run_id"].astype(str) == run_id].sort_values("start")
        ax.plot(part["start"], part["score"], label="score", linewidth=1.4)
        if threshold is not None:
            ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.0, label="threshold")
        if y is not None:
            anomalous = part["label"].to_numpy().astype(int) == 1
            if anomalous.any():
                ymin, ymax = ax.get_ylim()
                ax.fill_between(
                    part["start"],
                    ymin,
                    ymax,
                    where=anomalous,
                    color="tab:orange",
                    alpha=0.18,
                    step="mid",
                    label="label",
                )
        ax.set_title(str(run_id))
        ax.set_ylabel("score")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("window start")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
