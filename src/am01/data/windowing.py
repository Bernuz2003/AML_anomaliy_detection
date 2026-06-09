from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowedData:
    X: np.ndarray
    y: np.ndarray | None
    run_ids: np.ndarray
    starts: np.ndarray
    feature_cols: list[str]

    @property
    def n_windows(self) -> int:
        return int(self.X.shape[0])

    @property
    def window_length(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[2])


def split_by_run(
    df: pd.DataFrame,
    *,
    run_col: str = "run_id",
    label_col: str | None = "label",
    normal_label: int | float | str = 0,
    split_col: str | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    stratify_by_label: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe by runs, preventing overlapping windows across splits."""
    if split_col and split_col in df.columns:
        split_values = set(df[split_col].astype(str).str.lower().unique())
        required = {"train", "val", "test"}
        if not required.issubset(split_values):
            raise ValueError(
                f"Configured split_col='{split_col}' must contain train/val/test. Found: {sorted(split_values)}"
            )
        low = df[split_col].astype(str).str.lower()
        return (
            df[low == "train"].copy(),
            df[low == "val"].copy(),
            df[low == "test"].copy(),
        )

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1. Got {total}.")

    runs = np.array(sorted(df[run_col].unique()))
    if len(runs) < 3:
        raise ValueError("At least three runs are required for train/val/test splitting.")
    rng = np.random.default_rng(seed)

    def split_run_array(run_values: np.ndarray) -> tuple[set, set, set]:
        run_values = np.array(sorted(run_values))
        rng.shuffle(run_values)
        n_train = max(1, int(round(len(run_values) * train_ratio)))
        n_val = max(1, int(round(len(run_values) * val_ratio)))
        if n_train + n_val >= len(run_values):
            n_train = max(1, len(run_values) - 2)
            n_val = 1
        train_values = set(run_values[:n_train])
        val_values = set(run_values[n_train:n_train + n_val])
        test_values = set(run_values[n_train + n_val:])
        return train_values, val_values, test_values

    if stratify_by_label and label_col and label_col in df.columns:
        run_labels = df.groupby(run_col, sort=False)[label_col].apply(
            lambda values: int((values.to_numpy() != normal_label).any())
        )
        class_counts = run_labels.value_counts()
        if len(class_counts) > 1 and int(class_counts.min()) >= 3:
            train_runs: set = set()
            val_runs: set = set()
            test_runs: set = set()
            for class_value in sorted(run_labels.unique()):
                class_runs = run_labels[run_labels == class_value].index.to_numpy()
                tr, va, te = split_run_array(class_runs)
                train_runs.update(tr)
                val_runs.update(va)
                test_runs.update(te)
            if train_runs and val_runs and test_runs:
                return (
                    df[df[run_col].isin(train_runs)].copy(),
                    df[df[run_col].isin(val_runs)].copy(),
                    df[df[run_col].isin(test_runs)].copy(),
                )

    rng.shuffle(runs)

    train_runs, val_runs, test_runs = split_run_array(runs)

    return (
        df[df[run_col].isin(train_runs)].copy(),
        df[df[run_col].isin(val_runs)].copy(),
        df[df[run_col].isin(test_runs)].copy(),
    )


def make_windows(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    run_col: str = "run_id",
    label_col: str | None = "label",
    normal_label: int | float | str = 0,
    window_length: int = 64,
    stride: int = 16,
    anomaly_fraction: float = 0.10,
) -> WindowedData:
    """Create fixed-length windows from each run independently."""
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if not (0.0 <= anomaly_fraction <= 1.0):
        raise ValueError("anomaly_fraction must be in [0, 1].")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    run_ids: list[object] = []
    starts: list[int] = []
    has_labels = label_col is not None and label_col in df.columns

    for run_id, part in df.groupby(run_col, sort=False):
        values = part[feature_cols].to_numpy(dtype=np.float32)
        if len(values) < window_length:
            continue
        run_labels = None
        if has_labels:
            run_labels = (part[label_col].to_numpy() != normal_label).astype(np.int64)
        for start in range(0, len(values) - window_length + 1, stride):
            end = start + window_length
            windows.append(values[start:end])
            run_ids.append(run_id)
            starts.append(start)
            if has_labels and run_labels is not None:
                frac = float(run_labels[start:end].mean())
                labels.append(int(frac >= anomaly_fraction and frac > 0.0))

    if not windows:
        raise ValueError(
            f"No windows generated. Check window_length={window_length}, stride={stride}, and run lengths."
        )
    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64) if has_labels else None
    return WindowedData(
        X=X,
        y=y,
        run_ids=np.asarray(run_ids),
        starts=np.asarray(starts, dtype=np.int64),
        feature_cols=list(feature_cols),
    )


class WindowDataset(Dataset):
    """PyTorch dataset wrapping window tensors and metadata."""

    def __init__(self, data: WindowedData, *, normal_only: bool = False):
        mask = np.ones(data.X.shape[0], dtype=bool)
        if normal_only and data.y is not None:
            mask = data.y == 0
            if not mask.any():
                raise ValueError("normal_only=True but no normal windows are available.")
        self.X = torch.from_numpy(data.X[mask]).float()
        self.y = None if data.y is None else torch.from_numpy(data.y[mask]).long()
        self.run_ids = data.run_ids[mask]
        self.starts = data.starts[mask]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.y is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            y = self.y[idx]
        return x, y, str(self.run_ids[idx]), int(self.starts[idx])


def assert_disjoint_runs(*frames: pd.DataFrame, run_col: str = "run_id") -> None:
    sets = [set(frame[run_col].unique()) for frame in frames]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(f"Run leakage detected between split {i} and {j}: {sorted(overlap)[:10]}")
