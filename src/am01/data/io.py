from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESERVED_DEFAULTS = {
    "run_id",
    "t",
    "time",
    "timestamp",
    "label",
    "anomaly",
    "split",
    "action",
    "source",
    "source_file",
}


KUKA_COLUMN_NAMES_FILE = "KukaColumnNames.npy"
KUKA_NORMAL_FILE = "KukaNormal.npy"
KUKA_SLOW_FILE = "KukaSlow.npy"


def load_timeseries_csv(
    path: str | Path,
    *,
    run_col: str = "run_id",
    time_col: str | None = "t",
) -> pd.DataFrame:
    """Load a single CSV file or a directory of CSV files.

    If a directory is provided and a file does not contain ``run_col``, the file stem is
    used as run identifier. This is useful when each CSV corresponds to one robot run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    if path.is_dir():
        frames: list[pd.DataFrame] = []
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {path}")
        for csv_path in csv_files:
            frame = pd.read_csv(csv_path)
            if run_col not in frame.columns:
                frame[run_col] = csv_path.stem
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)

    if run_col not in df.columns:
        raise ValueError(
            f"Missing run column '{run_col}'. Provide it in the CSV or use one CSV per run."
        )

    if time_col and time_col in df.columns:
        df = df.sort_values([run_col, time_col], kind="mergesort").reset_index(drop=True)
    else:
        df = df.sort_values([run_col], kind="mergesort").reset_index(drop=True)
    return df


def _looks_like_kuka_velocity_dataset(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / KUKA_COLUMN_NAMES_FILE).exists()
        and (path / KUKA_NORMAL_FILE).exists()
        and (path / KUKA_SLOW_FILE).exists()
    )


def _frame_from_kuka_array(
    array_path: Path,
    *,
    column_names: list[str],
    dataset_name: str,
    label_col: str,
    anomaly_col: str,
) -> pd.DataFrame:
    values = np.load(array_path)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D NumPy array in {array_path}, got shape {values.shape}.")

    if values.shape[1] == len(column_names):
        cols = list(column_names)
    elif values.shape[1] == len(column_names) - 1 and column_names[-1] == anomaly_col:
        cols = list(column_names[:-1])
    else:
        raise ValueError(
            f"Column mismatch for {array_path.name}: array has {values.shape[1]} columns, "
            f"but {KUKA_COLUMN_NAMES_FILE} declares {len(column_names)} columns."
        )

    frame = pd.DataFrame(values, columns=cols)
    if anomaly_col in frame.columns:
        frame[label_col] = (frame[anomaly_col].to_numpy() != 0).astype(np.int64)
        if anomaly_col != label_col:
            frame = frame.drop(columns=[anomaly_col])
    else:
        frame[label_col] = 0
    frame["source_file"] = dataset_name
    return frame


def _assign_kuka_run_ids(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    run_col: str,
    time_col: str | None,
    action_col: str,
    run_strategy: str,
    fixed_run_length: int,
) -> pd.DataFrame:
    frame = frame.copy()
    if run_strategy == "file":
        frame[run_col] = dataset_name
        if time_col:
            frame[time_col] = np.arange(len(frame), dtype=np.int64)
        return frame

    if run_strategy == "fixed_length" or action_col not in frame.columns:
        if fixed_run_length <= 0:
            raise ValueError("fixed_run_length must be positive.")
        segment_ids = np.arange(len(frame), dtype=np.int64) // fixed_run_length
    elif run_strategy == "action_segments":
        if len(frame) == 0:
            frame[run_col] = []
            if time_col:
                frame[time_col] = []
            return frame
        action = frame[action_col].to_numpy()
        boundaries = np.empty(len(frame), dtype=bool)
        boundaries[0] = True
        boundaries[1:] = action[1:] != action[:-1]
        segment_ids = np.cumsum(boundaries, dtype=np.int64) - 1
    else:
        raise ValueError(f"Unknown Kuka run strategy: {run_strategy}")

    frame["_segment_id"] = segment_ids
    frame[run_col] = [
        f"{dataset_name}_seg_{int(seg):04d}"
        for seg in segment_ids
    ]
    if time_col:
        frame[time_col] = frame.groupby("_segment_id", sort=False).cumcount().astype(np.int64)
    return frame.drop(columns=["_segment_id"])


def load_kuka_velocity_dataset(
    path: str | Path,
    *,
    run_col: str = "run_id",
    time_col: str | None = "t",
    label_col: str = "label",
    anomaly_col: str = "anomaly",
    action_col: str = "action",
    run_strategy: str = "action_segments",
    fixed_run_length: int = 512,
) -> pd.DataFrame:
    """Load the AM01 KukaVelocityDataset distributed as NumPy arrays.

    The real dataset currently present in ``data/raw/KukaVelocityDataset`` contains
    one normal array without an anomaly column and one slow/anomalous array with an
    ``anomaly`` column. This adapter aligns both arrays into a single dataframe with
    ``run_id``, optional ``t`` and a binary ``label`` column.
    """
    path = Path(path)
    if not _looks_like_kuka_velocity_dataset(path):
        raise ValueError(
            f"{path} does not look like the expected KukaVelocityDataset directory. "
            f"Required files: {KUKA_COLUMN_NAMES_FILE}, {KUKA_NORMAL_FILE}, {KUKA_SLOW_FILE}."
        )

    column_names = [str(name) for name in np.load(path / KUKA_COLUMN_NAMES_FILE, allow_pickle=True)]
    frames = []
    for dataset_name, file_name in (("normal", KUKA_NORMAL_FILE), ("slow", KUKA_SLOW_FILE)):
        frame = _frame_from_kuka_array(
            path / file_name,
            column_names=column_names,
            dataset_name=dataset_name,
            label_col=label_col,
            anomaly_col=anomaly_col,
        )
        frame = _assign_kuka_run_ids(
            frame,
            dataset_name=dataset_name,
            run_col=run_col,
            time_col=time_col,
            action_col=action_col,
            run_strategy=run_strategy,
            fixed_run_length=fixed_run_length,
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    sort_cols = [run_col]
    if time_col and time_col in df.columns:
        sort_cols.append(time_col)
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def load_timeseries_data(
    path: str | Path,
    *,
    run_col: str = "run_id",
    time_col: str | None = "t",
    label_col: str = "label",
    data_format: str = "auto",
    anomaly_col: str = "anomaly",
    action_col: str = "action",
    kuka_run_strategy: str = "action_segments",
    kuka_fixed_run_length: int = 512,
) -> pd.DataFrame:
    """Load supported AM01 time-series formats.

    Supported formats are regular CSV data and the real KukaVelocityDataset NumPy
    directory. ``data_format='auto'`` chooses the Kuka adapter when the expected
    NumPy files are present; otherwise it falls back to CSV loading.
    """
    path = Path(path)
    if data_format not in {"auto", "csv", "kuka_npy"}:
        raise ValueError(f"Unknown data format: {data_format}")
    if data_format == "kuka_npy" or (data_format == "auto" and _looks_like_kuka_velocity_dataset(path)):
        return load_kuka_velocity_dataset(
            path,
            run_col=run_col,
            time_col=time_col,
            label_col=label_col,
            anomaly_col=anomaly_col,
            action_col=action_col,
            run_strategy=kuka_run_strategy,
            fixed_run_length=kuka_fixed_run_length,
        )
    return load_timeseries_csv(path, run_col=run_col, time_col=time_col)


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str] | None = None,
    run_col: str = "run_id",
    time_col: str | None = "t",
    label_col: str | None = "label",
    split_col: str | None = None,
) -> list[str]:
    """Return numeric sensor columns, excluding metadata columns."""
    if feature_cols:
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Configured feature columns are missing: {missing}")
        return list(feature_cols)

    reserved = {run_col}
    if time_col:
        reserved.add(time_col)
    if label_col:
        reserved.add(label_col)
    if split_col:
        reserved.add(split_col)
    reserved |= RESERVED_DEFAULTS

    numeric_cols = [
        col for col in df.columns
        if col not in reserved and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        raise ValueError(
            "Could not infer any numeric feature columns. Configure data.feature_cols explicitly."
        )
    return numeric_cols


def validate_no_missing_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Feature columns must be numeric: {non_numeric}")


def save_dataframe_summary(df: pd.DataFrame, feature_cols: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "feature": feature_cols,
        "mean": [df[c].mean() for c in feature_cols],
        "std": [df[c].std() for c in feature_cols],
        "min": [df[c].min() for c in feature_cols],
        "max": [df[c].max() for c in feature_cols],
        "missing": [df[c].isna().sum() for c in feature_cols],
    })
    summary.to_csv(output_path, index=False)
