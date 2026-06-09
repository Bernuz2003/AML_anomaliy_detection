import numpy as np
import pandas as pd

from am01.data.preprocessing import TimeSeriesScaler, select_scaler_fit_rows
from am01.data.windowing import assert_disjoint_runs, make_windows, split_by_run


def test_split_by_run_has_no_overlap():
    df = pd.DataFrame({
        "run_id": np.repeat(["a", "b", "c", "d", "e"], 10),
        "t": list(range(10)) * 5,
        "label": 0,
        "x": np.random.randn(50),
    })
    train, val, test = split_by_run(df, seed=0)
    assert_disjoint_runs(train, val, test)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_by_run_stratifies_run_labels_when_possible():
    runs = [f"n{i}" for i in range(6)] + [f"a{i}" for i in range(6)]
    labels = [0] * 6 + [1] * 6
    df = pd.DataFrame({
        "run_id": np.repeat(runs, 4),
        "t": list(range(4)) * len(runs),
        "label": np.repeat(labels, 4),
        "x": np.random.randn(len(runs) * 4),
    })
    train, val, test = split_by_run(df, seed=0, stratify_by_label=True)
    for split in [train, val, test]:
        run_labels = split.groupby("run_id")["label"].max()
        assert set(run_labels.unique()) == {0, 1}


def test_window_labels_use_anomaly_fraction():
    df = pd.DataFrame({
        "run_id": ["r"] * 8,
        "t": np.arange(8),
        "label": [0, 0, 1, 1, 0, 0, 0, 0],
        "x": np.arange(8, dtype=float),
        "y": np.arange(8, dtype=float) * 2,
    })
    windows = make_windows(
        df,
        feature_cols=["x", "y"],
        window_length=4,
        stride=2,
        anomaly_fraction=0.25,
    )
    assert windows.X.shape == (3, 4, 2)
    assert windows.y.tolist() == [1, 1, 0]


def test_scaler_fit_rows_use_only_normal_training_samples():
    df = pd.DataFrame({
        "label": [0, 0, 1, 1],
        "x": [1.0, 2.0, 100.0, 101.0],
    })
    fit_df = select_scaler_fit_rows(df, label_col="label", normal_label=0, fit_only_normal=True)
    scaler = TimeSeriesScaler("standard").fit(fit_df, ["x"])
    transformed = scaler.transform(df)
    # Normal rows are centered around zero; anomalous rows remain very large.
    assert abs(transformed.loc[0, "x"] + 1.0) < 1e-6
    assert transformed.loc[2, "x"] > 100
