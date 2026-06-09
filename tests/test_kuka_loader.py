from pathlib import Path

import numpy as np

from am01.data.io import infer_feature_columns, load_timeseries_data


def test_kuka_velocity_numpy_loader_aligns_labels_and_runs(tmp_path: Path):
    names = np.array(["action", "sensor_a", "sensor_b", "anomaly"])
    np.save(tmp_path / "KukaColumnNames.npy", names)
    normal = np.array([
        [0, 1.0, 2.0],
        [0, 1.1, 2.1],
        [1, 1.2, 2.2],
        [1, 1.3, 2.3],
    ])
    slow = np.array([
        [0, 9.0, 8.0, 1],
        [0, 9.1, 8.1, 1],
        [2, 9.2, 8.2, 1],
    ])
    np.save(tmp_path / "KukaNormal.npy", normal)
    np.save(tmp_path / "KukaSlow.npy", slow)

    df = load_timeseries_data(tmp_path, data_format="kuka_npy")

    assert set(df["source_file"]) == {"normal", "slow"}
    assert df["label"].tolist().count(0) == 4
    assert df["label"].tolist().count(1) == 3
    assert df["run_id"].nunique() == 4
    assert "anomaly" not in df.columns

    features = infer_feature_columns(df)
    assert features == ["sensor_a", "sensor_b"]
