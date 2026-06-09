from pathlib import Path

import yaml

from am01.data.synthetic import save_synthetic_csv
from am01.pipeline import run_experiment


def _small_cfg(model_type: str):
    cfg = {
        "seed": 7,
        "data": {
            "run_col": "run_id",
            "time_col": "t",
            "label_col": "label",
            "normal_label": 0,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
        },
        "preprocessing": {"scaler": "standard", "fit_only_normal": True, "missing_strategy": "interpolate"},
        "windowing": {"window_length": 24, "stride": 12, "anomaly_fraction": 0.1},
        "evaluation": {"threshold": "best_f1", "fallback_percentile": 99.0},
    }
    if model_type == "ae_mlp":
        cfg["model"] = {"type": "ae_mlp", "hidden_dims": [16], "latent_dim": 4, "dropout": 0.0}
        cfg["training"] = {"epochs": 2, "batch_size": 16, "lr": 1e-3, "weight_decay": 0.0, "loss": "mse", "patience": 2, "device": "cpu", "show_progress": False}
    elif model_type == "aae_mlp":
        cfg["model"] = {"type": "aae_mlp", "hidden_dims": [16], "latent_dim": 4, "dropout": 0.0, "discriminator_hidden_dims": [8], "lambda_adv": 0.1}
        cfg["training"] = {"epochs": 2, "batch_size": 16, "lr": 1e-3, "lr_discriminator": 1e-3, "lr_adversarial": 1e-3, "weight_decay": 0.0, "loss": "mse", "patience": 2, "device": "cpu", "show_progress": False}
    elif model_type == "pca":
        cfg["model"] = {"type": "pca", "n_components": 0.95}
    else:
        raise ValueError(model_type)
    return cfg


def test_pipeline_runs_pca_and_ae(tmp_path: Path):
    data_path = tmp_path / "synthetic.csv"
    save_synthetic_csv(data_path, runs=9, length=96, joints=2, seed=3)
    for model_type in ["pca", "ae_mlp", "aae_mlp"]:
        result = run_experiment(_small_cfg(model_type), data_path=data_path, output_dir=tmp_path / model_type)
        assert "test_metrics" in result
        assert "threshold" in result
        assert (tmp_path / model_type / "metrics.json").exists()


def test_prepare_outputs_include_processed_splits(tmp_path: Path):
    data_path = tmp_path / "synthetic.csv"
    save_synthetic_csv(data_path, runs=9, length=96, joints=2, seed=11)
    result = run_experiment(_small_cfg("pca"), data_path=data_path, output_dir=tmp_path / "pca")
    assert "split_summary" in result
    for name in ["processed_train.npz", "processed_val.npz", "processed_test.npz"]:
        assert (tmp_path / "pca" / name).exists()
    assert (tmp_path / "pca" / "dataset_summary.csv").exists()
    assert (tmp_path / "pca" / "preprocessing_config.json").exists()
