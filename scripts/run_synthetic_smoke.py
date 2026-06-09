#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from am01.data.synthetic import save_synthetic_csv
from am01.pipeline import run_experiment

DATA = ROOT / "data" / "raw" / "synthetic_smoke.csv"
OUT = ROOT / "results" / "runs" / "synthetic_smoke"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    save_synthetic_csv(DATA, runs=9, length=96, joints=2, seed=13)
    print(f"Synthetic dataset written to {DATA}")

    base = {
        "seed": 42,
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
    configs = {
        "pca": {**base, "model": {"type": "pca", "n_components": 0.95}},
        "ae_mlp": {**base, "model": {"type": "ae_mlp", "hidden_dims": [16], "latent_dim": 4, "dropout": 0.0},
                   "training": {"epochs": 2, "batch_size": 16, "lr": 1e-3, "weight_decay": 0.0, "loss": "mse", "patience": 2, "device": "cpu", "show_progress": False}},
        "aae_mlp": {**base, "model": {"type": "aae_mlp", "hidden_dims": [16], "latent_dim": 4, "dropout": 0.0, "discriminator_hidden_dims": [8], "lambda_adv": 0.1},
                    "training": {"epochs": 2, "batch_size": 16, "lr": 1e-3, "lr_discriminator": 1e-3, "lr_adversarial": 1e-3, "weight_decay": 0.0, "loss": "mse", "patience": 2, "device": "cpu", "show_progress": False}},
    }

    for name, cfg in configs.items():
        print(f"Running {name}...")
        result = run_experiment(cfg, data_path=DATA, output_dir=OUT / name)
        assert "test_metrics" in result
        assert "threshold" in result
        print(json.dumps({"model": name, "test_f1": result["test_metrics"].get("f1")}, indent=2))
    print(f"Smoke test completed. Results in {OUT}")


if __name__ == "__main__":
    main()
