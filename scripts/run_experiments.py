#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.pipeline import run_experiment
from am01.utils.config import load_config


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _optional(values):
    return values if values else [None]


def _set_if_not_none(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        mapping[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of AM01 experiments.")
    parser.add_argument("--configs", nargs="+", required=True, help="One or more YAML config files.")
    parser.add_argument("--data", required=True, help="CSV data or KukaVelocityDataset directory.")
    parser.add_argument("--output", default="results/runs", help="Output root directory.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--window-lengths", nargs="*", type=int, default=None)
    parser.add_argument("--latent-dims", nargs="*", type=int, default=None)
    parser.add_argument("--lambda-advs", nargs="*", type=float, default=None)
    parser.add_argument("--scalers", nargs="*", choices=["standard", "robust", "minmax"], default=None)
    parser.add_argument("--losses", nargs="*", choices=["mse", "mae", "huber"], default=None)
    parser.add_argument("--threshold-methods", nargs="*", choices=["best_f1", "percentile", "normal_percentile"], default=None)
    parser.add_argument("--summary-name", default="experiment_summary.csv")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for config_path in args.configs:
        base_cfg = load_config(config_path)
        config_stem = Path(config_path).stem
        model_type = base_cfg.get("model", {}).get("type", "ae_mlp")
        seeds = args.seeds if args.seeds else [int(base_cfg.get("seed", 42))]
        latent_dims = args.latent_dims if model_type in {"ae_mlp", "aae_mlp", "ae_conv1d"} else None
        lambda_advs = args.lambda_advs if model_type == "aae_mlp" else None
        losses = args.losses if model_type in {"ae_mlp", "aae_mlp", "ae_conv1d"} else None

        for seed, window_length, latent_dim, lambda_adv, scaler, loss, threshold_method in itertools.product(
            seeds,
            _optional(args.window_lengths),
            _optional(latent_dims),
            _optional(lambda_advs),
            _optional(args.scalers),
            _optional(losses),
            _optional(args.threshold_methods),
        ):
            cfg = deepcopy(base_cfg)
            cfg.setdefault("data", {})
            cfg.setdefault("preprocessing", {})
            cfg.setdefault("windowing", {})
            cfg.setdefault("model", {})
            cfg.setdefault("training", {})
            cfg.setdefault("evaluation", {})
            cfg["seed"] = seed
            _set_if_not_none(cfg["windowing"], "window_length", window_length)
            _set_if_not_none(cfg["preprocessing"], "scaler", scaler)
            _set_if_not_none(cfg["training"], "loss", loss)
            _set_if_not_none(cfg["evaluation"], "threshold", threshold_method)
            if latent_dim is not None and model_type in {"ae_mlp", "aae_mlp", "ae_conv1d"}:
                cfg["model"]["latent_dim"] = latent_dim
            if lambda_adv is not None and model_type == "aae_mlp":
                cfg["model"]["lambda_adv"] = lambda_adv

            run_parts = [config_stem, f"seed{seed}"]
            if window_length is not None:
                run_parts.append(f"w{window_length}")
            if latent_dim is not None and model_type in {"ae_mlp", "aae_mlp", "ae_conv1d"}:
                run_parts.append(f"z{latent_dim}")
            if lambda_adv is not None and model_type == "aae_mlp":
                run_parts.append(f"lam{lambda_adv:g}")
            if scaler is not None:
                run_parts.append(f"scaler{scaler}")
            if loss is not None and model_type in {"ae_mlp", "aae_mlp", "ae_conv1d"}:
                run_parts.append(f"loss{loss}")
            if threshold_method is not None:
                run_parts.append(f"thr{threshold_method}")
            run_name = "_".join(_slug(part) for part in run_parts)
            run_dir = output_root / run_name

            print(f"Running {run_name}...")
            result = run_experiment(cfg, data_path=args.data, output_dir=run_dir)
            row = {
                "run_name": run_name,
                "config": str(config_path),
                "model_type": result["model_type"],
                "seed": seed,
                "window_length": cfg.get("windowing", {}).get("window_length"),
                "latent_dim": cfg.get("model", {}).get("latent_dim"),
                "lambda_adv": cfg.get("model", {}).get("lambda_adv"),
                "scaler": cfg.get("preprocessing", {}).get("scaler"),
                "loss": cfg.get("training", {}).get("loss"),
                "threshold_method": cfg.get("evaluation", {}).get("threshold"),
                "threshold": result["threshold"],
            }
            for key, value in result.get("test_metrics", {}).items():
                row[f"test_{key}"] = value
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = output_root / args.summary_name
    summary.to_csv(summary_path, index=False)
    print(f"Experiment summary written to {summary_path}")


if __name__ == "__main__":
    main()
