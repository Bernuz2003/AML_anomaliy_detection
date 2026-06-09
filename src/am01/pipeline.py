from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from am01.data.audit import save_dataset_summary
from am01.data.io import infer_feature_columns, load_timeseries_data, save_dataframe_summary, validate_no_missing_feature_columns
from am01.data.preprocessing import TimeSeriesScaler, clean_missing_values, select_scaler_fit_rows
from am01.data.windowing import WindowDataset, WindowedData, assert_disjoint_runs, make_windows, split_by_run
from am01.evaluation.metrics import all_metrics, select_threshold
from am01.evaluation.scoring import score_autoencoder
from am01.models.aae import LatentDiscriminator
from am01.models.autoencoders import Conv1dAutoencoder, MLPAutoencoder
from am01.models.baselines import IsolationForestDetector, PCADetector
from am01.training.trainer import train_adversarial_autoencoder, train_autoencoder
from am01.utils.device import resolve_device
from am01.utils.seed import set_seed


@dataclass
class PreparedData:
    train: WindowedData
    val: WindowedData
    test: WindowedData
    scaler: TimeSeriesScaler
    feature_cols: list[str]
    split_summary: dict[str, Any]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _filter_normal_windows(data: WindowedData) -> np.ndarray:
    if data.y is None:
        return data.X
    normal = data.X[data.y == 0]
    if len(normal) == 0:
        raise ValueError("No normal training windows available. Check labels and split configuration.")
    return normal


def _save_windowed_npz(path: Path, data: WindowedData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=data.X,
        y=np.asarray([], dtype=np.int64) if data.y is None else data.y,
        has_labels=np.asarray(data.y is not None),
        run_ids=data.run_ids.astype(str),
        starts=data.starts,
        feature_cols=np.asarray(data.feature_cols, dtype=str),
    )


def prepare_data(data_path: str | Path, cfg: dict[str, Any], output_dir: str | Path | None = None) -> PreparedData:
    data_cfg = cfg.get("data", {})
    prep_cfg = cfg.get("preprocessing", {})
    win_cfg = cfg.get("windowing", {})
    seed = int(cfg.get("seed", 42))

    run_col = data_cfg.get("run_col", "run_id")
    time_col = data_cfg.get("time_col", "t")
    label_col = data_cfg.get("label_col", "label")
    split_col = data_cfg.get("split_col", None)
    normal_label = data_cfg.get("normal_label", 0)

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
        split_col=split_col,
    )
    validate_no_missing_feature_columns(df, feature_cols)
    df = clean_missing_values(
        df,
        feature_cols,
        run_col=run_col,
        strategy=prep_cfg.get("missing_strategy", "interpolate"),
    )

    train_df, val_df, test_df = split_by_run(
        df,
        run_col=run_col,
        label_col=label_col,
        normal_label=normal_label,
        split_col=split_col,
        train_ratio=float(data_cfg.get("train_ratio", 0.6)),
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        test_ratio=float(data_cfg.get("test_ratio", 0.2)),
        stratify_by_label=bool(data_cfg.get("stratify_by_label", True)),
        seed=seed,
    )
    assert_disjoint_runs(train_df, val_df, test_df, run_col=run_col)

    scaler_fit_df = select_scaler_fit_rows(
        train_df,
        label_col=label_col,
        normal_label=normal_label,
        fit_only_normal=bool(prep_cfg.get("fit_only_normal", True)),
    )
    scaler = TimeSeriesScaler(name=prep_cfg.get("scaler", "standard"))
    scaler.fit(scaler_fit_df, feature_cols)
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    make_kwargs = dict(
        feature_cols=feature_cols,
        run_col=run_col,
        label_col=label_col,
        normal_label=normal_label,
        window_length=int(win_cfg.get("window_length", 64)),
        stride=int(win_cfg.get("stride", 16)),
        anomaly_fraction=float(win_cfg.get("anomaly_fraction", 0.10)),
    )
    train_w = make_windows(train_scaled, **make_kwargs)
    val_w = make_windows(val_scaled, **make_kwargs)
    test_w = make_windows(test_scaled, **make_kwargs)

    summary = {
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "runs": {
            "train": int(train_df[run_col].nunique()),
            "val": int(val_df[run_col].nunique()),
            "test": int(test_df[run_col].nunique()),
        },
        "windows": {
            "train": train_w.n_windows,
            "val": val_w.n_windows,
            "test": test_w.n_windows,
            "train_normal": int((_filter_normal_windows(train_w)).shape[0]),
        },
        "feature_cols": feature_cols,
        "window_length": train_w.window_length,
        "n_channels": train_w.n_channels,
    }
    if train_w.y is not None:
        summary["anomalous_windows"] = {
            "train": int(train_w.y.sum()),
            "val": int(val_w.y.sum()),
            "test": int(test_w.y.sum()),
        }

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_json(summary, output_dir / "split_summary.json")
        save_json(
            {
                "seed": seed,
                "data": data_cfg,
                "preprocessing": prep_cfg,
                "windowing": win_cfg,
            },
            output_dir / "preprocessing_config.json",
        )
        save_dataset_summary(
            df,
            feature_cols,
            output_dir / "dataset_summary.csv",
            run_col=run_col,
            label_col=label_col,
            normal_label=normal_label,
            include_feature_stats=bool(prep_cfg.get("dataset_summary_feature_stats", True)),
        )
        save_dataframe_summary(df, feature_cols, output_dir / "feature_summary.csv")
        _save_windowed_npz(output_dir / "processed_train.npz", train_w)
        _save_windowed_npz(output_dir / "processed_val.npz", val_w)
        _save_windowed_npz(output_dir / "processed_test.npz", test_w)
        joblib.dump(scaler, output_dir / "scaler.joblib")

    return PreparedData(train=train_w, val=val_w, test=test_w, scaler=scaler, feature_cols=feature_cols, split_summary=summary)


def build_autoencoder(model_cfg: dict[str, Any], *, window_length: int, n_channels: int) -> torch.nn.Module:
    model_type = model_cfg.get("type", "ae_mlp")
    if model_type in {"ae_mlp", "aae_mlp"}:
        return MLPAutoencoder(
            window_length=window_length,
            n_channels=n_channels,
            latent_dim=int(model_cfg.get("latent_dim", 16)),
            hidden_dims=list(model_cfg.get("hidden_dims", [256, 128])),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if model_type == "ae_conv1d":
        return Conv1dAutoencoder(
            window_length=window_length,
            n_channels=n_channels,
            latent_dim=int(model_cfg.get("latent_dim", 16)),
            hidden_channels=int(model_cfg.get("hidden_channels", 32)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported autoencoder type: {model_type}")


def _make_loaders(prepared: PreparedData, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    train_ds = WindowDataset(prepared.train, normal_only=True)
    val_train_ds = WindowDataset(prepared.val, normal_only=True) if prepared.val.y is not None and (prepared.val.y == 0).any() else WindowDataset(prepared.val)
    val_eval_ds = WindowDataset(prepared.val)
    test_eval_ds = WindowDataset(prepared.test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_train_loader = DataLoader(val_train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    val_eval_loader = DataLoader(val_eval_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_eval_loader = DataLoader(test_eval_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_train_loader, val_eval_loader, test_eval_loader


def _save_scores(path: Path, scores: np.ndarray, labels: np.ndarray | None, run_ids: np.ndarray, starts: np.ndarray) -> None:
    frame = pd.DataFrame({"run_id": run_ids, "start": starts, "score": scores})
    if labels is not None:
        frame["label"] = labels
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_experiment(config: dict[str, Any], *, data_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    save_json(config, output_dir / "config_used.json")

    prepared = prepare_data(data_path, config, output_dir=output_dir)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    model_type = model_cfg.get("type", "ae_mlp")

    if model_type == "pca":
        detector = PCADetector(n_components=model_cfg.get("n_components", 0.95))
        detector.fit(_filter_normal_windows(prepared.train))
        val_scores = detector.score(prepared.val.X)
        test_scores = detector.score(prepared.test.X)
        threshold = select_threshold(
            prepared.val.y,
            val_scores,
            method=eval_cfg.get("threshold", "best_f1"),
            fallback_percentile=float(eval_cfg.get("fallback_percentile", 99.0)),
        )
        joblib.dump(detector, output_dir / "model.joblib")
    elif model_type == "isolation_forest":
        detector = IsolationForestDetector(
            n_estimators=int(model_cfg.get("n_estimators", 200)),
            contamination=model_cfg.get("contamination", "auto"),
            random_state=seed,
            feature_mode=model_cfg.get("feature_mode", "statistical"),
        )
        detector.fit(_filter_normal_windows(prepared.train))
        val_scores = detector.score(prepared.val.X)
        test_scores = detector.score(prepared.test.X)
        threshold = select_threshold(
            prepared.val.y,
            val_scores,
            method=eval_cfg.get("threshold", "best_f1"),
            fallback_percentile=float(eval_cfg.get("fallback_percentile", 99.0)),
        )
        joblib.dump(detector, output_dir / "model.joblib")
    else:
        device = resolve_device(train_cfg.get("device", "auto"))
        batch_size = int(train_cfg.get("batch_size", 128))
        train_loader, val_train_loader, val_eval_loader, test_eval_loader = _make_loaders(prepared, batch_size)
        autoencoder = build_autoencoder(
            model_cfg,
            window_length=prepared.train.window_length,
            n_channels=prepared.train.n_channels,
        )
        if model_type == "aae_mlp":
            latent_dim = int(model_cfg.get("latent_dim", 16))
            discriminator = LatentDiscriminator(
                latent_dim=latent_dim,
                hidden_dims=list(model_cfg.get("discriminator_hidden_dims", [128, 64])),
                dropout=float(model_cfg.get("dropout", 0.0)),
            )
            history = train_adversarial_autoencoder(
                autoencoder,
                discriminator,
                train_loader,
                val_train_loader,
                device=device,
                latent_dim=latent_dim,
                epochs=int(train_cfg.get("epochs", 50)),
                lr=float(train_cfg.get("lr", 1e-3)),
                lr_discriminator=float(train_cfg.get("lr_discriminator", 1e-3)),
                lr_adversarial=float(train_cfg.get("lr_adversarial", 5e-4)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
                loss_name=train_cfg.get("loss", "mse"),
                lambda_adv=float(model_cfg.get("lambda_adv", 0.1)),
                patience=int(train_cfg.get("patience", 10)),
                show_progress=bool(train_cfg.get("show_progress", True)),
            )
            torch.save({
                "autoencoder_state_dict": autoencoder.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "config": config,
                "feature_cols": prepared.feature_cols,
            }, output_dir / "model.pt")
        else:
            history = train_autoencoder(
                autoencoder,
                train_loader,
                val_train_loader,
                device=device,
                epochs=int(train_cfg.get("epochs", 50)),
                lr=float(train_cfg.get("lr", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
                loss_name=train_cfg.get("loss", "mse"),
                patience=int(train_cfg.get("patience", 10)),
                show_progress=bool(train_cfg.get("show_progress", True)),
            )
            torch.save({
                "autoencoder_state_dict": autoencoder.state_dict(),
                "config": config,
                "feature_cols": prepared.feature_cols,
            }, output_dir / "model.pt")
        save_json(history.as_dict(), output_dir / "history.json")
        val_scores, val_labels, val_run_ids, val_starts = score_autoencoder(
            autoencoder,
            val_eval_loader,
            device=device,
            error_mode=train_cfg.get("loss", "mse"),
        )
        test_scores, test_labels, test_run_ids, test_starts = score_autoencoder(
            autoencoder,
            test_eval_loader,
            device=device,
            error_mode=train_cfg.get("loss", "mse"),
        )
        threshold = select_threshold(
            val_labels,
            val_scores,
            method=eval_cfg.get("threshold", "best_f1"),
            fallback_percentile=float(eval_cfg.get("fallback_percentile", 99.0)),
        )
        _save_scores(output_dir / "scores_val.csv", val_scores, val_labels, val_run_ids, val_starts)
        _save_scores(output_dir / "scores_test.csv", test_scores, test_labels, test_run_ids, test_starts)
        val_metrics = all_metrics(val_labels, val_scores, threshold, val_run_ids, val_starts)
        test_metrics = all_metrics(test_labels, test_scores, threshold, test_run_ids, test_starts)
        result = {
            "model_type": model_type,
            "threshold": threshold,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "split_summary": prepared.split_summary,
        }
        save_json(result, output_dir / "metrics.json")
        return result

    # Classical baseline evaluation path.
    _save_scores(output_dir / "scores_val.csv", val_scores, prepared.val.y, prepared.val.run_ids, prepared.val.starts)
    _save_scores(output_dir / "scores_test.csv", test_scores, prepared.test.y, prepared.test.run_ids, prepared.test.starts)
    val_metrics = all_metrics(prepared.val.y, val_scores, threshold, prepared.val.run_ids, prepared.val.starts)
    test_metrics = all_metrics(prepared.test.y, test_scores, threshold, prepared.test.run_ids, prepared.test.starts)
    result = {
        "model_type": model_type,
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "split_summary": prepared.split_summary,
    }
    save_json(result, output_dir / "metrics.json")
    return result
