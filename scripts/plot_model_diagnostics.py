#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from am01.pipeline import build_autoencoder


def _load_processed(path: Path):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    has_labels = bool(data["has_labels"])
    y = data["y"].astype(int) if has_labels else None
    feature_cols = [str(value) for value in data["feature_cols"]]
    return X, y, feature_cols


def _plot_reconstruction(
    x: np.ndarray,
    x_hat: np.ndarray,
    *,
    feature_cols: list[str],
    output_path: Path,
    title: str,
    max_features: int,
) -> None:
    n_features = min(max_features, x.shape[1])
    fig, axes = plt.subplots(n_features, 1, figsize=(10, max(3, 1.8 * n_features)), sharex=True)
    if n_features == 1:
        axes = [axes]
    t = np.arange(x.shape[0])
    for ax, idx in zip(axes, range(n_features)):
        ax.plot(t, x[:, idx], label="original", linewidth=1.2)
        ax.plot(t, x_hat[:, idx], label="reconstructed", linewidth=1.2, linestyle="--")
        ax.set_ylabel(feature_cols[idx] if idx < len(feature_cols) else f"feature_{idx}")
        ax.legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_latent(z: np.ndarray, y: np.ndarray | None, output_path: Path) -> None:
    if z.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=0).fit_transform(z)
    else:
        coords = np.column_stack([z[:, 0], np.zeros(z.shape[0])])
    fig, ax = plt.subplots(figsize=(6, 5))
    if y is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.7)
    else:
        normal = y == 0
        ax.scatter(coords[normal, 0], coords[normal, 1], s=10, alpha=0.65, label="normal")
        ax.scatter(coords[~normal, 0], coords[~normal, 1], s=10, alpha=0.65, label="anomalous")
        ax.legend()
    ax.set_title("Latent space")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AE/AAE reconstruction and latent diagnostics.")
    parser.add_argument("--run-dir", required=True, help="Directory produced by scripts/train.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="Output directory. Defaults to <run-dir>/figures.")
    parser.add_argument("--max-features", type=int, default=4)
    parser.add_argument("--latent-sample", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with (run_dir / "config_used.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_type = cfg.get("model", {}).get("type", "")
    if model_type not in {"ae_mlp", "aae_mlp", "ae_conv1d"}:
        raise ValueError(f"Model diagnostics are only available for autoencoders, got {model_type!r}.")

    X, y, feature_cols = _load_processed(run_dir / f"processed_{args.split}.npz")
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")
    model = build_autoencoder(cfg.get("model", {}), window_length=X.shape[1], n_channels=X.shape[2])
    model.load_state_dict(checkpoint["autoencoder_state_dict"])
    model.eval()

    out = Path(args.output) if args.output else run_dir / "figures"
    tensor = torch.from_numpy(X).float()
    rec_batches = []
    for start in range(0, len(tensor), args.batch_size):
        rec_batches.append(model(tensor[start:start + args.batch_size]).detach().cpu().numpy())
    rec = np.concatenate(rec_batches, axis=0)

    if y is not None and (y == 0).any():
        idx = int(np.where(y == 0)[0][0])
        _plot_reconstruction(
            X[idx],
            rec[idx],
            feature_cols=feature_cols,
            output_path=out / f"{args.split}_normal_reconstruction.png",
            title=f"{args.split} normal reconstruction",
            max_features=args.max_features,
        )
    if y is not None and (y == 1).any():
        idx = int(np.where(y == 1)[0][0])
        _plot_reconstruction(
            X[idx],
            rec[idx],
            feature_cols=feature_cols,
            output_path=out / f"{args.split}_anomalous_reconstruction.png",
            title=f"{args.split} anomalous reconstruction",
            max_features=args.max_features,
        )

    sample_n = min(args.latent_sample, len(X))
    sample_idx = np.linspace(0, len(X) - 1, sample_n, dtype=int) if sample_n else np.asarray([], dtype=int)
    if len(sample_idx):
        z_batches = []
        sample_tensor = tensor[sample_idx]
        for start in range(0, len(sample_tensor), args.batch_size):
            z_batches.append(model.encode(sample_tensor[start:start + args.batch_size]).detach().cpu().numpy())
        z = np.concatenate(z_batches, axis=0)
        y_sample = None if y is None else y[sample_idx]
        _plot_latent(z, y_sample, out / f"{args.split}_latent_space.png")

    print(f"Model diagnostic figures written to {out}")


if __name__ == "__main__":
    main()
