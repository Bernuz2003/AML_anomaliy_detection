from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from am01.training.losses import per_window_reconstruction_error


@torch.inference_mode()
def score_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    error_mode: str = "mse",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    run_ids: list[str] = []
    starts: list[int] = []
    for x, y, batch_run_ids, batch_starts in loader:
        x = x.to(device)
        x_hat = model(x)
        batch_scores = per_window_reconstruction_error(x, x_hat, mode=error_mode)
        scores.append(batch_scores.detach().cpu().numpy())
        y_np = y.detach().cpu().numpy()
        labels.append(y_np)
        run_ids.extend([str(r) for r in batch_run_ids])
        starts.extend([int(s) for s in batch_starts])
    score_arr = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    label_arr = np.concatenate(labels) if labels else None
    if label_arr is not None and np.all(label_arr < 0):
        label_arr = None
    return score_arr, label_arr, np.asarray(run_ids), np.asarray(starts, dtype=np.int64)
