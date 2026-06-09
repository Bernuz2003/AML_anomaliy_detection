from __future__ import annotations

import torch
from torch import nn


def reconstruction_loss_fn(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown reconstruction loss: {name}")


def per_window_reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor, *, mode: str = "mse") -> torch.Tensor:
    if x.shape != x_hat.shape:
        raise ValueError(f"x and x_hat must have the same shape, got {x.shape} and {x_hat.shape}")
    if mode == "mse":
        err = (x - x_hat) ** 2
    elif mode == "mae":
        err = torch.abs(x - x_hat)
    elif mode == "huber":
        err = torch.nn.functional.smooth_l1_loss(x_hat, x, reduction="none")
    else:
        raise ValueError(f"Unknown error mode: {mode}")
    return err.flatten(start_dim=1).mean(dim=1)
