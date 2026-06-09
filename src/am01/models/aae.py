from __future__ import annotations

import torch
from torch import nn

from .autoencoders import make_mlp


class LatentDiscriminator(nn.Module):
    """Discriminator used by the adversarial autoencoder.

    It returns logits, not probabilities. Use BCEWithLogitsLoss during training.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = make_mlp([latent_dim, *list(hidden_dims), 1], dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def sample_standard_normal(batch_size: int, latent_dim: int, *, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, device=device)
