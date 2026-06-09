from __future__ import annotations

import torch
from torch import nn


def _activation(name: str = "relu") -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unknown activation: {name}")


def make_mlp(
    dims: list[int],
    *,
    activation: str = "relu",
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not is_last or final_activation:
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MLPAutoencoder(nn.Module):
    """Fully-connected autoencoder over flattened time windows.

    Input and output tensors use shape ``[batch, window_length, n_channels]``.
    """

    def __init__(
        self,
        *,
        window_length: int,
        n_channels: int,
        latent_dim: int = 16,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        self.window_length = int(window_length)
        self.n_channels = int(n_channels)
        self.input_dim = self.window_length * self.n_channels
        hidden = list(hidden_dims)
        self.encoder = make_mlp([self.input_dim, *hidden, latent_dim], dropout=dropout)
        self.decoder = make_mlp([latent_dim, *reversed(hidden), self.input_dim], dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.view(z.shape[0], self.window_length, self.n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class Conv1dAutoencoder(nn.Module):
    """A compact temporal autoencoder preserving the time dimension.

    The encoder uses Conv1D layers over ``[batch, channels, time]`` and maps the
    flattened temporal representation to a latent vector. The decoder mirrors this
    process and reconstructs the original ``[batch, time, channels]`` window.
    """

    def __init__(
        self,
        *,
        window_length: int,
        n_channels: int,
        latent_dim: int = 16,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so that padding preserves length.")
        self.window_length = int(window_length)
        self.n_channels = int(n_channels)
        self.latent_dim = int(latent_dim)
        pad = kernel_size // 2
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(n_channels, hidden_channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=pad),
            nn.ReLU(),
        )
        flat_dim = hidden_channels * window_length
        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)
        self.conv_decoder = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden_channels, n_channels, kernel_size, padding=pad),
        )
        self.hidden_channels = hidden_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.conv_encoder(h)
        return self.to_latent(h.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(z.shape[0], self.hidden_channels, self.window_length)
        out = self.conv_decoder(h)
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
