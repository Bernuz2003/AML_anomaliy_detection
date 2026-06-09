import torch

from am01.models.aae import LatentDiscriminator
from am01.models.autoencoders import Conv1dAutoencoder, MLPAutoencoder


def test_mlp_autoencoder_shapes():
    model = MLPAutoencoder(window_length=16, n_channels=4, latent_dim=3, hidden_dims=[12])
    x = torch.randn(5, 16, 4)
    z = model.encode(x)
    y = model(x)
    assert z.shape == (5, 3)
    assert y.shape == x.shape


def test_conv1d_autoencoder_shapes():
    model = Conv1dAutoencoder(window_length=16, n_channels=4, latent_dim=3, hidden_channels=8, kernel_size=3)
    x = torch.randn(5, 16, 4)
    z = model.encode(x)
    y = model(x)
    assert z.shape == (5, 3)
    assert y.shape == x.shape


def test_discriminator_logits_shape():
    disc = LatentDiscriminator(latent_dim=3, hidden_dims=[5])
    z = torch.randn(7, 3)
    logits = disc(z)
    assert logits.shape == (7,)
