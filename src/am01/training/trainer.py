from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from am01.models.aae import LatentDiscriminator, sample_standard_normal
from am01.training.losses import reconstruction_loss_fn


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    discriminator_loss: list[float] = field(default_factory=list)
    adversarial_loss: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "discriminator_loss": self.discriminator_loss,
            "adversarial_loss": self.adversarial_loss,
        }


def _batch_x(batch) -> torch.Tensor:
    return batch[0]


def evaluate_reconstruction_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_name: str = "mse",
) -> float:
    criterion = reconstruction_loss_fn(loss_name)
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            x = _batch_x(batch).to(device)
            x_hat = model(x)
            losses.append(float(criterion(x_hat, x).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loss_name: str = "mse",
    patience: int = 10,
    show_progress: bool = True,
) -> TrainingHistory:
    model.to(device)
    criterion = reconstruction_loss_fn(loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainingHistory()
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    iterator = range(epochs)
    if show_progress:
        iterator = tqdm(iterator, desc="train AE", leave=False)

    for epoch in iterator:
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            x = _batch_x(batch).to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = train_loss
        if val_loader is not None:
            val_loss = evaluate_reconstruction_loss(model, val_loader, device=device, loss_name=loss_name)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"train": train_loss, "val": val_loss})

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return history


def train_adversarial_autoencoder(
    autoencoder: nn.Module,
    discriminator: LatentDiscriminator,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    device: torch.device,
    latent_dim: int,
    epochs: int = 50,
    lr: float = 1e-3,
    lr_discriminator: float = 1e-3,
    lr_adversarial: float = 5e-4,
    weight_decay: float = 1e-5,
    loss_name: str = "mse",
    lambda_adv: float = 0.1,
    patience: int = 10,
    show_progress: bool = True,
) -> TrainingHistory:
    """Train an AAE with reconstruction, discriminator and adversarial phases.

    The reconstruction score remains directly comparable with a standard AE because
    encoder and decoder are trained with the same reconstruction objective, while the
    adversarial phase regularizes only the latent distribution.
    """
    if not hasattr(autoencoder, "encode") or not hasattr(autoencoder, "decode"):
        raise TypeError("AAE training requires an autoencoder exposing encode() and decode().")

    autoencoder.to(device)
    discriminator.to(device)
    criterion_rec = reconstruction_loss_fn(loss_name)
    criterion_bce = nn.BCEWithLogitsLoss()

    opt_rec = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=lr_discriminator, weight_decay=weight_decay)
    opt_adv = torch.optim.AdamW(autoencoder.encoder.parameters(), lr=lr_adversarial, weight_decay=weight_decay)

    history = TrainingHistory()
    best_state = deepcopy(autoencoder.state_dict())
    best_disc_state = deepcopy(discriminator.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    iterator: Iterable[int] = range(epochs)
    if show_progress:
        iterator = tqdm(iterator, desc="train AAE", leave=False)

    for epoch in iterator:
        autoencoder.train()
        discriminator.train()
        rec_losses: list[float] = []
        disc_losses: list[float] = []
        adv_losses: list[float] = []

        for batch in train_loader:
            x = _batch_x(batch).to(device)
            batch_size = x.shape[0]

            # 1) Reconstruction phase: update encoder and decoder.
            x_hat = autoencoder(x)
            loss_rec = criterion_rec(x_hat, x)
            opt_rec.zero_grad(set_to_none=True)
            loss_rec.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=5.0)
            opt_rec.step()
            rec_losses.append(float(loss_rec.item()))

            # 2) Discriminator phase: distinguish prior samples from encoded samples.
            with torch.no_grad():
                z_fake = autoencoder.encode(x)
            z_real = sample_standard_normal(batch_size, latent_dim, device=device)
            logits_real = discriminator(z_real)
            logits_fake = discriminator(z_fake.detach())
            targets_real = torch.ones_like(logits_real)
            targets_fake = torch.zeros_like(logits_fake)
            loss_disc = criterion_bce(logits_real, targets_real) + criterion_bce(logits_fake, targets_fake)
            opt_disc.zero_grad(set_to_none=True)
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            opt_disc.step()
            disc_losses.append(float(loss_disc.item()))

            # 3) Adversarial encoder phase: make encoded samples look like prior samples.
            z_fake = autoencoder.encode(x)
            logits_fake_for_encoder = discriminator(z_fake)
            loss_adv = criterion_bce(logits_fake_for_encoder, torch.ones_like(logits_fake_for_encoder))
            opt_adv.zero_grad(set_to_none=True)
            (lambda_adv * loss_adv).backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.encoder.parameters(), max_norm=5.0)
            opt_adv.step()
            adv_losses.append(float(loss_adv.item()))

        train_loss = float(np.mean(rec_losses)) if rec_losses else float("nan")
        val_loss = train_loss
        if val_loader is not None:
            val_loss = evaluate_reconstruction_loss(autoencoder, val_loader, device=device, loss_name=loss_name)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.discriminator_loss.append(float(np.mean(disc_losses)) if disc_losses else float("nan"))
        history.adversarial_loss.append(float(np.mean(adv_losses)) if adv_losses else float("nan"))

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({
                "rec": train_loss,
                "val": val_loss,
                "disc": history.discriminator_loss[-1],
                "adv": history.adversarial_loss[-1],
            })

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = deepcopy(autoencoder.state_dict())
            best_disc_state = deepcopy(discriminator.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    autoencoder.load_state_dict(best_state)
    discriminator.load_state_dict(best_disc_state)
    return history
