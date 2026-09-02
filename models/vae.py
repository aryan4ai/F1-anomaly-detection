"""
Variational Autoencoder for telemetry anomaly detection.

Uses reconstruction error + KL divergence as the anomaly score.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config import VAE_DEFAULTS
from .base import BaseAnomalyDetector


class _VAE(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], latent_dim: int):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def _vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


class VAEDetector(BaseAnomalyDetector):
    name = "vae"

    def __init__(
        self,
        latent_dim: int = VAE_DEFAULTS["latent_dim"],
        hidden_layers: list[int] | None = None,
        lr: float = VAE_DEFAULTS["lr"],
        epochs: int = VAE_DEFAULTS["epochs"],
        batch_size: int = VAE_DEFAULTS["batch_size"],
        threshold_k: float = 2.0,
    ):
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers or VAE_DEFAULTS["hidden_layers"]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_k = threshold_k
        self.scaler_ = MinMaxScaler()
        self.model_ = None
        self.n_channels_ = None
        self.resample_points_ = None
        self.train_losses_ = []

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        self.n_channels_ = X.shape[2] if X.ndim == 3 else self.n_channels_
        self.resample_points_ = X.shape[1] if X.ndim == 3 else self.resample_points_
        return X.reshape(X.shape[0], -1) if X.ndim == 3 else X

    def fit(self, X: np.ndarray, **kwargs) -> "VAEDetector":
        X_flat = self._flatten(X).astype(np.float32)
        X_scaled = self.scaler_.fit_transform(X_flat).astype(np.float32)

        input_dim = X_scaled.shape[1]
        self.model_ = _VAE(input_dim, self.hidden_layers, self.latent_dim)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        tensor = torch.tensor(X_scaled)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        self.train_losses_ = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                recon, mu, logvar = self.model_(batch)
                loss = _vae_loss(recon, batch, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.train_losses_.append(epoch_loss / len(tensor))

        self.set_threshold(X, k=self.threshold_k)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X).astype(np.float32)
        X_scaled = self.scaler_.transform(X_flat).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            recon, mu, logvar = self.model_(torch.tensor(X_scaled))
            recon = recon.numpy()
        return np.mean((X_scaled - recon) ** 2, axis=1)

    def channel_errors(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X).astype(np.float32)
        X_scaled = self.scaler_.transform(X_flat).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            recon, _, _ = self.model_(torch.tensor(X_scaled))
            recon = recon.numpy()
        diff = (X_scaled - recon).reshape(-1, self.resample_points_, self.n_channels_)
        return np.mean(diff ** 2, axis=1)

    def get_params(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "hidden_layers": self.hidden_layers,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "threshold_k": self.threshold_k,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model_.state_dict(),
            "scaler": self.scaler_,
            "params": self.get_params(),
            "threshold": getattr(self, "threshold_", None),
            "n_channels": self.n_channels_,
            "resample_points": self.resample_points_,
            "train_losses": self.train_losses_,
        }, path / "vae.pt")

    def load(self, path: Path) -> "VAEDetector":
        ckpt = torch.load(Path(path) / "vae.pt", weights_only=False)
        self.scaler_ = ckpt["scaler"]
        self.threshold_ = ckpt["threshold"]
        self.n_channels_ = ckpt["n_channels"]
        self.resample_points_ = ckpt["resample_points"]
        self.train_losses_ = ckpt.get("train_losses", [])
        params = ckpt["params"]
        input_dim = self.resample_points_ * self.n_channels_
        self.model_ = _VAE(input_dim, params["hidden_layers"], params["latent_dim"])
        self.model_.load_state_dict(ckpt["model_state"])
        return self
