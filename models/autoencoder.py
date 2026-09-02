"""
Standard dense autoencoder for telemetry anomaly detection.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config import AE_DEFAULTS
from .base import BaseAnomalyDetector


class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], bottleneck: int):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, bottleneck))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        prev = bottleneck
        for h in reversed(hidden):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseAnomalyDetector):
    name = "autoencoder"

    def __init__(
        self,
        bottleneck: int = AE_DEFAULTS["bottleneck"],
        hidden_layers: list[int] | None = None,
        lr: float = AE_DEFAULTS["lr"],
        epochs: int = AE_DEFAULTS["epochs"],
        batch_size: int = AE_DEFAULTS["batch_size"],
        threshold_k: float = 2.0,
    ):
        self.bottleneck = bottleneck
        self.hidden_layers = hidden_layers or AE_DEFAULTS["hidden_layers"]
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
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X: np.ndarray, **kwargs) -> "AutoencoderDetector":
        X_flat = self._flatten(X).astype(np.float32)
        X_scaled = self.scaler_.fit_transform(X_flat).astype(np.float32)

        input_dim = X_scaled.shape[1]
        self.model_ = _Autoencoder(input_dim, self.hidden_layers, self.bottleneck)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        tensor = torch.tensor(X_scaled)
        loader = DataLoader(TensorDataset(tensor, tensor), batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        self.train_losses_ = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_in, batch_out in loader:
                pred = self.model_(batch_in)
                loss = criterion(pred, batch_out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.train_losses_.append(epoch_loss / len(loader))

        self.set_threshold(X, k=self.threshold_k)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X).astype(np.float32)
        X_scaled = self.scaler_.transform(X_flat).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(torch.tensor(X_scaled)).numpy()
        return np.mean((X_scaled - recon) ** 2, axis=1)

    def channel_errors(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X).astype(np.float32)
        X_scaled = self.scaler_.transform(X_flat).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(torch.tensor(X_scaled)).numpy()
        # Reshape back to (n, resample_points, n_channels)
        diff = (X_scaled - recon).reshape(-1, self.resample_points_, self.n_channels_)
        return np.mean(diff ** 2, axis=1)  # (n, n_channels)

    def get_params(self) -> dict:
        return {
            "bottleneck": self.bottleneck,
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
        }, path / "autoencoder.pt")

    def load(self, path: Path) -> "AutoencoderDetector":
        ckpt = torch.load(Path(path) / "autoencoder.pt", weights_only=False)
        self.scaler_ = ckpt["scaler"]
        self.threshold_ = ckpt["threshold"]
        self.n_channels_ = ckpt["n_channels"]
        self.resample_points_ = ckpt["resample_points"]
        self.train_losses_ = ckpt.get("train_losses", [])
        params = ckpt["params"]
        input_dim = self.resample_points_ * self.n_channels_
        self.model_ = _Autoencoder(input_dim, params["hidden_layers"], params["bottleneck"])
        self.model_.load_state_dict(ckpt["model_state"])
        return self
