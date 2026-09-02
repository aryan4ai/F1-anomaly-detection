"""
LSTM-based autoencoder for telemetry anomaly detection.

Treats each lap as a sequence of time steps rather than a flattened vector.
This preserves temporal structure — speed dropping then recovering means
something different than speed staying low throughout.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config import LSTM_DEFAULTS
from .base import BaseAnomalyDetector


class _LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]  # Last layer hidden state


class _LSTMDecoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, z):
        # Repeat encoded vector across sequence length
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z)
        return self.fc(out)


class _LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, seq_len: int):
        super().__init__()
        self.encoder = _LSTMEncoder(n_features, hidden_size, num_layers)
        self.decoder = _LSTMDecoder(n_features, hidden_size, num_layers, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    name = "lstm_autoencoder"

    def __init__(
        self,
        hidden_size: int = LSTM_DEFAULTS["hidden_size"],
        num_layers: int = LSTM_DEFAULTS["num_layers"],
        lr: float = LSTM_DEFAULTS["lr"],
        epochs: int = LSTM_DEFAULTS["epochs"],
        batch_size: int = LSTM_DEFAULTS["batch_size"],
        threshold_k: float = 2.0,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_k = threshold_k
        self.scaler_ = MinMaxScaler()
        self.model_ = None
        self.n_channels_ = None
        self.resample_points_ = None
        self.train_losses_ = []

    def _scale_3d(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale 3D array (n, seq, features) using 2D scaler."""
        self.n_channels_ = X.shape[2]
        self.resample_points_ = X.shape[1]
        n, s, f = X.shape
        flat = X.reshape(-1, f)
        if fit:
            scaled = self.scaler_.fit_transform(flat)
        else:
            scaled = self.scaler_.transform(flat)
        return scaled.reshape(n, s, f).astype(np.float32)

    def fit(self, X: np.ndarray, **kwargs) -> "LSTMAutoencoderDetector":
        X_scaled = self._scale_3d(X, fit=True)

        self.model_ = _LSTMAE(self.n_channels_, self.hidden_size, self.num_layers, self.resample_points_)
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
        X_scaled = self._scale_3d(X, fit=False)
        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(torch.tensor(X_scaled)).numpy()
        return np.mean((X_scaled - recon) ** 2, axis=(1, 2))

    def channel_errors(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scale_3d(X, fit=False)
        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(torch.tensor(X_scaled)).numpy()
        return np.mean((X_scaled - recon) ** 2, axis=1)  # (n, n_channels)

    def get_params(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
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
        }, path / "lstm_ae.pt")

    def load(self, path: Path) -> "LSTMAutoencoderDetector":
        ckpt = torch.load(Path(path) / "lstm_ae.pt", weights_only=False)
        self.scaler_ = ckpt["scaler"]
        self.threshold_ = ckpt["threshold"]
        self.n_channels_ = ckpt["n_channels"]
        self.resample_points_ = ckpt["resample_points"]
        self.train_losses_ = ckpt.get("train_losses", [])
        params = ckpt["params"]
        self.model_ = _LSTMAE(self.n_channels_, params["hidden_size"], params["num_layers"], self.resample_points_)
        self.model_.load_state_dict(ckpt["model_state"])
        return self
