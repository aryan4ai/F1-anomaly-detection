"""
Isolation Forest baseline for telemetry anomaly detection.

Scikit-learn's IsolationForest wrapped to conform to the project's
BaseAnomalyDetector interface.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from .base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
        threshold_k: float = 2.0,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.threshold_k = threshold_k
        self.scaler_ = MinMaxScaler()
        self.model_ = None
        self.n_channels_ = None
        self.resample_points_ = None

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            self.n_channels_ = X.shape[2]
            self.resample_points_ = X.shape[1]
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X: np.ndarray, **kwargs) -> "IsolationForestDetector":
        X_flat = self._flatten(X).astype(np.float32)
        X_scaled = self.scaler_.fit_transform(X_flat)

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model_.fit(X_scaled)
        self.set_threshold(X, k=self.threshold_k)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X).astype(np.float32)
        X_scaled = self.scaler_.transform(X_flat)
        # Isolation forest returns negative scores; more negative = more anomalous.
        # Negate so higher = more anomalous, consistent with other models.
        return -self.model_.score_samples(X_scaled)

    def channel_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Isolation Forest has no per-channel reconstruction.
        Approximate by computing per-channel isolation scores: train a
        separate small forest per channel and return those scores.
        """
        n = X.shape[0]
        n_ch = X.shape[2] if X.ndim == 3 else self.n_channels_
        rp = X.shape[1] if X.ndim == 3 else self.resample_points_
        if X.ndim == 3:
            errors = np.zeros((n, n_ch))
            for c in range(n_ch):
                ch_data = X[:, :, c]  # (n, resample_points)
                ch_scaled = MinMaxScaler().fit_transform(ch_data)
                iso = IsolationForest(n_estimators=50, contamination=self.contamination,
                                      random_state=self.random_state)
                iso.fit(ch_scaled)
                errors[:, c] = -iso.score_samples(ch_scaled)
            return errors
        return np.zeros((n, self.n_channels_))

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "threshold_k": self.threshold_k,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "isolation_forest.pkl", "wb") as f:
            pickle.dump({
                "model": self.model_,
                "scaler": self.scaler_,
                "params": self.get_params(),
                "threshold": getattr(self, "threshold_", None),
                "n_channels": self.n_channels_,
                "resample_points": self.resample_points_,
            }, f)

    def load(self, path: Path) -> "IsolationForestDetector":
        with open(Path(path) / "isolation_forest.pkl", "rb") as f:
            ckpt = pickle.load(f)
        self.model_ = ckpt["model"]
        self.scaler_ = ckpt["scaler"]
        self.threshold_ = ckpt["threshold"]
        self.n_channels_ = ckpt["n_channels"]
        self.resample_points_ = ckpt["resample_points"]
        return self
