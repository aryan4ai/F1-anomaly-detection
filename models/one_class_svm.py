"""
One-Class SVM baseline for telemetry anomaly detection.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from .base import BaseAnomalyDetector


class OneClassSVMDetector(BaseAnomalyDetector):
    name = "one_class_svm"

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.05,
        pca_components: int = 50,
        threshold_k: float = 2.0,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.pca_components = pca_components
        self.threshold_k = threshold_k
        self.scaler_ = MinMaxScaler()
        self.pca_ = PCA(n_components=pca_components)
        self.model_ = None
        self.n_channels_ = None
        self.resample_points_ = None

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            self.n_channels_ = X.shape[2]
            self.resample_points_ = X.shape[1]
            return X.reshape(X.shape[0], -1)
        return X

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X_flat = self._flatten(X).astype(np.float64)
        if fit:
            X_scaled = self.scaler_.fit_transform(X_flat)
            return self.pca_.fit_transform(X_scaled)
        else:
            X_scaled = self.scaler_.transform(X_flat)
            return self.pca_.transform(X_scaled)

    def fit(self, X: np.ndarray, **kwargs) -> "OneClassSVMDetector":
        X_reduced = self._preprocess(X, fit=True)
        self.model_ = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)
        self.model_.fit(X_reduced)
        self.set_threshold(X, k=self.threshold_k)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X_reduced = self._preprocess(X, fit=False)
        return -self.model_.score_samples(X_reduced)

    def channel_errors(self, X: np.ndarray) -> np.ndarray:
        """
        One-Class SVM has no per-channel decomposition.
        Approximate by computing per-channel distance from normal boundary.
        """
        n = X.shape[0]
        n_ch = X.shape[2] if X.ndim == 3 else self.n_channels_
        errors = np.zeros((n, n_ch))
        if X.ndim == 3:
            for c in range(n_ch):
                ch_data = X[:, :, c]
                ch_scaled = MinMaxScaler().fit_transform(ch_data)
                pca_ch = PCA(n_components=min(10, ch_data.shape[1]))
                ch_reduced = pca_ch.fit_transform(ch_scaled)
                svm_ch = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)
                svm_ch.fit(ch_reduced)
                errors[:, c] = -svm_ch.score_samples(ch_reduced)
        return errors

    def get_params(self) -> dict:
        return {
            "kernel": self.kernel,
            "gamma": self.gamma,
            "nu": self.nu,
            "pca_components": self.pca_components,
            "threshold_k": self.threshold_k,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "ocsvm.pkl", "wb") as f:
            pickle.dump({
                "model": self.model_,
                "scaler": self.scaler_,
                "pca": self.pca_,
                "params": self.get_params(),
                "threshold": getattr(self, "threshold_", None),
                "n_channels": self.n_channels_,
                "resample_points": self.resample_points_,
            }, f)

    def load(self, path: Path) -> "OneClassSVMDetector":
        with open(Path(path) / "ocsvm.pkl", "rb") as f:
            ckpt = pickle.load(f)
        self.model_ = ckpt["model"]
        self.scaler_ = ckpt["scaler"]
        self.pca_ = ckpt["pca"]
        self.threshold_ = ckpt["threshold"]
        self.n_channels_ = ckpt["n_channels"]
        self.resample_points_ = ckpt["resample_points"]
        return self
