"""
Abstract base class for all anomaly detection models.

Every model conforms to the same interface so they can be swapped
interchangeably in the evaluation pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAnomalyDetector(ABC):
    """Interface that every anomaly detection model must implement."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> "BaseAnomalyDetector":
        """
        Train on normal data only.

        Args:
            X: shape (n_samples, resample_points, n_channels)
        Returns:
            self
        """
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly score per sample. Higher = more anomalous.

        Args:
            X: shape (n_samples, resample_points, n_channels)
        Returns:
            scores: shape (n_samples,)
        """
        ...

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Binary prediction: 1 = anomalous, 0 = normal.
        If threshold is None, uses self.threshold_ (set during fit).
        """
        scores = self.score(X)
        t = threshold if threshold is not None else getattr(self, "threshold_", None)
        if t is None:
            raise ValueError("No threshold set. Either pass one or call fit() first.")
        return (scores > t).astype(int)

    def set_threshold(self, X_normal: np.ndarray, k: float = 2.0) -> float:
        """Set threshold as mean + k * std of scores on normal training data."""
        scores = self.score(X_normal)
        self.threshold_ = float(np.mean(scores) + k * np.std(scores))
        return self.threshold_

    @abstractmethod
    def channel_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Per-channel reconstruction error (or feature importance) per sample.

        Args:
            X: shape (n_samples, resample_points, n_channels)
        Returns:
            errors: shape (n_samples, n_channels)
        """
        ...

    def get_params(self) -> dict:
        """Return model hyperparameters as a dict."""
        return {}

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> "BaseAnomalyDetector":
        """Load model from disk."""
        ...
