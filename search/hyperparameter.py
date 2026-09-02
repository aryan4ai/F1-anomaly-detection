"""
Hyperparameter search using Optuna.

Searches over model hyperparameters and threshold multiplier.
Logs every trial's metrics to a CSV for later analysis.
"""

import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from config import RESULTS_DIR
from models.autoencoder import AutoencoderDetector
from models.vae import VAEDetector
from models.lstm_ae import LSTMAutoencoderDetector

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterSearch:
    """Optuna-based hyperparameter search for anomaly detection models."""

    def __init__(self, output_dir: Path = RESULTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search_autoencoder(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        meta_val: pd.DataFrame,
        n_trials: int = 50,
    ) -> tuple[dict, pd.DataFrame]:
        """
        Search over autoencoder hyperparameters.
        Returns (best_params, trial_log).
        """
        y_val = (meta_val["Label"] == "pre_failure").astype(int).values

        trial_log = []

        def objective(trial):
            bottleneck = trial.suggest_categorical("bottleneck", [8, 16, 32, 64])
            n_hidden = trial.suggest_int("n_hidden_layers", 1, 3)
            hidden = []
            prev = X_train.shape[1] * X_train.shape[2]
            for i in range(n_hidden):
                h = trial.suggest_categorical(f"hidden_{i}", [64, 128, 256, 512])
                if h < bottleneck:
                    h = bottleneck * 2
                hidden.append(h)
                prev = h
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            epochs = trial.suggest_categorical("epochs", [50, 100, 150])
            threshold_k = trial.suggest_float("threshold_k", 1.0, 4.0)

            model = AutoencoderDetector(
                bottleneck=bottleneck,
                hidden_layers=hidden,
                lr=lr,
                epochs=epochs,
                threshold_k=threshold_k,
            )
            model.fit(X_train)
            preds = model.predict(X_val)
            score = float(f1_score(y_val, preds, zero_division=0))

            trial_log.append({
                "trial": trial.number,
                "bottleneck": bottleneck,
                "hidden_layers": str(hidden),
                "lr": lr,
                "epochs": epochs,
                "threshold_k": threshold_k,
                "f1": score,
            })
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        log_df = pd.DataFrame(trial_log)
        log_df.to_csv(self.output_dir / "hp_search_autoencoder.csv", index=False)
        return study.best_params, log_df

    def search_vae(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        meta_val: pd.DataFrame,
        n_trials: int = 50,
    ) -> tuple[dict, pd.DataFrame]:
        y_val = (meta_val["Label"] == "pre_failure").astype(int).values
        trial_log = []

        def objective(trial):
            latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64])
            hidden = [
                trial.suggest_categorical("hidden_0", [128, 256, 512]),
                trial.suggest_categorical("hidden_1", [32, 64, 128]),
            ]
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            epochs = trial.suggest_categorical("epochs", [50, 100, 150])
            threshold_k = trial.suggest_float("threshold_k", 1.0, 4.0)

            model = VAEDetector(
                latent_dim=latent_dim, hidden_layers=hidden, lr=lr,
                epochs=epochs, threshold_k=threshold_k,
            )
            model.fit(X_train)
            preds = model.predict(X_val)
            score = float(f1_score(y_val, preds, zero_division=0))

            trial_log.append({
                "trial": trial.number, "latent_dim": latent_dim,
                "hidden_layers": str(hidden), "lr": lr,
                "epochs": epochs, "threshold_k": threshold_k, "f1": score,
            })
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        log_df = pd.DataFrame(trial_log)
        log_df.to_csv(self.output_dir / "hp_search_vae.csv", index=False)
        return study.best_params, log_df

    def search_lstm(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        meta_val: pd.DataFrame,
        n_trials: int = 30,
    ) -> tuple[dict, pd.DataFrame]:
        y_val = (meta_val["Label"] == "pre_failure").astype(int).values
        trial_log = []

        def objective(trial):
            hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            epochs = trial.suggest_categorical("epochs", [50, 100])
            threshold_k = trial.suggest_float("threshold_k", 1.0, 4.0)

            model = LSTMAutoencoderDetector(
                hidden_size=hidden_size, num_layers=num_layers,
                lr=lr, epochs=epochs, threshold_k=threshold_k,
            )
            model.fit(X_train)
            preds = model.predict(X_val)
            score = float(f1_score(y_val, preds, zero_division=0))

            trial_log.append({
                "trial": trial.number, "hidden_size": hidden_size,
                "num_layers": num_layers, "lr": lr,
                "epochs": epochs, "threshold_k": threshold_k, "f1": score,
            })
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        log_df = pd.DataFrame(trial_log)
        log_df.to_csv(self.output_dir / "hp_search_lstm.csv", index=False)
        return study.best_params, log_df
