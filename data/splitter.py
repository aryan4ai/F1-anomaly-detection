"""
Train / validation / test splitting strategies.

Two modes:
  1. Temporal: train on earlier seasons, test on later.
  2. Leave-one-circuit-out: train on all circuits except one, test on that one.
"""

import numpy as np
import pandas as pd

from config import (
    CIRCUIT_FAMILIES,
    TEST_SEASON_ROUNDS,
    TRAIN_SEASONS,
    VAL_SEASON_ROUNDS,
)


class DataSplitter:
    """Splits telemetry data into train/val/test sets."""

    @staticmethod
    def temporal_split(
        X: np.ndarray,
        meta: pd.DataFrame,
    ) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
        """
        Split by season/round boundaries defined in config.

        Returns dict with keys 'train', 'val', 'test', each mapping to (X, meta).
        """
        train_mask = meta["Season"].isin(TRAIN_SEASONS)

        val_season, val_start, val_end = VAL_SEASON_ROUNDS
        val_mask = (
            (meta["Season"] == val_season)
            & (meta["Round"] >= val_start)
            & (meta["Round"] <= val_end)
        )

        test_season, test_start, test_end = TEST_SEASON_ROUNDS
        test_mask = (
            (meta["Season"] == test_season)
            & (meta["Round"] >= test_start)
            & (meta["Round"] <= test_end)
        )

        return {
            "train": (X[train_mask.values], meta[train_mask].reset_index(drop=True)),
            "val": (X[val_mask.values], meta[val_mask].reset_index(drop=True)),
            "test": (X[test_mask.values], meta[test_mask].reset_index(drop=True)),
        }

    @staticmethod
    def leave_one_circuit_out(
        X: np.ndarray,
        meta: pd.DataFrame,
        test_circuit: str,
    ) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
        """
        Train on all circuits except test_circuit, test on test_circuit.
        No separate validation set — use with cross-validation.
        """
        test_mask = meta["Circuit"].str.contains(test_circuit, case=False, na=False)
        train_mask = ~test_mask
        return {
            "train": (X[train_mask.values], meta[train_mask].reset_index(drop=True)),
            "test": (X[test_mask.values], meta[test_mask].reset_index(drop=True)),
        }

    @staticmethod
    def leave_one_family_out(
        X: np.ndarray,
        meta: pd.DataFrame,
        test_family: str,
    ) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
        """
        Train on all circuit families except one, test on that family.
        Families defined in config.CIRCUIT_FAMILIES.
        """
        test_circuits = CIRCUIT_FAMILIES.get(test_family, [])
        test_mask = meta["Circuit"].apply(
            lambda c: any(tc.lower() in c.lower() for tc in test_circuits)
        )
        train_mask = ~test_mask
        return {
            "train": (X[train_mask.values], meta[train_mask].reset_index(drop=True)),
            "test": (X[test_mask.values], meta[test_mask].reset_index(drop=True)),
        }

    @staticmethod
    def get_normal_only(
        X: np.ndarray,
        meta: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Filter to only 'normal' labeled laps (for training anomaly detectors)."""
        if "Label" not in meta.columns:
            return X, meta
        mask = meta["Label"] == "normal"
        return X[mask.values], meta[mask].reset_index(drop=True)
