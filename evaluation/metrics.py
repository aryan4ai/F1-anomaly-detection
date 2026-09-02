"""
Evaluation framework.

Takes any model's predictions and ground truth labels, and computes:
  - Precision, Recall, F1
  - ROC-AUC, PR-AUC
  - Confusion matrix
  - Average warning lead time (laps before known failure that model first flags)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import CHANNELS
from models.base import BaseAnomalyDetector

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates anomaly detection models against ground truth labels."""

    def __init__(self, channels: list[str] | None = None):
        self.channels = channels or CHANNELS

    def evaluate(
        self,
        model: BaseAnomalyDetector,
        X: np.ndarray,
        meta: pd.DataFrame,
        threshold: float | None = None,
    ) -> dict:
        """
        Full evaluation of a model on labeled data.

        Args:
            model: trained anomaly detector
            X: shape (n, resample_points, n_channels)
            meta: DataFrame with 'Label' column
            threshold: override model's threshold (optional)

        Returns:
            dict with all metrics
        """
        scores = model.score(X)
        preds = model.predict(X, threshold=threshold)

        # Binary ground truth: pre_failure=1, everything else=0
        y_true = (meta["Label"] == "pre_failure").astype(int).values

        results = {
            "model": model.name,
            "params": model.get_params(),
            "threshold": threshold or getattr(model, "threshold_", None),
            "n_samples": len(X),
            "n_positive": int(y_true.sum()),
            "n_predicted_positive": int(preds.sum()),
        }

        # Core metrics
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            results["precision"] = float(precision_score(y_true, preds, zero_division=0))
            results["recall"] = float(recall_score(y_true, preds, zero_division=0))
            results["f1"] = float(f1_score(y_true, preds, zero_division=0))
            results["roc_auc"] = float(roc_auc_score(y_true, scores))
            results["pr_auc"] = float(average_precision_score(y_true, scores))
            results["confusion_matrix"] = confusion_matrix(y_true, preds).tolist()

            # ROC and PR curves (for plotting)
            fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
            prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, scores)
            results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            results["pr_curve"] = {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()}
        else:
            results["precision"] = 0.0
            results["recall"] = 0.0
            results["f1"] = 0.0
            results["roc_auc"] = None
            results["pr_auc"] = None
            results["confusion_matrix"] = None
            results["roc_curve"] = None
            results["pr_curve"] = None
            logger.warning("Cannot compute ROC/PR: only one class present in labels.")

        # Warning lead time
        results["avg_warning_lead_time"] = self._warning_lead_time(preds, meta)

        return results

    def compare_models(
        self,
        models: list[BaseAnomalyDetector],
        X: np.ndarray,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """Evaluate multiple models and return a comparison DataFrame."""
        rows = []
        for m in models:
            try:
                r = self.evaluate(m, X, meta)
                rows.append({
                    "Model": r["model"],
                    "Precision": r["precision"],
                    "Recall": r["recall"],
                    "F1": r["f1"],
                    "ROC-AUC": r["roc_auc"],
                    "PR-AUC": r["pr_auc"],
                    "Predicted +": r["n_predicted_positive"],
                    "Actual +": r["n_positive"],
                    "Lead Time": r["avg_warning_lead_time"],
                })
            except Exception as e:
                logger.error("Failed to evaluate %s: %s", m.name, e)
        return pd.DataFrame(rows)

    def threshold_sensitivity(
        self,
        model: BaseAnomalyDetector,
        X: np.ndarray,
        meta: pd.DataFrame,
        k_values: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Evaluate model at multiple threshold multipliers.
        Returns DataFrame with metrics per threshold.
        """
        if k_values is None:
            k_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

        scores = model.score(X)
        y_true = (meta["Label"] == "pre_failure").astype(int).values

        rows = []
        for k in k_values:
            t = float(np.mean(scores) + k * np.std(scores))
            preds = (scores > t).astype(int)
            rows.append({
                "k": k,
                "threshold": t,
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1": float(f1_score(y_true, preds, zero_division=0)),
                "n_flagged": int(preds.sum()),
            })
        return pd.DataFrame(rows)

    def _warning_lead_time(self, preds: np.ndarray, meta: pd.DataFrame) -> float | None:
        """
        For each known mechanical DNF, find the earliest flagged lap before it.
        Return average number of laps of advance warning.
        """
        if "Label" not in meta.columns:
            return None

        pre_failure_groups = meta[meta["Label"] == "pre_failure"].groupby(
            ["Season", "Round", "Driver"]
        )
        if len(pre_failure_groups) == 0:
            return None

        lead_times = []
        for (season, rnd, drv), group in pre_failure_groups:
            failure_lap = group["LapNumber"].max()
            group_idx = group.index.tolist()
            flagged = [i for i in group_idx if preds[i] == 1]
            if flagged:
                first_flag_lap = meta.loc[min(flagged), "LapNumber"]
                lead_times.append(failure_lap - first_flag_lap)

        if not lead_times:
            return 0.0
        return float(np.mean(lead_times))
