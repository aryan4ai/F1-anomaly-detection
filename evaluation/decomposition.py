"""
Per-channel anomaly decomposition.

When a model flags a lap as anomalous, this module classifies the failure
mode hypothesis based on which channels have elevated reconstruction error.

Channel patterns → failure mode mapping:
  - Speed only:           traffic / track limits / off-track excursion
  - RPM only:             engine issue
  - Throttle only:        driver error / ERS deployment anomaly
  - Brake only:           brake system issue / unusual braking profile
  - Speed + RPM:          drivetrain / power delivery failure
  - Speed + Brake:        brake failure under deceleration
  - RPM + Throttle:       engine response anomaly
  - Multiple channels:    major mechanical failure
  - nGear only:           gearbox issue
  - All elevated:         sensor fault or total system failure
"""

import numpy as np
import pandas as pd

from config import CHANNELS
from models.base import BaseAnomalyDetector

# Rule-based failure mode classification
FAILURE_PATTERNS = {
    frozenset(["Speed"]): "traffic_or_off_track",
    frozenset(["RPM"]): "engine_anomaly",
    frozenset(["Throttle"]): "driver_error_or_ers",
    frozenset(["Brake"]): "brake_system",
    frozenset(["nGear"]): "gearbox",
    frozenset(["Speed", "RPM"]): "drivetrain",
    frozenset(["Speed", "Brake"]): "brake_failure_decel",
    frozenset(["RPM", "Throttle"]): "engine_response",
    frozenset(["Speed", "RPM", "Throttle"]): "power_unit",
    frozenset(["Speed", "RPM", "nGear"]): "drivetrain_gearbox",
}


class AnomalyDecomposer:
    """Classifies anomalous laps by failure mode based on channel error patterns."""

    def __init__(self, channels: list[str] | None = None):
        self.channels = channels or CHANNELS

    def decompose(
        self,
        model: BaseAnomalyDetector,
        X: np.ndarray,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For every lap the model flags as anomalous, determine which
        channels are responsible and classify the failure mode.

        Returns DataFrame with columns:
          Season, Round, Driver, LapNumber, Label,
          anomaly_score, dominant_channel, elevated_channels, failure_mode,
          + per-channel error columns
        """
        scores = model.score(X)
        preds = model.predict(X)
        channel_errs = model.channel_errors(X)

        # Only analyze flagged laps
        flagged_mask = preds == 1
        if not flagged_mask.any():
            return pd.DataFrame()

        flagged_meta = meta[flagged_mask].copy().reset_index(drop=True)
        flagged_scores = scores[flagged_mask]
        flagged_ch_errs = channel_errs[flagged_mask]

        # Determine which channels are elevated (> mean + 1.5*std per channel)
        ch_means = channel_errs.mean(axis=0)
        ch_stds = channel_errs.std(axis=0)
        ch_thresholds = ch_means + 1.5 * ch_stds

        results = []
        for i in range(len(flagged_meta)):
            row = flagged_meta.iloc[i].to_dict()
            row["anomaly_score"] = float(flagged_scores[i])

            errs = flagged_ch_errs[i]
            for c, ch_name in enumerate(self.channels):
                row[f"error_{ch_name}"] = float(errs[c])

            # Dominant channel = highest error
            dominant_idx = int(np.argmax(errs))
            row["dominant_channel"] = self.channels[dominant_idx]

            # Elevated channels = those above threshold
            elevated = [self.channels[c] for c in range(len(self.channels))
                        if errs[c] > ch_thresholds[c]]
            row["elevated_channels"] = elevated if elevated else [self.channels[dominant_idx]]

            # Failure mode classification
            elevated_set = frozenset(row["elevated_channels"])
            row["failure_mode"] = FAILURE_PATTERNS.get(elevated_set, "complex_multi_channel")

            results.append(row)

        return pd.DataFrame(results)

    def failure_mode_summary(self, decomposition: pd.DataFrame) -> pd.DataFrame:
        """Summarize failure mode counts."""
        if decomposition.empty:
            return pd.DataFrame(columns=["failure_mode", "count", "pct"])
        counts = decomposition["failure_mode"].value_counts().reset_index()
        counts.columns = ["failure_mode", "count"]
        counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
        return counts

    def channel_importance(self, decomposition: pd.DataFrame) -> pd.DataFrame:
        """Rank channels by how often they are the dominant anomaly source."""
        if decomposition.empty:
            return pd.DataFrame(columns=["channel", "dominant_count", "elevated_count"])
        dominant_counts = decomposition["dominant_channel"].value_counts()
        elevated_counts = {}
        for channels_list in decomposition["elevated_channels"]:
            for ch in channels_list:
                elevated_counts[ch] = elevated_counts.get(ch, 0) + 1
        rows = []
        for ch in self.channels:
            rows.append({
                "channel": ch,
                "dominant_count": int(dominant_counts.get(ch, 0)),
                "elevated_count": int(elevated_counts.get(ch, 0)),
            })
        return pd.DataFrame(rows).sort_values("dominant_count", ascending=False)
