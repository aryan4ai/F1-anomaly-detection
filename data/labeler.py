"""
Ground truth labeling.

Cross-references race control messages, retirements, pit stops, and safety
car periods against lap data to produce labels for evaluation.

Labels:
  - normal:       regular racing lap
  - pit:          pit in or pit out lap
  - pre_failure:  within N laps before a known mechanical DNF
  - sc:           lap under safety car or virtual safety car
  - formation:    formation / warm-up lap
"""

import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from config import CACHE_DIR, LABELS_DIR, PRE_FAILURE_WINDOW

logger = logging.getLogger(__name__)

# Known mechanical retirement reasons (partial string matches)
MECHANICAL_KEYWORDS = [
    "engine", "power unit", "gearbox", "hydraulic", "brake", "suspension",
    "electrical", "water leak", "oil leak", "overheating", "turbo",
    "exhaust", "fuel system", "driveshaft", "wheel nut", "fire",
    "mechanical", "technical", "retired", "stopped",
]


class GroundTruthLabeler:
    """Generates ground truth labels for telemetry laps."""

    def __init__(
        self,
        pre_failure_window: int = PRE_FAILURE_WINDOW,
        output_dir: Path = LABELS_DIR,
    ):
        self.pre_failure_window = pre_failure_window
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE_DIR))

    def label_season(self, season: int) -> pd.DataFrame:
        """Label all races in a season. Returns DataFrame with labels."""
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"].tolist()
        all_labels = []
        for rnd in rounds:
            try:
                labels = self._label_race(season, rnd)
                if labels is not None:
                    all_labels.append(labels)
            except Exception as e:
                logger.warning("Labeling failed for %d round %d: %s", season, rnd, e)
        if not all_labels:
            return pd.DataFrame()
        combined = pd.concat(all_labels, ignore_index=True)
        combined.to_parquet(self.output_dir / f"labels_{season}.parquet", index=False)
        return combined

    def label_all(self, seasons: list[int]) -> pd.DataFrame:
        """Label multiple seasons."""
        frames = []
        for s in seasons:
            cached = self.output_dir / f"labels_{s}.parquet"
            if cached.exists():
                frames.append(pd.read_parquet(cached))
            else:
                frames.append(self.label_season(s))
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(self.output_dir / "labels_all.parquet", index=False)
        return combined

    def merge_with_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Merge labels into the metadata DataFrame."""
        labels_path = self.output_dir / "labels_all.parquet"
        if not labels_path.exists():
            raise FileNotFoundError("Run label_all() first.")
        labels = pd.read_parquet(labels_path)
        merge_keys = ["Season", "Round", "Driver", "LapNumber"]
        merged = metadata.merge(labels[merge_keys + ["Label"]], on=merge_keys, how="left")
        merged["Label"] = merged["Label"].fillna("normal")
        return merged

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _label_race(self, season: int, rnd: int) -> pd.DataFrame | None:
        session = fastf1.get_session(season, rnd, "R")
        session.load(telemetry=False, weather=False, messages=True)

        laps = session.laps[["Driver", "LapNumber", "PitInTime", "PitOutTime",
                              "TrackStatus", "LapTime"]].copy()
        if laps.empty:
            return None

        laps["Season"] = season
        laps["Round"] = rnd
        laps["Label"] = "normal"

        # Mark pit laps
        pit_mask = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
        laps.loc[pit_mask, "Label"] = "pit"

        # Mark safety car laps (TrackStatus codes: 4=SC, 6=VSC, 7=VSC ending)
        if "TrackStatus" in laps.columns:
            sc_mask = laps["TrackStatus"].astype(str).isin(["4", "6", "7"])
            laps.loc[sc_mask & (laps["Label"] == "normal"), "Label"] = "sc"

        # Mark formation lap (lap 0 or 1 with abnormally slow time)
        if not laps.empty:
            first_lap_mask = laps["LapNumber"] == 1
            laps.loc[first_lap_mask & (laps["Label"] == "normal"), "Label"] = "formation"

        # Mark pre-failure laps
        dnf_drivers = self._find_mechanical_dnfs(session)
        for drv, last_lap in dnf_drivers.items():
            pre_fail_mask = (
                (laps["Driver"] == drv)
                & (laps["LapNumber"] > last_lap - self.pre_failure_window)
                & (laps["LapNumber"] <= last_lap)
                & (laps["Label"].isin(["normal", "sc"]))
            )
            laps.loc[pre_fail_mask, "Label"] = "pre_failure"

        return laps[["Season", "Round", "Driver", "LapNumber", "Label"]]

    def _find_mechanical_dnfs(self, session) -> dict[str, int]:
        """
        Returns {driver_abbrev: last_completed_lap} for drivers who retired
        due to mechanical issues.
        """
        results = {}
        try:
            race_results = session.results
        except Exception:
            return results

        if race_results is None or race_results.empty:
            return results

        for _, row in race_results.iterrows():
            status = str(row.get("Status", "")).lower()
            if status == "finished" or status.startswith("+"):
                continue
            if any(kw in status for kw in MECHANICAL_KEYWORDS):
                drv = row.get("Abbreviation", "")
                drv_laps = session.laps[session.laps["Driver"] == drv]
                if not drv_laps.empty:
                    results[drv] = int(drv_laps["LapNumber"].max())
        return results
