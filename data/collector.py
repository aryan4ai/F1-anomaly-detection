"""
Data collection pipeline.

Iterates over seasons, races, and drivers. Extracts telemetry from FastF1,
resamples each lap to a fixed number of points, and stores the result as
parquet files keyed by season/round/driver.
"""

import logging
import warnings
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CACHE_DIR, CHANNELS, PARQUET_DIR, RESAMPLE_POINTS

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


class TelemetryCollector:
    """Collects and preprocesses F1 telemetry data."""

    def __init__(
        self,
        seasons: list[int] | None = None,
        channels: list[str] | None = None,
        resample_points: int = RESAMPLE_POINTS,
        output_dir: Path = PARQUET_DIR,
    ):
        self.seasons = seasons or [2024]
        self.channels = channels or CHANNELS
        self.resample_points = resample_points
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE_DIR))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_all(self, force: bool = False) -> pd.DataFrame:
        """Collect telemetry for all seasons. Returns combined metadata DataFrame."""
        all_meta = []
        for season in self.seasons:
            meta = self._collect_season(season, force=force)
            all_meta.append(meta)
        if not all_meta:
            return pd.DataFrame()
        combined = pd.concat(all_meta, ignore_index=True)
        combined.to_parquet(self.output_dir / "metadata.parquet", index=False)
        logger.info("Collection complete. %d total laps across %d seasons.", len(combined), len(self.seasons))
        return combined

    def load_laps_matrix(
        self,
        season: int | None = None,
        circuit: str | None = None,
        driver: str | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Load saved parquet files and return (X, meta).
        X shape: (n_laps, resample_points, n_channels)
        meta: DataFrame with Season, Round, Circuit, Driver, LapNumber, Label columns.
        """
        meta_path = self.output_dir / "metadata.parquet"
        if not meta_path.exists():
            raise FileNotFoundError("Run collect_all() first.")
        meta = pd.read_parquet(meta_path)

        if season is not None:
            meta = meta[meta["Season"] == season]
        if circuit is not None:
            meta = meta[meta["Circuit"] == circuit]
        if driver is not None:
            meta = meta[meta["Driver"] == driver]
        meta = meta.reset_index(drop=True)

        X_list = []
        valid_idx = []
        for i, row in meta.iterrows():
            fpath = self.output_dir / f"{row['Season']}_{row['Round']}_{row['Driver']}.parquet"
            if not fpath.exists():
                continue
            lap_df = pd.read_parquet(fpath)
            lap_data = lap_df[lap_df["LapNumber"] == row["LapNumber"]]
            if lap_data.empty:
                continue
            arr = lap_data[self.channels].values
            if arr.shape[0] != self.resample_points:
                continue
            X_list.append(arr)
            valid_idx.append(i)

        if not X_list:
            return np.empty((0, self.resample_points, len(self.channels))), pd.DataFrame()

        X = np.stack(X_list).astype(np.float32)
        meta = meta.loc[valid_idx].reset_index(drop=True)
        return X, meta

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_season(self, season: int, force: bool = False) -> pd.DataFrame:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        race_rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"].tolist()
        all_meta = []

        for rnd in tqdm(race_rounds, desc=f"Season {season}"):
            outfile = self.output_dir / f"round_meta_{season}_{rnd}.parquet"
            if outfile.exists() and not force:
                all_meta.append(pd.read_parquet(outfile))
                continue
            try:
                meta = self._collect_race(season, rnd)
                if meta is not None and not meta.empty:
                    meta.to_parquet(outfile, index=False)
                    all_meta.append(meta)
            except Exception as e:
                logger.warning("Failed season=%d round=%d: %s", season, rnd, e)
        if not all_meta:
            return pd.DataFrame()
        return pd.concat(all_meta, ignore_index=True)

    def _collect_race(self, season: int, rnd: int) -> pd.DataFrame | None:
        try:
            session = fastf1.get_session(season, rnd, "R")
            session.load(telemetry=True, weather=False, messages=True)
        except Exception as e:
            logger.warning("Could not load session %d round %d: %s", season, rnd, e)
            return None

        circuit = session.event["EventName"]
        drivers = session.laps["Driver"].unique()
        all_rows = []
        all_meta = []

        for drv in drivers:
            drv_laps = session.laps.pick_drivers(drv).pick_accurate()
            if drv_laps.empty:
                continue
            driver_rows = []

            for _, lap in drv_laps.iterrows():
                try:
                    tel = lap.get_telemetry()
                except Exception:
                    continue
                if tel.empty or len(tel) < 10:
                    continue

                resampled = self._resample_lap(tel)
                if resampled is None:
                    continue

                resampled["LapNumber"] = int(lap["LapNumber"])
                driver_rows.append(resampled)
                all_meta.append({
                    "Season": season,
                    "Round": rnd,
                    "Circuit": circuit,
                    "Driver": drv,
                    "LapNumber": int(lap["LapNumber"]),
                    "LapTime_s": lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else np.nan,
                    "Compound": lap.get("Compound", "UNKNOWN"),
                    "TyreLife": int(lap.get("TyreLife", 0)),
                    "IsPit": bool(pd.notna(lap.get("PitInTime"))),
                })

            if driver_rows:
                drv_df = pd.concat(driver_rows, ignore_index=True)
                drv_df.to_parquet(
                    self.output_dir / f"{season}_{rnd}_{drv}.parquet", index=False
                )

        if not all_meta:
            return None
        return pd.DataFrame(all_meta)

    def _resample_lap(self, tel: pd.DataFrame) -> pd.DataFrame | None:
        """Resample telemetry to fixed number of equally spaced points."""
        available = [c for c in self.channels if c in tel.columns]
        if len(available) < len(self.channels):
            return None

        tel = tel[available].copy()
        tel = tel.dropna()
        if len(tel) < 20:
            return None

        n = len(tel)
        idx_original = np.linspace(0, 1, n)
        idx_new = np.linspace(0, 1, self.resample_points)

        resampled = {}
        for ch in available:
            resampled[ch] = np.interp(idx_new, idx_original, tel[ch].values)

        return pd.DataFrame(resampled)
