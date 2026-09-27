"""Stage 3: align lagged weather and soil covariates with the flight tiles."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DISEASES, PipelineConfig

log = logging.getLogger(__name__)

# Epidemiological infection windows (literature-guided defaults; tune per site).
INFECTION_WINDOWS = {
    "stem_rust":   {"t_lo": 15.0, "t_hi": 35.0, "leaf_wetness_h": 4.0, "lag_days": 14},
    "stripe_rust": {"t_lo": 5.0,  "t_hi": 20.0, "leaf_wetness_h": 4.0, "lag_days": 14},
    "leaf_rust":   {"t_lo": 15.0, "t_hi": 30.0, "leaf_wetness_h": 4.0, "lag_days": 14},
    "septoria":    {"t_lo": 10.0, "t_hi": 25.0, "leaf_wetness_h": 8.0, "lag_days": 21},
    "fusarium":    {"t_lo": 20.0, "t_hi": 30.0, "leaf_wetness_h": 6.0, "lag_days": 10},
}

WEATHER_COLS = ["temp_mean", "rh_mean", "leaf_wetness_hours", "rain_mm"]


@dataclass
class AlignedCovariates:
    """Weather + soil evidence aligned to one flight."""

    weather_priors: dict     # disease -> prior probability in [0, 1]
    weather_seq: np.ndarray  # (N, W, 3) scaled daily [temp/30, rh/100, lwh/24] for the LSTM branch
    soil_score: float        # 0 = unfavourable, 1 = highly favourable to disease
    weather_meta: dict


def load_weather(csv_path, end_date, window_days):
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    missing = [c for c in WEATHER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"weather CSV missing columns: {missing}")
    end = pd.Timestamp(end_date)
    df = df[df["date"] <= end].tail(window_days)
    if df.empty:
        raise ValueError(f"no weather rows on/before {end.date()} in {csv_path}")
    return df.reset_index(drop=True)


def _temp_suitability(t, lo, hi):
    mid, span = (lo + hi) / 2.0, (hi - lo) / 2.0
    return float(np.exp(-(((t - mid) / span) ** 2)))


def infection_weather_prior(wdf, disease):
    """Transparent lagged-window prior: 0.45*temp suitability + 0.35*wetness + 0.20*rain."""
    cfg = INFECTION_WINDOWS[disease]
    w = wdf.tail(int(cfg["lag_days"]))
    temp_suit = float(np.mean([_temp_suitability(t, cfg["t_lo"], cfg["t_hi"]) for t in w["temp_mean"]]))
    wet_score = float(np.clip(w["leaf_wetness_hours"].mean() / cfg["leaf_wetness_h"], 0, 1.5) / 1.5)
    rain_score = float(1.0 - np.exp(-w["rain_mm"].sum() / 25.0))
    return float(np.clip(0.45 * temp_suit + 0.35 * wet_score + 0.20 * rain_score, 0.0, 1.0))


def soil_suitability_score(soil: dict) -> float:
    """Higher score = soil conditions favouring disease development (heuristics, tunable)."""
    pH = float(soil.get("ph", 6.8))
    pH_risk = float(1.0 - np.exp(-(((pH - 6.8) / 0.9) ** 2)))  # near-neutral pH -> low risk
    drainage = str(soil.get("drainage", "moderate")).lower()
    drainage_risk = {"very_poor": 1.0, "poor": 0.8, "moderate": 0.5,
                     "good": 0.25, "excessive": 0.15}.get(drainage, 0.5)
    residue = 1.0 if str(soil.get("previous_cereal", "false")).lower() in ("true", "yes", "1") else 0.0
    return float(np.clip(0.5 * pH_risk + 0.5 * drainage_risk + 0.05 * residue, 0.0, 1.0))


def align_weather_soil(flight, config: PipelineConfig) -> AlignedCovariates:
    """Join station weather (lagged infection windows) and soil records to the tiles."""
    n = len(flight.tiles)
    capture = flight.meta.get("capture_date") or config.flight_date
    if not capture:
        raise ValueError("capture_date missing: set flight_meta.json or PipelineConfig.flight_date")

    if config.weather_csv and Path(config.weather_csv).exists():
        wdf = load_weather(config.weather_csv, capture, config.weather_window_days)
        priors = {d: infection_weather_prior(wdf, d) for d in DISEASES}
        seq = np.stack([
            wdf["temp_mean"].to_numpy() / 30.0,
            wdf["rh_mean"].to_numpy() / 100.0,
            wdf["leaf_wetness_hours"].to_numpy() / 24.0,
        ], axis=1).astype(np.float32)
        if len(seq) < config.weather_window_days:
            pad = np.zeros((config.weather_window_days - len(seq), 3), dtype=np.float32)
            seq = np.vstack([pad, seq])
        weather_seq = np.repeat(seq[None, :, :], n, axis=0)
        wmeta = {"source": str(config.weather_csv), "rows": int(len(wdf)),
                 "end": str(pd.Timestamp(capture).date())}
    else:
        log.warning("No weather CSV found - using neutral priors (0.5) and zero sequences")
        priors = {d: 0.5 for d in DISEASES}
        weather_seq = np.zeros((n, config.weather_window_days, 3), dtype=np.float32)
        wmeta = {"source": None}

    if config.soil_csv and Path(config.soil_csv).exists():
        sdf = pd.read_csv(config.soil_csv)
        if "site_id" in sdf.columns:
            row = sdf[sdf["site_id"].astype(str) == str(flight.meta.get("site_id", config.site_id))]
            soil_row = (row if not row.empty else sdf).iloc[0].to_dict()
        else:
            soil_row = sdf.iloc[0].to_dict()
        soil_score = soil_suitability_score(soil_row)
    else:
        log.warning("No soil CSV found - using neutral soil score 0.5")
        soil_score = 0.5

    log.info("Aligned covariates: priors=%s soil=%.3f", {k: round(v, 2) for k, v in priors.items()}, soil_score)
    return AlignedCovariates(weather_priors=priors, weather_seq=weather_seq,
                             soil_score=soil_score, weather_meta=wmeta)
