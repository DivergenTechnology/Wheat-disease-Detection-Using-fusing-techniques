"""Stage 4: SSCNN inference with MC-Dropout uncertainty and attention extraction."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DISEASES, PipelineConfig

log = logging.getLogger(__name__)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def run_sscnn_inference(model, flight, aligned, config: PipelineConfig) -> pd.DataFrame:
    """Run MC-Dropout inference over all tiles and assemble the prediction table."""
    mean_p, std_p, attention = model.predict_with_uncertainty(
        flight.tiles, flight.indices, aligned.weather_seq, n_mc=config.mc_passes
    )

    # Temperature calibration (trust layer): sharpen/flatten probabilities.
    if config.calibration == "temperature" and getattr(model, "temperature", 1.0) not in (None, 1.0):
        mean_p = _sigmoid(_logit(mean_p) / float(model.temperature))
    mean_p = np.clip(mean_p, 0.0, 1.0)

    rows = []
    for i in range(len(flight.tiles)):
        row = {
            "tile_id": f"T{i:04d}",
            "x": float(flight.centroids[i, 0]),
            "y": float(flight.centroids[i, 1]),
            "attention": float(attention[i]),
        }
        for j, d in enumerate(DISEASES):
            row[f"p_{d}"] = float(mean_p[i, j])
            row[f"u_{d}"] = float(std_p[i, j])
        probs = np.array([row[f"p_{d}"] for d in DISEASES])
        row["max_prob"] = float(probs.max())
        row["dominant_disease"] = DISEASES[int(probs.argmax())]
        row["mean_uncertainty"] = float(std_p[i].mean())
        # Open-set policy: [lower, threshold) is the ambiguous zone -> expert review;
        # below `lower` the model is confidently healthy; above `threshold` it makes
        # a confident disease call.
        row["in_distribution"] = bool(
            row["max_prob"] >= config.open_set_threshold or row["max_prob"] < config.open_set_lower
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("Inference done: %d tiles, mean uncertainty %.4f", len(df), df["mean_uncertainty"].mean())
    return df
