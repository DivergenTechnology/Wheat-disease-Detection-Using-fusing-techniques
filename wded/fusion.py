"""Stage 5: evidential fusion of model output with lagged-weather and soil evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DISEASES, RISK_TIERS, PipelineConfig

TIER_ORDER = ["low", "moderate", "high", "critical"]
_TIER_BOUNDS = {"low": 0.25, "moderate": 0.50, "high": 0.75}


def tier_of(risk: float) -> str:
    """Map a fused risk in [0, 1] to its tier (RISK_TIERS ordered high -> low)."""
    for name, thr in RISK_TIERS:
        if risk >= thr:
            return name
    return "low"


def escalate_tier(tier: str, risk: float, margin: float) -> str:
    """Asymmetric-cost escalation: risk just below a boundary is treated as the higher tier.

    Missing an emerging infection costs ~5x a false alarm (config.cost_ratio_fn_fp),
    so a moderate risk of 0.48 is handled as high, not moderate.
    """
    bound = _TIER_BOUNDS.get(tier)
    if bound is not None and risk >= bound - margin:
        return TIER_ORDER[min(TIER_ORDER.index(tier) + 1, len(TIER_ORDER) - 1)]
    return tier


def fuse_decision(preds: pd.DataFrame, aligned, config: PipelineConfig) -> pd.DataFrame:
    """risk_d = 0.5*model + 0.3*weather prior + 0.2*soil, then tiering + trust flags."""
    df = preds.copy()
    fw = config.fusion

    for d in DISEASES:
        df[f"risk_{d}"] = np.clip(
            fw.model * df[f"p_{d}"]
            + fw.weather * aligned.weather_priors[d]
            + fw.soil * aligned.soil_score,
            0.0,
            1.0,
        )

    risk_cols = [f"risk_{d}" for d in DISEASES]
    df["overall_risk"] = df[risk_cols].max(axis=1)
    df["dominant_risk_driver"] = df[risk_cols].idxmax(axis=1).str.replace("risk_", "", regex=False)
    df["overall_tier"] = df["overall_risk"].map(lambda r: tier_of(float(r)))

    escalated = df.apply(
        lambda r: escalate_tier(r["overall_tier"], float(r["overall_risk"]), config.escalation_margin),
        axis=1,
    )
    df["cost_escalated"] = escalated != df["overall_tier"]
    df["overall_tier"] = escalated

    df["needs_expert_review"] = (df["mean_uncertainty"] > config.uncertainty_threshold) | (
        ~df["in_distribution"]
    )
    return df
