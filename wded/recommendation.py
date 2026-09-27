"""Stage 6: turn calibrated, fused risk into agronomically actionable recommendations."""
from __future__ import annotations

import pandas as pd

TIER_ACTIONS = {
    "critical": (
        "Apply fungicide immediately; expert on-ground confirmation within 24 h; "
        "flag the block for harvest segregation."
    ),
    "high": (
        "Confirmatory scouting within 48 h; prepare the spray plan; "
        "re-fly the block in 5-7 days."
    ),
    "moderate": (
        "Intensify monitoring: re-flight within 72 h and scout flagged tiles; "
        "watch the 7-day weather forecast for infection windows."
    ),
    "low": (
        "Routine monitoring cadence; no targeted action before the next scheduled flight."
    ),
}

DISEASE_NOTES = {
    "stem_rust": "Ug99-race watch: if severity exceeds 10%, send samples for race typing.",
    "stripe_rust": "Cool-season favourite: prioritise upper-canopy checking after cool, wet spells.",
    "leaf_rust": "Protect the flag leaf; check for post-flushing tillers after rain.",
    "septoria": "Prioritise lower-canopy lesions; protect the flag leaf at T0/T1 timing.",
    "fusarium": "If the crop is at anthesis, assess FHB and DON mycotoxin risk before harvest.",
}

# Escalation speed and re-scouting cadence per tier (days until the block
# should be re-flown / re-scouted after the current observation).
TIER_URGENCY = {
    "critical": "immediate - act within 24 h",
    "high": "urgent - act within 48 h",
    "moderate": "elevated - monitor within 72 h",
    "low": "routine - next scheduled flight",
}
TIER_RESCOUT_DAYS = {"critical": 3, "high": 5, "moderate": 7, "low": 14}


def generate_recommendation(fused: pd.DataFrame) -> pd.DataFrame:
    df = fused.copy()
    df["recommendation"] = df["overall_tier"].map(TIER_ACTIONS)
    df["urgency"] = df["overall_tier"].map(TIER_URGENCY)
    df["rescout_days"] = (
        df["overall_tier"].map(TIER_RESCOUT_DAYS).astype("int64")
    )
    df["disease_note"] = (
        df["dominant_disease"].map(DISEASE_NOTES).fillna("Confirm symptoms before treatment.")
    )
    return df
