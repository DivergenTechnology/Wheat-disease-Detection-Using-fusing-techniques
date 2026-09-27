"""Central configuration for the WDED inference pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Target diseases of the study (order defines model output columns).
DISEASES = ("stem_rust", "stripe_rust", "leaf_rust", "septoria", "fusarium")

# Risk tiers, ordered from the highest boundary for lookup convenience.
RISK_TIERS = (("critical", 0.75), ("high", 0.50), ("moderate", 0.25), ("low", 0.0))


@dataclass(frozen=True)
class FusionWeights:
    """Evidence fusion weights agreed in the implementation plan (sum = 1.0)."""

    model: float = 0.5      # calibrated SSCNN probability
    weather: float = 0.3    # lagged infection-window weather prior
    soil: float = 0.2       # soil suitability-for-disease score


@dataclass
class PipelineConfig:
    """All tunables for one pipeline run."""

    # --- inputs ---------------------------------------------------------
    model_dir: Path = Path("models/trained")
    flight_dir: Path = Path("sample_data/flight")
    weather_csv: Path | None = None
    soil_csv: Path | None = None
    output_dir: Path = Path("runs/latest")

    # --- sensor / geometry ----------------------------------------------
    band_order: tuple = ("green", "red", "red_edge", "nir")   # Mavic 3M (RedEdge-P adds "blue")
    band_wavelengths_nm: dict = field(
        default_factory=lambda: {"green": 560, "red": 650, "red_edge": 730, "nir": 840}
    )
    patch_size: int = 32

    # --- weather alignment ----------------------------------------------
    weather_window_days: int = 21   # longest infection-window lag (Septoria)

    # --- trust layer -----------------------------------------------------
    mc_passes: int = 30                     # MC-Dropout stochastic passes
    calibration: str = "temperature"        # none | temperature
    uncertainty_threshold: float = 0.08     # mean predictive std above -> expert review
    open_set_threshold: float = 0.55        # max disease prob below -> ambiguous zone
    open_set_lower: float = 0.35            # max disease prob below -> confidently healthy
    seed: int = 42

    # --- decision engine --------------------------------------------------
    cost_ratio_fn_fp: float = 5.0           # missed infection costs 5x a false alarm
    escalation_margin: float = 0.05         # asymmetric-cost tier escalation margin
    fusion: FusionWeights = field(default_factory=FusionWeights)

    # --- metadata fallbacks (real values normally come from flight_meta.json) ---
    site_id: str = "bishoftu"
    flight_date: str = ""                   # ISO date; inferred from flight meta when empty
