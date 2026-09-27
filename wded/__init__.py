"""WDED — Wheat Disease Early Detection inference pipeline.

Fuses UAV multispectral reflectance dynamics with lagged-weather epidemiology
and soil covariates to detect wheat infection windows (stem rust, stripe rust,
leaf rust, Septoria, Fusarium) before visible symptoms, and converts them into
calibrated, actionable field risk tiers.
"""

__version__ = "0.5.0"

from .config import DISEASES, RISK_TIERS, FusionWeights, PipelineConfig  # noqa: F401
