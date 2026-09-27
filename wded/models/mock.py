"""Deterministic mock SSCNN for demos, integration tests and CI.

Stands in for the trained spectral-spatial CNN + weather-LSTM dual branch.
Each disease responds to a distinct spectral signature (agronomy-guided):
stem rust crashes NDVI (severe canopy loss), stripe rust depresses NDRE first
(early chlorophyll loss), Septoria is texture-driven (GLCM contrast), so
different hotspots surface as different dominant diseases. Seeded MC noise
makes the whole pipeline reproducible without torch.

Replace with ``wded.models.torch_adapter.TorchSSCNNAdapter`` for real checkpoints.
"""
from __future__ import annotations

import numpy as np

from ..config import DISEASES

_IDX_FEATURES = ["ndvi", "ndre", "reip", "gndvi", "glcm_contrast", "glcm_homogeneity", "glcm_energy"]

# Standardisation for the spectral branch - the mock's "training distribution"
# (calibrated so the synthetic demo domain and the mock agree; the real torch
# model ships its own normalisation inside the checkpoint).
_STD = {
    "ndvi": (0.75, 0.06),
    "ndre": (0.30, 0.030),
    "reip": (702.0, 2.5),
    "gndvi": (0.66, 0.05),
    "glcm_contrast": (0.15, 0.12),
    "glcm_homogeneity": (0.93, 0.09),
    "glcm_energy": (0.85, 0.18),
}

# Spectral-branch gain (logits = gain * z) - kept low so per-disease signatures
# differentiate instead of saturating (all diseases share the texture terms,
# so high gain would compress the p-gap and let the weather mix flip ranks).
_SPEC_GAIN = 0.12

# Per-disease spectral response (ndvi, ndre, reip, gndvi, contrast, homogeneity, energy).
_W_SPEC = {
    "stem_rust":   [-3.0, -0.8, 0.4, -1.0, 0.8, -0.4, -0.3],   # canopy collapse -> NDVI crash + REIP jump
    "stripe_rust": [-0.6, -2.2, -1.2, -0.8, 0.8, -0.5, -0.4],  # red-edge shift -> NDRE drop + REIP blue-shift
    "leaf_rust":   [-1.0, -1.2, 0.1, -0.7, 0.9, -0.5, -0.3],   # moderate general decline
    "septoria":    [-0.5, -0.4, -0.3, -0.5, 1.6, -0.5, -0.4],  # necrotic lesion texture (GLCM)
    "fusarium":    [-0.5, -0.4, 0.2, -0.4, 0.9, -0.4, -0.3],   # head-centric, weak on leaves
}

# (temp, wetness, rh) weights per disease for the weather branch
_W_WEATHER = {
    "stem_rust": (0.9, 0.4, 0.2),
    "stripe_rust": (1.1, 0.5, 0.1),
    "leaf_rust": (0.8, 0.4, 0.2),
    "septoria": (0.4, 1.0, 0.3),
    "fusarium": (0.6, 0.8, 0.4),
}

_T_MID = {"stem_rust": 25.0, "stripe_rust": 12.5, "leaf_rust": 22.0,
          "septoria": 17.0, "fusarium": 25.0}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class MockSSCNN:
    """Same interface as TorchSSCNNAdapter: predict_with_uncertainty()."""

    temperature = 1.0
    mc_logit_noise = 0.30
    weather_mix = 0.10  # base = (1-mix)*p_spec + mix*P_weather + bias

    def __init__(self, seed: int = 42):
        assert tuple(_W_WEATHER) == DISEASES, "weather-branch weights must follow DISEASES order"
        self._seed = int(seed)
        rng = np.random.default_rng(self._seed)
        # light per-run jitter so the mock is not trivially static, yet deterministic
        jitter = 1.0 + 0.10 * rng.normal(size=(len(DISEASES), len(_IDX_FEATURES)))
        self._w_spec = np.array([_W_SPEC[d] for d in DISEASES]) * jitter       # (5, 7)
        self._biases = np.array([-0.17, -0.12, -0.17, -0.12, -0.42])             # order = DISEASES

    def _spectral_features(self, indices):
        cols = []
        for name in _IDX_FEATURES:
            mu, sd = _STD[name]
            cols.append((np.asarray(indices[name], dtype=np.float64) - mu) / sd)
        return np.stack(cols, axis=1)                                          # (N, 7)

    def predict_with_uncertainty(self, tiles, indices, weather_seq, n_mc=30):
        """Return (mean_p (N, D), std_p (N, D), attention (N,))."""
        feats = self._spectral_features(indices)                               # (N, 7)
        p_spec = _sigmoid(_SPEC_GAIN * feats @ self._w_spec.T)                 # (N, D)

        temp = weather_seq[..., 0].mean(axis=1) * 30.0                         # weather branch (N,)
        rh = weather_seq[..., 1].mean(axis=1) * 100.0
        lwh = weather_seq[..., 2].mean(axis=1) * 24.0
        P_weather = np.stack(
            [
                _sigmoid(
                    a * np.tanh((temp - _T_MID[d]) / 6.0)
                    + b * (lwh / 12.0)
                    + c * ((rh - 70.0) / 20.0)
                )
                for d, (a, b, c) in _W_WEATHER.items()
            ],
            axis=1,
        )                                                                      # (N, D)

        base = np.clip(
            (1.0 - self.weather_mix) * p_spec + self.weather_mix * P_weather + self._biases[None, :],
            0.02,
            0.98,
        )

        rng = np.random.default_rng(self._seed)                                # deterministic MC
        logits = _logit(base)[None, :, :] + rng.normal(0.0, self.mc_logit_noise, size=(n_mc, *base.shape))
        p_mc = _sigmoid(logits)
        mean_p = p_mc.mean(axis=0)
        std_p = p_mc.std(axis=0)

        ndvi = np.asarray(indices["ndvi"], dtype=np.float64)
        contrast = np.asarray(indices["glcm_contrast"], dtype=np.float64)
        attention = (
            np.clip((0.75 - ndvi) / 0.10, 0.0, 1.0) * 0.7
            + np.clip(contrast / 0.35, 0.0, 1.0) * 0.3
        )
        return mean_p, std_p, attention.astype(np.float32)
