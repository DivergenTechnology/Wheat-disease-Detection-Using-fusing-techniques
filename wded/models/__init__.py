"""Model loading: chooses the torch checkpoint adapter or the deterministic mock.

Contract:
  model_dir/mock_model.json  -> MockSSCNN (demos, tests, CI)
  model_dir/*.pt|*.pth|*.ckpt -> TorchSSCNNAdapter (real trained SSCNN)

Torch checkpoints are loaded by :func:`wded.models.integration.integrate_sscnn`,
which auto-detects the save format (TorchScript, full pickled module, bundle
dict, state_dict + architecture) and dry-run validates the forward contract.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import PipelineConfig
from .integration import (  # noqa: F401  (re-exported for convenience)
    CheckpointFormatError,
    describe_model_dir,
    discover_model_files,
    integrate_sscnn,
)
from .mock import MockSSCNN

__all__ = [
    "load_model",
    "integrate_sscnn",
    "describe_model_dir",
    "discover_model_files",
    "CheckpointFormatError",
    "MockSSCNN",
]


def load_model(config: PipelineConfig):
    """Resolve the model for one pipeline run (mock first, then trained checkpoint)."""
    model_dir = Path(config.model_dir)

    marker = model_dir / "mock_model.json"
    if marker.exists():
        seed = int(json.loads(marker.read_text()).get("seed", config.seed))
        return MockSSCNN(seed=seed)

    # Trained checkpoint path — the integration loader raises the friendly
    # FileNotFoundError / ImportError / CheckpointFormatError messages.
    return integrate_sscnn(
        model_dir,
        n_bands=len(config.band_order),
        patch_size=config.patch_size,
        weather_days=config.weather_window_days,
        weather_features=3,
    )
