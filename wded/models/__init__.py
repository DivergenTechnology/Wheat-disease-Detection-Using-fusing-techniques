"""Model loading: chooses the torch checkpoint adapter or the deterministic mock.

Contract:
  model_dir/mock_model.json  -> MockSSCNN (demos, tests, CI)
  model_dir/*.pt|*.pth       -> TorchSSCNNAdapter (real trained SSCNN)
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import PipelineConfig
from .mock import MockSSCNN

__all__ = ["load_model", "MockSSCNN"]


def load_model(config: PipelineConfig):
    model_dir = Path(config.model_dir)

    marker = model_dir / "mock_model.json"
    if marker.exists():
        seed = int(json.loads(marker.read_text()).get("seed", config.seed))
        return MockSSCNN(seed=seed)

    weights = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.pth"))
    if weights:
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "A torch checkpoint was found but torch is not installed: pip install 'wded[torch]'"
            ) from exc
        from .torch_adapter import TorchSSCNNAdapter

        bundle = torch.load(weights[0], map_location="cpu", weights_only=False)
        meta_path = model_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        if isinstance(bundle, dict) and "state_dict" in bundle:
            raise NotImplementedError(
                "state_dict-only checkpoints need the architecture class. Save the full module "
                "(`torch.save(model, ...)`) or a bundle {'model': model, 'meta': {...}} - see README."
            )
        model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle

        temperature = 1.0
        cal_path = model_dir / "calibration.json"
        if cal_path.exists():
            temperature = float(json.loads(cal_path.read_text()).get("temperature", 1.0))
        return TorchSSCNNAdapter(
            model,
            device=meta.get("device", "cpu"),
            gradcam_layer=meta.get("gradcam_layer"),
            temperature=temperature,
        )

    raise FileNotFoundError(
        f"No model found in {model_dir}: add mock_model.json (demo) or a torch checkpoint (.pt/.pth)"
    )
