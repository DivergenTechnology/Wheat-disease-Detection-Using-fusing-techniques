"""PyTorch adapter for the real trained SSCNN checkpoint (MC-Dropout + Grad-CAM).

Checkpoint contract (see README "Using the trained model"):
  model_dir/model.pt         torch.save({"model": module, "meta": {...}}) or the module itself
  model_dir/metadata.json    {"gradcam_layer": "backbone.6", "device": "cpu"}   (optional)
  model_dir/calibration.json {"temperature": 1.42}                            (optional)

The wrapped module must implement ``forward(spectral, weather) -> logits (N, D)`` with
spectral shaped (N, B, H, W) and weather shaped (N, W, F). Disease order must follow
``wded.config.DISEASES``.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _enable_mc_dropout(model):
    """Keep the network in eval mode but reactivate Dropout layers for MC sampling.

    Also handles TorchScript modules, whose submodules are RecursiveScriptModule
    wrappers (isinstance checks fail; ``original_name`` is e.g. ``'Dropout'``).
    """
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.modules.dropout._DropoutNd)):
            m.train()
            continue
        original = str(getattr(m, "original_name", "")).rsplit(".", 1)[-1]
        if original.startswith("Dropout"):
            m.train()


class TorchSSCNNAdapter:
    temperature = 1.0

    def __init__(self, model, device="cpu", gradcam_layer=None, temperature=1.0):
        import torch

        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = float(temperature)
        self._gradcam_layer = gradcam_layer
        self._acts = None
        self._grads = None
        if gradcam_layer:
            layer = dict([*self.model.named_modules()])[gradcam_layer]
            layer.register_forward_hook(self._save_act)
            layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inputs, output):
        self._acts = output.detach()

    def _save_grad(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    def _gradcam_attention(self, spectral, weather):
        """Per-tile Grad-CAM saliency of the dominant class, spatially averaged."""
        if not self._gradcam_layer or self._acts is None:
            return None
        t = self.torch
        s = t.as_tensor(spectral, dtype=t.float32, device=self.device).requires_grad_(True)
        w = t.as_tensor(weather, dtype=t.float32, device=self.device)
        out = self.model(s, w)                      # (N, D) logits; hooks populate acts/grads
        cls = out.argmax(dim=1)
        self.model.zero_grad(set_to_none=True)
        out[t.arange(len(cls)), cls].sum().backward()
        if self._grads is None or self._acts is None:
            return None
        cams = t.relu((self._acts * self._grads).sum(dim=1))          # (N, h, w)
        denom = cams.flatten(1).sum(dim=1).clamp_min(1e-9)
        per_tile = (cams.flatten(1) / denom[:, None]).mean(dim=1)     # focus intensity
        total = per_tile.sum().clamp_min(1e-9)
        # normalise so attention sums to 1 across tiles — consistent with the
        # uniform fallback (attention = share of the model's focus per tile)
        return (per_tile / total).detach().cpu().numpy()

    def predict_with_uncertainty(self, tiles, indices, weather_seq, n_mc=30):
        t = self.torch
        s = t.as_tensor(tiles, dtype=t.float32, device=self.device)
        w = t.as_tensor(weather_seq, dtype=t.float32, device=self.device)
        self.model.eval()
        _enable_mc_dropout(self.model)

        with t.no_grad():
            passes = [self.model(s, w) for _ in range(n_mc)]
        logits = t.stack(passes)                     # (mc, N, D)
        if self.temperature not in (None, 1.0):
            logits = logits / self.temperature
        p_mc = t.softmax(logits, dim=-1).cpu().numpy()

        mean_p, std_p = p_mc.mean(axis=0), p_mc.std(axis=0)

        attention = None
        try:
            attention = self._gradcam_attention(tiles, weather_seq)
        except Exception as exc:  # pragma: no cover - depends on user model
            log.warning("Grad-CAM failed (%s); falling back to uniform attention", exc)
        if attention is None:
            attention = np.full(len(tiles), 1.0 / max(len(tiles), 1), dtype=np.float32)

        return mean_p.astype(np.float32), std_p.astype(np.float32), attention.astype(np.float32)
