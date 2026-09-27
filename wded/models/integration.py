"""Integration of a trained SSCNN checkpoint saved in a local directory.

Public entry points
-------------------
``integrate_sscnn(model_dir, ...)``  -> a ready-to-use ``TorchSSCNNAdapter``
``describe_model_dir(model_dir)``    -> diagnostic dict (never raises)
``discover_model_files(model_dir)``  -> file-level inventory of the directory

Supported checkpoint formats (auto-detected, no configuration needed)
---------------------------------------------------------------------
1. TorchScript archive       ``torch.jit.save(scripted_or_traced, "model.pt")``
2. Full pickled module       ``torch.save(model, "model.pt")``
3. Bundle dict with module   ``torch.save({"model": model, "meta": {...}})`` (also
                             accepts keys ``module`` / ``net`` / ``network``)
4. Bundle dict with weights  ``torch.save({"state_dict": sd, ...})`` (also
                             ``model_state_dict`` / ``model_sd`` / ``weights``)
5. Raw state_dict            ``torch.save(model.state_dict(), "weights.pt")``

Formats 4-5 need the network architecture so the weights can be re-instantiated.
It is resolved from (first match wins):
  a. the ``architecture=`` argument — a ``torch.nn.Module`` class, a pre-built
     instance, a zero-argument factory, or an import path string such as
     ``"mypkg.models:SpectralSpatialCNN"``
  b. ``metadata.json`` -> ``{"architecture": "mypkg.models:SpectralSpatialCNN"}``
  c. bundle keys ``arch`` / ``model_class`` (import path strings)

Expected directory layout (only the checkpoint is mandatory)
------------------------------------------------------------
    <model_dir>/
        model.pt | *.pt | *.pth | *.ckpt   checkpoint (``model.pt`` preferred)
        metadata.json                      optional: device, gradcam_layer,
                                           architecture, format
        calibration.json                   optional: {"temperature": 1.42}

Model forward contract (validated with a dry run unless ``validate=False``)
---------------------------------------------------------------------------
    forward(spectral, weather) -> logits
    spectral : (N, B, H, W)  e.g. (32, 4, 32, 32)   B bands, patch H x W
    weather  : (N, W, F)     e.g. (32, 21, 3)       W days x F daily features
    logits   : (N, D)        D = 5 = len(wded.config.DISEASES), order:
                             stem_rust, stripe_rust, leaf_rust, septoria, fusarium
"""
from __future__ import annotations

import importlib
import json
import logging
import zipfile
from pathlib import Path

from ..config import DISEASES

log = logging.getLogger(__name__)

__all__ = [
    "CheckpointFormatError",
    "discover_model_files",
    "describe_model_dir",
    "integrate_sscnn",
]

_CHECKPOINT_EXTS = (".pt", ".pth", ".ckpt")
_PREFERRED_NAMES = ("model.pt", "model.pth")
_MODULE_KEYS = ("model", "module", "net", "network", "estimator")
_STATE_KEYS = ("state_dict", "model_state_dict", "model_sd", "sd", "weights")
_ARCH_KEYS = ("arch", "model_class")
_DEFAULT_TORCH_HINT = "pip install 'wded[torch]'"


class CheckpointFormatError(RuntimeError):
    """Raised when a checkpoint exists but cannot be turned into a usable model."""


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def _pick_checkpoint(model_dir: Path) -> Path | None:
    """Choose the checkpoint file: model.pt > model.pth > remaining names sorted."""
    candidates: list[Path] = []
    for name in _PREFERRED_NAMES:
        p = model_dir / name
        if p.is_file():
            return p
    for ext in _CHECKPOINT_EXTS:
        candidates.extend(sorted(model_dir.glob(f"*{ext}")))
    return candidates[0] if candidates else None


def discover_model_files(model_dir: str | Path) -> dict:
    """Inventory a model directory without importing torch.

    Returns a dict with keys ``model_dir``, ``checkpoint``, ``metadata``,
    ``calibration`` and ``mock``. Missing optional files yield empty dicts.
    """
    d = Path(model_dir)
    out = {
        "model_dir": str(d),
        "checkpoint": None,
        "metadata": {},
        "calibration": {},
        "mock": False,
    }
    if not d.is_dir():
        return out
    out["checkpoint"] = _pick_checkpoint(d)
    for key, fname in (("metadata", "metadata.json"), ("calibration", "calibration.json")):
        fp = d / fname
        if fp.is_file():
            try:
                out[key] = json.loads(fp.read_text())
            except (OSError, ValueError) as exc:
                log.warning("Could not parse %s: %s", fp, exc)
    out["mock"] = (d / "mock_model.json").is_file()
    return out


# ---------------------------------------------------------------------------
# torch helpers (all torch imports are lazy / guarded)
# ---------------------------------------------------------------------------

def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "A torch checkpoint was found but torch is not installed: " + _DEFAULT_TORCH_HINT
        ) from exc
    return torch


def _torch_load(path: Path, torch, device: str):
    """torch.load across versions (weights_only kwarg added in torch 1.13)."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location=device)


def _is_torchscript_archive(path: Path, forced: bool = False) -> bool:
    """TorchScript archives are zip files carrying ``constants.pkl`` and ``code/``.

    Entry names may be prefixed with the archive name (``model.pt/code/...``),
    so match on the suffix after the first path component.
    """
    if forced:
        return True
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except Exception:
        return False

    def _stem(n: str) -> str:
        parts = n.split("/")
        return "/".join(parts[1:]) if len(parts) > 1 else n

    return any(
        _stem(n) == "constants.pkl" or _stem(n).startswith("code/") for n in names
    )


def _looks_like_state_dict(obj, torch) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    vals = list(obj.values())
    tensor_like = sum(1 for v in vals if isinstance(v, torch.Tensor))
    return tensor_like / len(vals) > 0.5


def _extract_model(bundle, torch):
    """Split a loaded checkpoint into (module | None, state_dict | None, hints)."""
    if isinstance(bundle, torch.nn.Module):
        return bundle, None, {}
    if isinstance(bundle, dict):
        hints = {k: bundle[k] for k in _ARCH_KEYS if k in bundle}
        for key in _MODULE_KEYS:
            if key in bundle and isinstance(bundle[key], torch.nn.Module):
                return bundle[key], None, hints
        for key in _STATE_KEYS:
            if key in bundle and _looks_like_state_dict(bundle[key], torch):
                return None, bundle[key], hints
        if _looks_like_state_dict(bundle, torch):
            return None, bundle, hints
        raise CheckpointFormatError(
            f"Checkpoint is a dict with keys {sorted(bundle)} but none of them holds a "
            "torch.nn.Module or a state_dict. Expected {'model': module}, {'state_dict': sd} "
            "or the module itself."
        )
    raise CheckpointFormatError(
        f"Unsupported checkpoint payload of type {type(bundle).__name__}; expected a "
        "torch.nn.Module, a bundle dict or a state_dict."
    )


def _import_object(path: str):
    """Import ``"pkg.module:Name"`` (or ``"pkg.module.Name"``) and return the object."""
    mod_path, sep, name = path.partition(":")
    if not sep:
        mod_path, _, name = path.rpartition(".")
        if not mod_path:
            raise CheckpointFormatError(
                f"Architecture reference {path!r} is not importable; use 'pkg.module:ClassName'."
            )
    try:
        module = importlib.import_module(mod_path)
    except ImportError as exc:
        raise CheckpointFormatError(
            f"Could not import architecture module {mod_path!r}: {exc}"
        ) from exc
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise CheckpointFormatError(
            f"Module {mod_path!r} has no attribute {name!r} (architecture reference {path!r})."
        ) from exc


def _resolve_architecture(architecture, hints, torch):
    """Turn an architecture spec into an instantiated torch.nn.Module."""
    target = architecture
    if target is None:
        target = hints.get("architecture") or hints.get("arch") or hints.get("model_class")
    if target is None:
        raise CheckpointFormatError(
            "The checkpoint contains weights only (state_dict). To rebuild the network, "
            "either (a) pass architecture=... to integrate_sscnn(), (b) add "
            '{"architecture": "pkg.module:ClassName"} to metadata.json, or (c) re-save the '
            "full module with torch.save(model, 'model.pt')."
        )
    if isinstance(target, str):
        target = _import_object(target)
    if isinstance(target, type) and issubclass(target, torch.nn.Module):
        model = target()
    elif isinstance(target, torch.nn.Module):
        model = target
    elif callable(target):
        model = target()
    else:
        raise CheckpointFormatError(
            f"architecture= must be a nn.Module class/instance, a factory or an import "
            f"path string, got {type(target).__name__}."
        )
    if not isinstance(model, torch.nn.Module):
        raise CheckpointFormatError(
            f"Resolved architecture produced {type(model).__name__}, not a torch.nn.Module."
        )
    return model


def _load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise CheckpointFormatError(
            f"state_dict does not match the architecture: {exc}"
        ) from exc
    return model


# ---------------------------------------------------------------------------
# Forward-contract validation
# ---------------------------------------------------------------------------

def _validate_forward(module, torch, device, n_bands, patch_size, weather_days,
                      weather_features, expect_diseases):
    """Dry-run the model on contract-shaped tensors; raise CheckpointFormatError on mismatch."""
    s = torch.rand(2, n_bands, patch_size, patch_size, device=device)
    w = torch.rand(2, weather_days, weather_features, device=device)
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            out = module(s, w)
    except Exception as exc:
        raise CheckpointFormatError(
            "forward(spectral (N,B,H,W), weather (N,W,F)) failed with "
            f"{type(exc).__name__}: {exc}. The module must accept spectral "
            f"(N,{n_bands},{patch_size},{patch_size}) and weather "
            f"(N,{weather_days},{weather_features}) and return logits (N,{expect_diseases})."
        ) from exc
    finally:
        if was_training:
            module.train()
    if isinstance(out, dict):
        out = out.get("logits", next(iter(out.values())))
    if isinstance(out, (tuple, list)):
        out = out[0]
    shape = tuple(out.shape)
    if len(shape) != 2 or shape[1] != expect_diseases:
        raise CheckpointFormatError(
            f"Model output shape {shape} does not match the pipeline contract "
            f"(N, {expect_diseases}) for diseases {DISEASES}. Check that the final layer "
            "has one logit per disease and the disease order matches wded.config.DISEASES."
        )
    return shape


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def integrate_sscnn(
    model_dir: str | Path,
    architecture=None,
    device: str | None = None,
    validate: bool = True,
    n_bands: int = 4,
    patch_size: int = 32,
    weather_days: int = 21,
    weather_features: int = 3,
    expect_diseases: int | None = None,
):
    """Load a trained SSCNN from ``model_dir`` and return a ready TorchSSCNNAdapter.

    Parameters
    ----------
    model_dir:
        Local directory holding the checkpoint (``model.pt`` preferred) and the
        optional ``metadata.json`` / ``calibration.json`` sidecars.
    architecture:
        Only needed for state_dict-only checkpoints: a ``nn.Module`` class, a
        pre-built instance, a zero-argument factory, or an import path string
        ``"pkg.module:ClassName"``. Falls back to ``metadata.json``.
    device:
        ``"cpu"``, ``"cuda"`` or ``None`` to use ``metadata.json["device"]``
        (default ``"cpu"``).
    validate:
        Dry-run a forward pass against the pipeline contract before wiring up
        MC-Dropout / Grad-CAM. Strongly recommended.
    n_bands, patch_size, weather_days, weather_features, expect_diseases:
        Shapes used for the validation dry run; defaults match the WDED pipeline
        (4 Mavic 3M bands, 32x32 patches, 21-day weather window, 3 daily features,
        5 diseases).

    Raises
    ------
    FileNotFoundError
        No checkpoint file in the directory.
    ImportError
        Checkpoint present but torch not installed (hint: ``pip install 'wded[torch]'``).
    CheckpointFormatError
        Checkpoint present but unusable (unknown payload, architecture mismatch,
        or forward contract violation).
    """
    d = Path(model_dir)

    # discovery first: a missing checkpoint must raise FileNotFoundError even
    # on torch-less environments (CI), not the "install torch" hint
    found = discover_model_files(d)
    ckpt = found["checkpoint"]
    meta = found["metadata"]
    if ckpt is None:
        raise FileNotFoundError(
            f"No model checkpoint found in {d.resolve()}: add model.pt (any of the supported "
            f"save formats) or mock_model.json for the deterministic demo model. "
            f"Inspect a directory with: wded model-info <model_dir>"
        )

    torch = _import_torch()
    expect_diseases = len(DISEASES) if expect_diseases is None else expect_diseases

    # device resolution: explicit argument > metadata.json > cpu
    if device is None:
        device = meta.get("device", "cpu")

    # ---- 1. TorchScript archives --------------------------------------
    # torch>=2.x jit.save may emit flatbuffer (non-zip) archives, so a jit.load
    # fallback covers anything the zip sniffing misses.
    forced_ts = str(meta.get("format", "")).lower() in {"torchscript", "jit", "script"}
    fmt = None
    model = None
    if forced_ts or _is_torchscript_archive(ckpt):
        try:
            model = torch.jit.load(str(ckpt), map_location=device)
            fmt = "torchscript"
        except Exception as exc:
            if forced_ts:
                raise CheckpointFormatError(
                    f"metadata.json forces format=torchscript but {ckpt} is not loadable "
                    f"as a TorchScript module: {exc}"
                ) from exc
            log.warning("Zip layout looked like TorchScript but jit.load failed: %s", exc)
    if model is None:
        try:
            bundle = _torch_load(ckpt, torch, device)
        except Exception as load_exc:
            try:
                model = torch.jit.load(str(ckpt), map_location=device)
                fmt = "torchscript"
            except Exception:
                raise CheckpointFormatError(
                    f"Could not load checkpoint {ckpt.name}: {type(load_exc).__name__}: {load_exc}"
                ) from load_exc
    if model is None:
        # bundles often embed their own meta block (e.g. {"model": net, "meta":
        # {"gradcam_layer": ...}}) — honor it unless the sidecar overrides it
        if isinstance(bundle, dict) and isinstance(bundle.get("meta"), dict):
            for key in ("gradcam_layer", "architecture", "arch", "model_class"):
                if key in bundle["meta"]:
                    meta.setdefault(key, bundle["meta"][key])
        model, state_dict, hints = _extract_model(bundle, torch)
        if model is not None:
            # torch.load can itself return a ScriptModule (zip-format jit archive)
            ScriptModule = getattr(torch.jit, "ScriptModule", ())
            fmt = (
                "torchscript"
                if ScriptModule and isinstance(model, ScriptModule)
                else ("bundle" if isinstance(bundle, dict) else "module")
            )
        else:
            arch_hints = dict(hints)
            arch_hints.update({k: v for k, v in meta.items() if k in ("architecture", "arch", "model_class")})
            model = _resolve_architecture(architecture, arch_hints, torch)
            model = _load_state_dict(model, state_dict)
            fmt = "state_dict"

    model = model.to(device).eval()

    # ---- 2. forward-contract dry run -----------------------------------
    if validate:
        _validate_forward(
            model, torch, device, n_bands, patch_size, weather_days, weather_features,
            expect_diseases,
        )

    # ---- 3. trust-layer sidecars ---------------------------------------
    temperature = 1.0
    cal = found["calibration"]
    if cal:
        temperature = float(cal.get("temperature", 1.0))

    from .torch_adapter import TorchSSCNNAdapter

    adapter = TorchSSCNNAdapter(
        model,
        device=device,
        gradcam_layer=meta.get("gradcam_layer"),
        temperature=temperature,
    )
    # provenance / diagnostics
    adapter.checkpoint_path = str(ckpt)
    adapter.checkpoint_format = fmt
    adapter.model_dir = str(d)
    n_params = sum(p.numel() for p in model.parameters())
    adapter.n_parameters = int(n_params)
    log.info(
        "Integrated trained SSCNN from %s (format=%s, params=%d, device=%s, temperature=%.3f%s)",
        ckpt, fmt, n_params, device, temperature,
        f", gradcam={meta.get('gradcam_layer')}" if meta.get("gradcam_layer") else "",
    )
    return adapter


def describe_model_dir(model_dir: str | Path, **integrate_kwargs) -> dict:
    """Best-effort diagnostic report for a local model directory (never raises).

    Combines file discovery with an actual integration attempt and returns a
    JSON-friendly dict — used by ``wded model-info`` and useful in notebooks.
    """
    d = Path(model_dir)
    report = {
        "model_dir": str(d.resolve() if d.exists() else d),
        "exists": d.is_dir(),
        "mock": False,
        "checkpoint": None,
        "checkpoint_format": None,
        "architecture": None,
        "n_parameters": None,
        "device": None,
        "temperature": None,
        "gradcam_layer": None,
        "validated": False,
        "ok": False,
        "error": None,
    }
    if not d.is_dir():
        report["error"] = "directory does not exist"
        return report

    found = discover_model_files(d)
    report["mock"] = found["mock"]
    report["checkpoint"] = str(found["checkpoint"]) if found["checkpoint"] else None
    if found["metadata"]:
        report["gradcam_layer"] = found["metadata"].get("gradcam_layer")
        report["device"] = found["metadata"].get("device")
    if found["calibration"]:
        report["temperature"] = found["calibration"].get("temperature")

    if found["mock"]:
        report["ok"] = True
        report["checkpoint_format"] = "mock"
        report["architecture"] = "MockSSCNN (deterministic demo model)"
        return report
    if found["checkpoint"] is None:
        report["error"] = (
            "no checkpoint (*.pt/*.pth/*.ckpt) and no mock_model.json in the directory"
        )
        return report

    try:
        adapter = integrate_sscnn(d, **integrate_kwargs)
        report.update(
            ok=True,
            checkpoint_format=adapter.checkpoint_format,
            architecture=type(adapter.model).__name__,
            n_parameters=adapter.n_parameters,
            device=adapter.device,
            temperature=adapter.temperature,
            validated=bool(integrate_kwargs.get("validate", True)),
        )
        if adapter._gradcam_layer:
            report["gradcam_layer"] = adapter._gradcam_layer
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report
