"""Tests for v0.6.0 — trained SSCNN model integration from a local directory.

Covers the format-auto-detecting loader (``wded.models.integration``), the
``wded model-info`` diagnostic command and an end-to-end pipeline run driven by
a real torch checkpoint instead of the deterministic mock.

Torch-dependent tests skip automatically on environments without torch (CI);
the no-torch tests always run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wded import __version__ as WDED_VERSION  # noqa: E402
from wded.cli import main as cli_main  # noqa: E402
from wded.config import DISEASES, PipelineConfig  # noqa: E402
from wded.models import (  # noqa: E402
    CheckpointFormatError,
    MockSSCNN,
    describe_model_dir,
    discover_model_files,
    integrate_sscnn,
    load_model,
)
from wded.models.integration import _is_torchscript_archive  # noqa: E402

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - CI installs only [geo,dev]
    torch = None
    nn = None
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# Stand-in architectures (module level so torch.save's pickle can find them)
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class TinySSCNN(nn.Module):
        """Minimal dual-branch module satisfying the WDED forward contract."""

        def __init__(self, bands=4, patch=32, days=21, feats=3, n=len(DISEASES)):
            super().__init__()
            self.conv = nn.Conv2d(bands, 8, 3, padding=1)
            self.drop = nn.Dropout(0.25)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.weather_fc = nn.Linear(days * feats, 16)
            self.head = nn.Linear(8 + 16, n)

        def forward(self, spectral, weather):
            x = self.pool(torch.relu(self.conv(spectral))).flatten(1)
            w = torch.relu(self.weather_fc(weather.flatten(1)))
            return self.head(self.drop(torch.cat([x, w], dim=1)))

    class TinySSCNNWrongDim(TinySSCNN):
        """Deliberately violates the contract: 4 outputs instead of 5."""

        def __init__(self, **kw):
            super().__init__(n=4)

    class BadSignature(nn.Module):
        """Module whose forward cannot consume contract-shaped weather tensors."""

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(14, len(DISEASES))

        def forward(self, spectral, weather):  # fc expects 14 features, gets 63
            return self.fc(weather.flatten(1))

else:  # torch-free import path keeps collection alive on CI

    class TinySSCNN:  # pragma: no cover - placeholder, never instantiated
        pass


def _write_sidecars(d: Path, metadata=None, calibration=None):
    if metadata is not None:
        (d / "metadata.json").write_text(json.dumps(metadata))
    if calibration is not None:
        (d / "calibration.json").write_text(json.dumps(calibration))


# ---------------------------------------------------------------------------
# No-torch tests: discovery, errors, CLI, package surface (run everywhere)
# ---------------------------------------------------------------------------


def test_version_bumped_to_060():
    assert WDED_VERSION == "0.6.0"
    pyproject = (REPO / "pyproject.toml").read_text()
    assert 'version = "0.6.0"' in pyproject


def test_package_exports_integration_api():
    import wded
    import wded.models

    for name in (
        "integrate_sscnn",
        "describe_model_dir",
        "discover_model_files",
        "CheckpointFormatError",
        "load_model",
    ):
        assert hasattr(wded, name), f"wded.{name} missing"
        assert hasattr(wded.models, name), f"wded.models.{name} missing"


def test_describe_missing_and_empty_dirs(tmp_path):
    report = describe_model_dir(tmp_path / "does_not_exist")
    assert report["ok"] is False
    assert report["error"] == "directory does not exist"

    empty = tmp_path / "empty"
    empty.mkdir()
    report = describe_model_dir(empty)
    assert report["ok"] is False
    assert report["checkpoint"] is None
    assert "no checkpoint" in report["error"]


def test_discovery_prefers_model_pt_and_parses_sidecars(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    (d / "aaa.pt").write_bytes(b"x")
    (d / "model.pt").write_bytes(b"x")
    (d / "zz.pth").write_bytes(b"x")
    _write_sidecars(
        d,
        metadata={"device": "cpu", "gradcam_layer": "backbone.6"},
        calibration={"temperature": 1.42},
    )
    found = discover_model_files(d)
    assert found["checkpoint"].name == "model.pt"
    assert found["metadata"]["gradcam_layer"] == "backbone.6"
    assert found["calibration"]["temperature"] == 1.42
    assert found["mock"] is False

    assert _is_torchscript_archive(d / "model.pt") is False  # not a zip


def test_discovery_ckpt_extension_and_mock_flag(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "epoch9.ckpt").write_bytes(b"x")
    (d / "mock_model.json").write_text(json.dumps({"seed": 7}))
    found = discover_model_files(d)
    assert found["checkpoint"].name == "epoch9.ckpt"
    assert found["mock"] is True


def test_load_model_missing_checkpoint_raises(tmp_path):
    cfg = PipelineConfig(model_dir=tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No model checkpoint"):
        load_model(cfg)


def test_load_model_still_prefers_mock(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    (d / "mock_model.json").write_text(json.dumps({"seed": 7}))
    cfg = PipelineConfig(model_dir=d)
    model = load_model(cfg)
    assert isinstance(model, MockSSCNN)


def test_checkpoint_without_torch_gives_install_hint(tmp_path, monkeypatch):
    d = tmp_path / "trained"
    d.mkdir()
    (d / "model.pt").write_bytes(b"PK\x03\x04 dummy")
    monkeypatch.setitem(sys.modules, "torch", None)  # forces `import torch` to fail
    with pytest.raises(ImportError, match=r"wded\[torch\]"):
        integrate_sscnn(d)


def test_cli_model_info_mock_dir_exit_zero(tmp_path, capsys):
    d = tmp_path / "trained"
    d.mkdir()
    (d / "mock_model.json").write_text(json.dumps({"seed": 7}))
    assert cli_main(["model-info", str(d)]) == 0
    out = capsys.readouterr().out
    assert "mock" in out.lower()
    assert "integrated successfully" in out


def test_cli_model_info_bad_dir_exit_one(tmp_path, capsys):
    d = tmp_path / "empty"
    d.mkdir()
    assert cli_main(["model-info", str(d)]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out


# ---------------------------------------------------------------------------
# Torch tests: every supported save format + end-to-end pipeline
# ---------------------------------------------------------------------------


@requires_torch
def test_integrate_full_module_checkpoint(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    model = TinySSCNN()
    torch.save(model, d / "model.pt")  # torch.save(model) — full pickled module

    adapter = integrate_sscnn(d)
    assert adapter.checkpoint_format == "module"
    assert adapter.n_parameters > 0
    assert adapter.checkpoint_path.endswith("model.pt")

    tiles = torch.rand(3, 4, 32, 32).numpy()
    weather = torch.rand(3, 21, 3).numpy()
    mean_p, std_p, attention = adapter.predict_with_uncertainty(tiles, None, weather, n_mc=4)
    assert mean_p.shape == (3, len(DISEASES))
    assert std_p.shape == (3, len(DISEASES))
    assert attention.shape == (3,)
    assert pytest.approx(float(attention.sum()), abs=1e-5) == 1.0


@requires_torch
def test_integrate_bundle_dict_checkpoint(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    model = TinySSCNN()
    torch.save({"model": model, "meta": {"epoch": 9}, "epoch": 9}, d / "model.pt")

    adapter = integrate_sscnn(d)
    assert adapter.checkpoint_format == "bundle"
    assert type(adapter.model).__name__ == "TinySSCNN"


@requires_torch
def test_integrate_bundle_module_key(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save({"module": TinySSCNN()}, d / "weights.pth")
    adapter = integrate_sscnn(d)
    assert adapter.checkpoint_format == "bundle"


@requires_torch
def test_integrate_state_dict_bundle_with_factory(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    model = TinySSCNN()
    torch.save({"state_dict": model.state_dict(), "optimizer": {}}, d / "model.pt")

    adapter = integrate_sscnn(d, architecture=TinySSCNN)
    assert adapter.checkpoint_format == "state_dict"
    mean_p, _, _ = adapter.predict_with_uncertainty(
        torch.rand(2, 4, 32, 32).numpy(), None, torch.rand(2, 21, 3).numpy(), n_mc=2
    )
    assert mean_p.shape == (2, len(DISEASES))


@requires_torch
def test_integrate_raw_state_dict_via_metadata_import_path(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(TinySSCNN().state_dict(), d / "weights.pt")
    _write_sidecars(d, metadata={"architecture": f"{__name__}:TinySSCNN"})

    adapter = integrate_sscnn(d)  # architecture resolved from metadata.json
    assert adapter.checkpoint_format == "state_dict"


@requires_torch
def test_integrate_state_dict_without_architecture_raises(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(TinySSCNN().state_dict(), d / "weights.pt")
    with pytest.raises(CheckpointFormatError, match="state_dict"):
        integrate_sscnn(d)


@requires_torch
def test_integrate_architecture_string_arg(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(TinySSCNN().state_dict(), d / "weights.pt")
    adapter = integrate_sscnn(d, architecture=f"{__name__}:TinySSCNN")
    assert adapter.checkpoint_format == "state_dict"


@requires_torch
def test_integrate_mismatched_state_dict_raises(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()

    class OtherNet(nn.Module):  # different layer names -> state_dict mismatch
        def __init__(self):
            super().__init__()
            self.unrelated = nn.Linear(4, len(DISEASES))

        def forward(self, spectral, weather):
            return self.unrelated(spectral.flatten(1))

    torch.save({"state_dict": OtherNet().state_dict()}, d / "model.pt")
    with pytest.raises(CheckpointFormatError, match="state_dict does not match"):
        integrate_sscnn(d, architecture=TinySSCNN, validate=False)


@requires_torch
def test_integrate_torchscript_checkpoint(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    model = TinySSCNN().eval()
    scripted = torch.jit.script(model)
    scripted.save(str(d / "model.pt"))  # torch>=2.x may write flatbuffer (non-zip)

    adapter = integrate_sscnn(d)  # auto-detected with or without zip sniffing
    assert adapter.checkpoint_format == "torchscript"
    mean_p, std_p, attention = adapter.predict_with_uncertainty(
        torch.rand(2, 4, 32, 32).numpy(), None, torch.rand(2, 21, 3).numpy(), n_mc=5
    )
    assert mean_p.shape == (2, len(DISEASES))
    assert std_p.shape == (2, len(DISEASES))
    # scripted model keeps its Dropout: MC re-activation must find and train() it
    dropout_active = any(
        str(getattr(m, "original_name", "")).rsplit(".", 1)[-1].startswith("Dropout")
        and m.training
        for m in adapter.model.modules()
    )
    assert dropout_active, "MC-Dropout was not re-activated on the scripted module"


@requires_torch
def test_validation_rejects_wrong_output_dim(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(TinySSCNNWrongDim(), d / "model.pt")
    with pytest.raises(CheckpointFormatError, match=r"\(N, 5\)"):
        integrate_sscnn(d)


@requires_torch
def test_validation_rejects_broken_forward(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(BadSignature(), d / "model.pt")  # module-level class: picklable
    with pytest.raises(CheckpointFormatError, match="forward"):
        integrate_sscnn(d)


@requires_torch
def test_validation_can_be_skipped(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(TinySSCNNWrongDim(), d / "model.pt")
    adapter = integrate_sscnn(d, validate=False)  # loads despite contract breach
    assert adapter.checkpoint_format == "module"


@requires_torch
def test_calibration_and_gradcam_metadata_wired(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    model = TinySSCNN()
    # deterministic conv (pure brightness detector) -> guaranteed Grad-CAM split:
    # dark tiles have zero activations (cam = 0), bright tiles get all the focus
    model.conv.weight.data.fill_(1.0)
    model.conv.bias.data.zero_()
    torch.save(model, d / "model.pt")
    _write_sidecars(
        d,
        metadata={"gradcam_layer": "conv", "device": "cpu"},
        calibration={"temperature": 1.23},
    )

    adapter = integrate_sscnn(d)
    assert adapter.temperature == pytest.approx(1.23)
    assert adapter._gradcam_layer == "conv"
    assert adapter.device == "cpu"

    tiles = torch.cat([torch.zeros(2, 4, 32, 32), torch.full((2, 4, 32, 32), 0.9)]).numpy()
    _, _, attention = adapter.predict_with_uncertainty(
        tiles, None, torch.rand(4, 21, 3).numpy(), n_mc=2
    )
    # Grad-CAM attention must be real (non-uniform) and normalised
    assert attention is not None
    assert pytest.approx(float(attention.sum()), abs=1e-4) == 1.0
    assert float(attention.std()) > 1e-4
    assert attention[2] > attention[0]


@requires_torch
def test_bundle_embedded_meta_wired(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(
        {"model": TinySSCNN(), "meta": {"gradcam_layer": "conv"}, "epoch": 5},
        d / "model.pt",
    )
    adapter = integrate_sscnn(d)  # gradcam_layer honoured from the bundle's meta block
    assert adapter._gradcam_layer == "conv"


@requires_torch
def test_describe_model_dir_with_trained_checkpoint(tmp_path):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(TinySSCNN(), d / "model.pt")
    _write_sidecars(d, calibration={"temperature": 0.9})
    report = describe_model_dir(d)
    assert report["ok"] is True
    assert report["checkpoint_format"] == "module"
    assert report["architecture"] == "TinySSCNN"
    assert report["validated"] is True
    assert report["temperature"] == 0.9


@requires_torch
def test_end_to_end_pipeline_runs_on_trained_checkpoint(tmp_path):
    """The full pipeline driven by a real checkpoint instead of the mock."""
    from wded.demo import generate_sample_data
    from wded.pipeline import run_pipeline

    src = tmp_path / "src"
    generate_sample_data(src, seed=42)

    summaries = []
    for run_name, seed in (("run_a", 101), ("run_b", 202)):
        d = tmp_path / f"model_{run_name}"
        d.mkdir()
        torch.manual_seed(seed)
        torch.save(TinySSCNN(), d / "model.pt")

        cfg = PipelineConfig(
            model_dir=d,
            flight_dir=src / "flight",
            weather_csv=src / "weather.csv",
            soil_csv=src / "soil.csv",
            output_dir=tmp_path / run_name,
            mc_passes=3,
            seed=seed,
        )
        summaries.append(run_pipeline(cfg))

    for s in summaries:
        assert "run_sscnn_inference" in s["stages_completed"]
        assert sum(s["tier_counts"].values()) == s["n_tiles"]
    preds = tmp_path / "run_a" / "tile_predictions.csv"
    assert preds.exists()
    text = preds.read_text()
    assert "p_stem_rust" in text and "dominant_disease" in text

    # two different trained weights must give different predictions
    # (proves the checkpoint — not the mock — drove the inference)
    assert (
        summaries[0]["mean_overall_risk"] != summaries[1]["mean_overall_risk"]
    )


@requires_torch
def test_cli_model_info_trained_checkpoint_exit_zero(tmp_path, capsys):
    d = tmp_path / "trained"
    d.mkdir()
    torch.save(TinySSCNN(), d / "model.pt")
    _write_sidecars(d, metadata={"gradcam_layer": "conv"})
    assert cli_main(["model-info", str(d)]) == 0
    out = capsys.readouterr().out
    assert "format        : module" in out
    assert "architecture  : TinySSCNN" in out
    assert "gradcam_layer : conv" in out
    assert "dry-run       : passed" in out
