"""Integration tests: the full WDED pipeline runs end-to-end on synthetic data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wded.alignment import align_weather_soil, infection_weather_prior
from wded.cli import main as cli_main
from wded.config import PipelineConfig
from wded.demo import generate_sample_data
from wded.fusion import escalate_tier, fuse_decision, tier_of
from wded.inference import run_sscnn_inference
from wded.models import load_model
from wded.pipeline import run_pipeline
from wded.preprocess import preprocess_new_flight


@pytest.fixture()
def demo_env(tmp_path):
    root = tmp_path / "inputs"
    generate_sample_data(root, seed=42)
    return PipelineConfig(
        model_dir=root / "models" / "trained",
        flight_dir=root / "flight",
        weather_csv=root / "weather.csv",
        soil_csv=root / "soil.csv",
        output_dir=tmp_path / "runs" / "demo",
        mc_passes=12,
    )


def test_end_to_end_demo_pipeline(demo_env):
    cfg = demo_env
    summary = run_pipeline(cfg)
    out = Path(cfg.output_dir)
    for name in (
        "tile_predictions.csv",
        "expert_review_queue.csv",
        "risk_polygons.geojson",
        "attention_heatmap.geojson",
        "summary.json",
    ):
        assert (out / name).exists(), f"missing output {name}"

    fc = json.loads((out / "risk_polygons.geojson").read_text())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == summary["n_tiles"]

    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert 38.0 < lon < 40.0 and 8.0 < lat < 9.5, "tiles must fall around Bishoftu (UTM 37N)"

    assert sum(summary["tier_counts"].values()) == summary["n_tiles"]
    assert summary["fusion_weights"] == {"model": 0.5, "weather": 0.3, "soil": 0.2}


def test_fusion_weights_are_exact(demo_env):
    cfg = demo_env
    flight = preprocess_new_flight(cfg)
    model = load_model(cfg)
    aligned = align_weather_soil(flight, cfg)
    preds = run_sscnn_inference(model, flight, aligned, cfg)
    fused = fuse_decision(preds, aligned, cfg)

    row = fused.iloc[0]
    expected = (
        0.5 * row["p_stem_rust"]
        + 0.3 * aligned.weather_priors["stem_rust"]
        + 0.2 * aligned.soil_score
    )
    assert float(row["risk_stem_rust"]) == pytest.approx(float(expected), abs=1e-9)


def test_review_flags_consistent(demo_env):
    cfg = demo_env
    run_pipeline(cfg)
    df = pd.read_csv(Path(cfg.output_dir) / "tile_predictions.csv")
    expected = (df["mean_uncertainty"] > cfg.uncertainty_threshold) | (~df["in_distribution"])
    assert (df["needs_expert_review"] == expected).all()
    # risk probabilities stay in [0, 1]
    for col in df.columns:
        if col.startswith("p_") or col.startswith("risk_"):
            assert df[col].between(0, 1).all(), col


def test_infection_priors_respond_to_temperature():
    base = {
        "date": pd.date_range("2026-08-01", periods=21).strftime("%Y-%m-%d"),
        "temp_mean": np.full(21, 12.0),
        "rh_mean": np.full(21, 85.0),
        "leaf_wetness_hours": np.full(21, 10.0),
        "rain_mm": np.full(21, 2.0),
    }
    cool = pd.DataFrame(base)
    warm = cool.copy()
    warm["temp_mean"] = 28.0
    assert infection_weather_prior(cool, "stripe_rust") > infection_weather_prior(
        warm, "stripe_rust"
    )
    assert infection_weather_prior(warm, "stem_rust") > infection_weather_prior(cool, "stem_rust")


def test_tiering_and_asymmetric_cost_escalation():
    assert tier_of(0.80) == "critical"
    assert tier_of(0.10) == "low"
    assert escalate_tier("moderate", 0.48, 0.05) == "high"    # within margin -> escalate
    assert escalate_tier("moderate", 0.40, 0.05) == "moderate"  # outside margin -> keep


def test_cli_demo(tmp_path):
    rc = cli_main(["demo", "--root", str(tmp_path / "in"), "--out", str(tmp_path / "out")])
    assert rc == 0
    assert (tmp_path / "out" / "summary.json").exists()


def test_geotiff_flight_input(tmp_path):
    """Real-raster path: synthetic bands as GeoTIFFs, read back without rasterio."""
    pytest.importorskip("tifffile")
    root = tmp_path / "inputs"
    generate_sample_data(root, seed=42, fmt="geotiff")
    assert (root / "flight" / "bands" / "nir.tif").exists()
    assert not (root / "flight" / "flight.npz").exists()

    cfg = PipelineConfig(
        model_dir=root / "models" / "trained",
        flight_dir=root / "flight",
        weather_csv=root / "weather.csv",
        soil_csv=root / "soil.csv",
        output_dir=tmp_path / "runs" / "geo",
    )
    summary = run_pipeline(cfg)
    assert summary["n_tiles"] == 16
    assert summary["site"] == "bishoftu"

    # georeferencing must survive the GeoTIFF round-trip: tiles fall around Bishoftu
    fc = json.loads((Path(cfg.output_dir) / "risk_polygons.geojson").read_text())
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert 38.0 < lon < 40.0 and 8.0 < lat < 9.5


def test_field_report_generated(demo_env, tmp_path):
    """Every pipeline run must leave a printable field report next to the artifacts."""
    cfg = demo_env
    run_pipeline(cfg)
    report = Path(cfg.output_dir) / "field_report.html"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "WDED Field Risk Report" in text
    assert "bishoftu" in text
    assert "Top-risk tiles" in text
    # the demo always produces review tiles - the queue section must list them
    assert "Expert review queue" in text and "MC uncertainty" in text

    # standalone regeneration via the public API must work too
    from wded.report import generate_field_report

    alt = generate_field_report(cfg.output_dir, tmp_path / "alt_report.html")
    assert alt.exists() and alt.stat().st_size > 5_000


def test_cli_report(demo_env):
    cfg = demo_env
    run_pipeline(cfg)
    out = Path(cfg.output_dir) / "cli_report.html"
    rc = cli_main(["report", "--run", str(cfg.output_dir), "--out", str(out)])
    assert rc == 0
    assert out.exists()
