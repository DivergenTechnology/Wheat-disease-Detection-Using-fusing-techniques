"""Tests for the v0.3.0 features: trend analysis, demo overrides, enriched recommendations."""
import json
from pathlib import Path

import pandas as pd
import pytest

from wded.cli import main as cli_main
from wded.config import PipelineConfig
from wded.demo import generate_sample_data
from wded.pipeline import run_pipeline
from wded.recommendation import TIER_RESCOUT_DAYS, generate_recommendation
from wded.trend import STABLE_EPS, build_trend, load_run


def _write_run(out_dir: Path, date: str, tiles: dict) -> Path:
    """Fabricate a minimal pipeline output dir (summary.json + tile_predictions.csv)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(tiles)
    mean_risk = sum(v["risk"] for v in tiles.values()) / n
    tier_counts = {"critical": 0, "high": 0, "moderate": 0, "low": 0}
    for v in tiles.values():
        tier_counts[v["tier"]] += 1
    (out_dir / "summary.json").write_text(json.dumps({
        "site": "bishoftu",
        "flight_date": date,
        "generated_at": f"{date}T10:00:00+00:00",
        "n_tiles": n,
        "tier_counts": tier_counts,
        "mean_overall_risk": mean_risk,
        "expert_review_count": sum(1 for v in tiles.values() if v["tier"] == "high"),
    }))
    rows = [
        {
            "tile_id": tid,
            "overall_risk": v["risk"],
            "overall_tier": v["tier"],
            "dominant_disease": v.get("disease", "septoria"),
            "mean_uncertainty": 0.05,
        }
        for tid, v in tiles.items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "tile_predictions.csv", index=False)
    return out_dir


def test_demo_date_override_flows_through(tmp_path):
    root = tmp_path / "inputs"
    info = generate_sample_data(root, seed=42, capture_date="2026-10-02")
    assert info["days"] == 60

    meta = json.loads((root / "flight" / "flight_meta.json").read_text())
    assert meta["capture_date"] == "2026-10-02"

    weather = pd.read_csv(root / "weather.csv")
    assert weather["date"].iloc[-1] == "2026-10-02"  # 60-day window ends on capture date


def test_demo_hotspot_scale_changes_bands(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_sample_data(a, seed=42, fmt="npz")
    generate_sample_data(b, seed=42, fmt="npz", hotspot_scale=1.3)

    import numpy as np

    na = np.load(a / "flight" / "flight.npz")
    nb = np.load(b / "flight" / "flight.npz")
    assert not np.allclose(na["nir"], nb["nir"]), "hotspot scaling must change the rasters"
    # healthy background is largely unaffected (blob amplitude only)
    assert abs(float(na["nir"][0, 0]) - float(nb["nir"][0, 0])) < 0.05


def test_recommendation_urgency_and_rescout():
    fused = pd.DataFrame({
        "overall_tier": ["critical", "high", "moderate", "low"],
        "dominant_disease": ["stem_rust", "stripe_rust", "septoria", None],
    })
    out = generate_recommendation(fused)
    assert "urgency" in out.columns and "rescout_days" in out.columns
    assert out.loc[0, "urgency"].startswith("immediate")
    assert out.loc[0, "rescout_days"] == TIER_RESCOUT_DAYS["critical"] == 3
    assert out.loc[1, "rescout_days"] == 5
    assert out.loc[3, "urgency"].startswith("routine")
    assert "race typing" in out.loc[0, "disease_note"]  # disease notes still mapped


def test_trend_build_and_classification(tmp_path):
    d1 = _write_run(tmp_path / "d1", "2026-09-04", {
        "T0001": {"risk": 0.30, "tier": "moderate"},
        "T0002": {"risk": 0.52, "tier": "high"},
        "T0003": {"risk": 0.40, "tier": "moderate"},
    })
    d2 = _write_run(tmp_path / "d2", "2026-09-11", {
        "T0001": {"risk": 0.40, "tier": "moderate"},
        "T0002": {"risk": 0.50, "tier": "high"},
        "T0003": {"risk": 0.40, "tier": "moderate"},
    })
    d3 = _write_run(tmp_path / "d3", "2026-09-18", {
        "T0001": {"risk": 0.55, "tier": "high"},   # +0.25 -> worsening
        "T0002": {"risk": 0.38, "tier": "moderate"},  # -0.14 -> improving
        "T0003": {"risk": 0.40, "tier": "moderate"},  # +0.00 -> stable
        "T0009": {"risk": 0.33, "tier": "moderate"},  # new tile
    })

    trend = build_trend([d2, d1, d3])  # deliberately out of order
    assert trend["n_dates"] == 3
    assert trend["first_date"] == "2026-09-04" and trend["last_date"] == "2026-09-18"
    assert [s["date"] for s in trend["field_series"]] == ["2026-09-04", "2026-09-11", "2026-09-18"]

    t = trend["tiles"]
    assert t["T0001"]["trend"] == "worsening" and t["T0001"]["delta"] == pytest.approx(0.25)
    assert t["T0002"]["trend"] == "improving"
    assert abs(t["T0003"]["delta"]) <= STABLE_EPS and t["T0003"]["trend"] == "stable"
    assert t["T0009"]["trend"] == "new" and t["T0009"]["delta"] is None

    counts = trend["tile_trend_counts"]
    assert counts == {"worsening": 1, "stable": 1, "improving": 1, "new": 1}

    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "does_not_exist")


def test_trend_cli_writes_and_publishes(tmp_path):
    d1 = _write_run(tmp_path / "d1", "2026-09-04", {"T1": {"risk": 0.30, "tier": "moderate"}})
    d2 = _write_run(tmp_path / "d2", "2026-09-11", {"T1": {"risk": 0.45, "tier": "high"}})
    out = tmp_path / "trend" / "trend.json"
    pub = tmp_path / "publish"

    rc = cli_main(["trend", "--runs", str(d1), str(d2),
                   "--out", str(out), "--publish-to", str(pub)])
    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["tiles"]["T1"]["trend"] == "worsening"
    assert (pub / "trend.json").exists()


def test_geojson_properties_carry_urgency(tmp_path):
    root = tmp_path / "inputs"
    generate_sample_data(root, seed=42)
    cfg = PipelineConfig(
        model_dir=root / "models" / "trained",
        flight_dir=root / "flight",
        weather_csv=root / "weather.csv",
        soil_csv=root / "soil.csv",
        output_dir=tmp_path / "runs" / "demo",
        mc_passes=6,
    )
    run_pipeline(cfg)
    fc = json.loads((cfg.output_dir / "risk_polygons.geojson").read_text())
    p0 = fc["features"][0]["properties"]
    assert "urgency" in p0 and "rescout_days" in p0
    assert p0["rescout_days"] in (3, 5, 7, 14)
