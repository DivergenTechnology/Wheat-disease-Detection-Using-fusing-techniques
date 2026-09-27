"""End-to-end orchestration of the WDED inference pipeline.

Stage chain (agreed design):
  load_model -> preprocess_new_flight -> align_weather_soil -> run_sscnn_inference
             -> fuse_decision -> generate_recommendation -> build_map_overlay
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .alignment import align_weather_soil
from .config import DISEASES, PipelineConfig
from .fusion import fuse_decision
from .inference import run_sscnn_inference
from .mapping import build_map_overlay
from .models import load_model
from .preprocess import preprocess_new_flight
from .recommendation import generate_recommendation

log = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig, publish_to: str | None = None) -> dict:
    stages = []

    def stage(name, fn):
        log.info("[stage] %s", name)
        out = fn()
        stages.append(name)
        return out

    flight = stage("preprocess_new_flight", lambda: preprocess_new_flight(config))
    model = stage("load_model", lambda: load_model(config))
    aligned = stage("align_weather_soil", lambda: align_weather_soil(flight, config))
    preds = stage(
        "run_sscnn_inference", lambda: run_sscnn_inference(model, flight, aligned, config)
    )
    fused = stage("fuse_decision", lambda: fuse_decision(preds, aligned, config))
    fused = stage("generate_recommendation", lambda: generate_recommendation(fused))

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fused.to_csv(out_dir / "tile_predictions.csv", index=False)
    review = fused[fused["needs_expert_review"]].sort_values("overall_risk", ascending=False)
    review.to_csv(out_dir / "expert_review_queue.csv", index=False)
    stage("build_map_overlay", lambda: build_map_overlay(flight, fused, out_dir))

    tier_counts = fused["overall_tier"].value_counts().to_dict()
    summary = {
        "site": flight.meta.get("site_id", config.site_id),
        "flight_date": flight.meta.get("capture_date"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tiles": int(len(fused)),
        "tier_counts": {k: int(tier_counts.get(k, 0)) for k in ("critical", "high", "moderate", "low")},
        "mean_overall_risk": round(float(fused["overall_risk"].mean()), 4),
        "expert_review_count": int(len(review)),
        "dominant_disease_counts": {d: int((fused["dominant_disease"] == d).sum()) for d in DISEASES},
        "fusion_weights": {
            "model": config.fusion.model,
            "weather": config.fusion.weather,
            "soil": config.fusion.soil,
        },
        "thresholds": {
            "uncertainty": config.uncertainty_threshold,
            "open_set": config.open_set_threshold,
            "open_set_lower": config.open_set_lower,
        },
        "weather_meta": aligned.weather_meta,
        "stages_completed": stages,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Pipeline finished: tiers=%s review=%d", json.dumps(summary["tier_counts"]), len(review))

    if publish_to:
        pub = Path(publish_to)
        pub.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_dir / "risk_polygons.geojson", pub / "latest_risk.geojson")
        shutil.copy(out_dir / "attention_heatmap.geojson", pub / "latest_attention.geojson")
        shutil.copy(out_dir / "summary.json", pub / "summary.json")
        log.info("Published dashboard artifacts to %s", pub)

    return summary
