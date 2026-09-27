"""Stage 7: build GeoJSON overlays (risk polygons + attention heatmap) for the dashboard."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _to_lonlat(polys, crs):
    """Project polygon corners to WGS84 for GeoJSON; fall back to raw coordinates."""
    try:
        from pyproj import Transformer

        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        flat = np.asarray(polys, dtype=float).reshape(-1, 2)
        lonlat = np.column_stack(tr.transform(flat[:, 0], flat[:, 1]))
        return lonlat.reshape(np.asarray(polys).shape)
    except Exception as exc:  # pragma: no cover - depends on optional pyproj
        log.warning("Coordinate transform failed (%s) - writing raw coordinates", exc)
        return np.asarray(polys, dtype=float)


def _feature(poly_lonlat, props):
    ring = [[float(x), float(y)] for x, y in poly_lonlat]
    ring.append(list(ring[0]))  # GeoJSON rings must be closed
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": props,
    }


def build_map_overlay(flight, fused, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lonlat = _to_lonlat(flight.polygons, flight.meta.get("crs", "EPSG:4326"))

    feats = []
    for i, row in fused.reset_index(drop=True).iterrows():
        props = {
            "tile_id": row["tile_id"],
            "risk": round(float(row["overall_risk"]), 3),
            "tier": row["overall_tier"],
            "dominant_disease": row["dominant_disease"],
            "dominant_risk_driver": row["dominant_risk_driver"],
            "model_prob": round(float(row["max_prob"]), 3),
            "uncertainty": round(float(row["mean_uncertainty"]), 4),
            "attention": round(float(row["attention"]), 4),
            "needs_expert_review": bool(row["needs_expert_review"]),
            "recommendation": row["recommendation"],
            "disease_note": row["disease_note"],
        }
        feats.append(_feature(lonlat[i], props))

    risk_fc = {"type": "FeatureCollection", "features": feats}
    (out_dir / "risk_polygons.geojson").write_text(json.dumps(risk_fc))

    heat = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lonlat[i][0][0]), float(lonlat[i][0][1])],
                },
                "properties": {
                    "tile_id": row["tile_id"],
                    "weight": round(float(row["attention"]), 4),
                    "tier": row["overall_tier"],
                },
            }
            for i, row in fused.reset_index(drop=True).iterrows()
        ],
    }
    (out_dir / "attention_heatmap.geojson").write_text(json.dumps(heat))
    log.info("Map overlays written: %d polygons, %d attention points", len(feats), len(heat["features"]))
    return risk_fc, heat
