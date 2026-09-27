# WDED Inference Pipeline — Design Specification

This document records the agreed design implemented in `wded/`. It is the contract
between the research team (who train the model and collect flights) and this codebase.

## 1. Stage chain

```
load_model() -> preprocess_new_flight() -> align_weather_soil() -> run_sscnn_inference()
            -> fuse_decision() -> generate_recommendation() -> build_map_overlay()
```

| Function | Module | Responsibility |
|----------|--------|----------------|
| `load_model` | `wded/models/__init__.py` | Select the torch checkpoint adapter or deterministic mock; attach temperature + Grad-CAM config |
| `preprocess_new_flight` | `wded/preprocess.py` | Load 4/5-band reflectance (GeoTIFF or NPZ), tile into 32 px patches, compute NDVI, NDRE, REIP (Guyot & Baret), GNDVI, GLCM contrast/homogeneity/energy, retain georeferencing |
| `align_weather_soil` | `wded/alignment.py` | Station weather → per-disease lagged infection-window priors (temp suitability × wetness × rain); soil pH/drainage/residue → suitability score; daily scaled sequences for the LSTM branch |
| `run_sscnn_inference` | `wded/inference.py` | MC-Dropout (`mc_passes`), temperature calibration, per-tile mean/std probabilities, dominant disease, open-set check, attention |
| `fuse_decision` | `wded/fusion.py` | Weighted evidence fusion, tiering, asymmetric-cost escalation, expert-review flags |
| `generate_recommendation` | `wded/recommendation.py` | Tier- and disease-specific agronomic actions |
| `build_map_overlay` | `wded/mapping.py` | WGS84 GeoJSON risk polygons + attention heatmap points for the dashboard |

## 2. Feature engineering (per implementation plan)

- **Spectral**: NDVI, NDRE, REIP, GNDVI (canopy vigour / chlorophyll / red-edge position).
- **Spatial**: GLCM texture (contrast, homogeneity, energy) on the green band —
  lesion patterning pre-dates index depression.
- **Weather (lagged)**: infection windows per disease —
  stem rust 15–35 °C / 4 h wetness / 14 d lag; stripe rust 5–20 °C / 14 d;
  leaf rust 15–30 °C / 14 d; Septoria 10–25 °C / 8 h wetness / 21 d;
  Fusarium 20–30 °C / 6 h wetness / 10 d (anthesis-driven in the field protocol).
- **Interaction**: the fusion step multiplies no features; instead the model's weather
  branch receives the 21-day daily sequence, and the fusion weights encode trust.

## 3. Trust layer

1. **MC-Dropout uncertainty** — mean predictive std across `mc_passes` stochastic passes.
2. **Probability calibration** — temperature scaling from `calibration.json`.
3. **Open-set rejection** — max disease probability in `[0.35, 0.55)` is the ambiguous zone
   (⇒ review); `< 0.35` is confidently healthy; `≥ 0.55` is a confident disease call.
4. **Expert review** — flagged when `uncertainty > 0.08` or out-of-distribution;
   exported to `expert_review_queue.csv` sorted by descending risk.

## 4. Decision engine

```
risk_d  = 0.5 · calibrated model P(d)
        + 0.3 · lagged weather prior(d)
        + 0.2 · soil score
```

- Tiers: `critical ≥ 0.75`, `high ≥ 0.50`, `moderate ≥ 0.25`, else `low`.
- **Asymmetric cost**: FN:FP ≈ 5:1 ⇒ risk within `0.05` below a boundary is escalated
  one tier (e.g. 0.48 moderate → treated as high).
- Actions per tier (see `wded/recommendation.py`): critical → immediate fungicide +
  24 h confirmation + harvest segregation; high → 48 h scouting + spray prep + 5–7 d re-flight;
  moderate → 72 h re-flight + targeted scouting; low → routine cadence.

## 5. Outputs per run

| File | Content |
|------|---------|
| `tile_predictions.csv` | Per-tile probabilities, uncertainties, risks, tiers, recommendations |
| `expert_review_queue.csv` | Flagged tiles sorted by risk |
| `risk_polygons.geojson` | Georeferenced tile polygons with full decision metadata |
| `attention_heatmap.geojson` | Per-tile attention points for the heatmap layer |
| `summary.json` | Tier counts, mean risk, review count, weights, thresholds, provenance |

## 6. Extension points

- Real sensor data: `wded/preprocess.py` already accepts GeoTIFF orthomosaic bands (rasterio).
- Real model: `wded/models/torch_adapter.py` implements MC-Dropout + Grad-CAM for any
  module matching `forward(spectral, weather) -> logits (N, D)`.
- Tile-level microclimate: replace the station-level prior with gridded weather per tile
  in `align_weather_soil` — the interface (`AlignedCovariates`) is unchanged.
