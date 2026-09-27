"""Generate a fully synthetic flight + weather + soil bundle for demos and tests.

Simulates a Mavic 3M 4-band capture over a 192 m x 192 m block near Bishoftu
(UTM 37N) with three disease "hotspots", each with a distinct spectral
signature so per-disease responses can be told apart:

- stem rust   -> canopy collapse: NIR crashes, red jumps (NDVI collapse)
- stripe rust -> red-edge shift: red-edge band rises, REIP blue-shifts (NDRE drop)
- septoria    -> necrotic lesion texture: speckle on visible bands (GLCM contrast)

plus 60 days of station weather and a single soil record.
"""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

SITE_META = {
    "site_id": "bishoftu",
    "crs": "EPSG:32637",
    # affine geotransform: x = a*col + b*row + c ; y = d*col + e*row + f
    "geotransform": [1.5, 0.0, 495000.0, 0.0, -1.5, 967000.0],
    "capture_date": "2026-09-18",
}


def _write_geotiff_bands(flight: Path, bands: dict, meta: dict) -> None:
    """Write one single-band GeoTIFF per band with full geo-referencing tags."""
    import tifffile

    bands_dir = flight / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)
    gt = meta["geotransform"]                      # [a, b, c, d, e, f] GDAL order
    sx, sy = float(gt[0]), float(-gt[4])
    x0, y0 = float(gt[2]), float(gt[5])
    epsg = int(meta["crs"].split(":")[-1]) if str(meta.get("crs", "")).startswith("EPSG:") else 32637
    extratags = [
        # (tag_id, dtype, count, value, write_count) - 12=DOUBLE, 3=SHORT
        (33550, 12, 3, (sx, sy, 0.0), True),                               # ModelPixelScale
        (33922, 12, 6, (0.0, 0.0, 0.0, x0, y0, 0.0), True),                # ModelTiepoint
        # GeoKeyDirectory: header (v1.1.0, 3 keys) + GTModelType=Projected,
        # GTRasterType=PixelIsArea, ProjectedCSType=<epsg>
        (34735, 3, 16,
         (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg), True),
    ]
    try:
        y, m, d = (int(v) for v in str(meta.get("capture_date", "2026-09-18")).split("-"))
        dt = datetime(y, m, d, 10, 30, 0)
    except ValueError:
        dt = datetime(2026, 9, 18, 10, 30, 0)
    for name, arr in bands.items():
        tifffile.imwrite(
            bands_dir / f"{name}.tif",
            np.ascontiguousarray(arr, dtype=np.float32),
            photometric="minisblack",
            metadata=None,
            datetime=dt,
            extratags=extratags,
        )


def generate_sample_data(
    root: Path,
    seed: int = 42,
    fmt: str = "npz",
    capture_date: str | None = None,
    hotspot_scale: float = 1.0,
) -> dict:
    """Generate the synthetic bundle.

    ``capture_date`` overrides the flight date (metadata, GeoTIFF DateTime and
    the 60-day weather window); ``hotspot_scale`` scales the disease hotspots
    so multi-date demo scenarios can simulate an epidemic ramp.
    """
    meta = dict(SITE_META)
    if capture_date:
        date.fromisoformat(capture_date)  # validate early, fail fast
        meta["capture_date"] = capture_date

    root = Path(root)
    flight = root / "flight"
    flight.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:128, 0:128]

    def blob(cx, cy, r):
        return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)))

    stem = 0.65 * hotspot_scale * blob(38, 44, 20)
    stripe = 0.55 * hotspot_scale * blob(88, 90, 16)
    sept = 0.45 * hotspot_scale * blob(70, 30, 12)
    disease = np.clip(stem + stripe + sept, 0, 1)
    healthy = 1.0 - disease

    def noise():
        return rng.normal(0, 0.012, (128, 128))

    # necrotic lesion speckle (septoria): high-frequency texture on visible bands
    speckle = rng.normal(0, 1, (128, 128)) * 0.06 * sept

    bands = {
        "green": np.clip(
            0.10 + 0.04 * disease + 0.02 * stem + 0.02 * stripe + 0.02 * sept + speckle + noise(),
            0.01, 0.95,
        ),
        "red": np.clip(
            0.06 + 0.08 * disease + 0.10 * stem + 0.02 * stripe + 0.03 * sept + noise(),
            0.01, 0.95,
        ),
        "red_edge": np.clip(
            0.28 - 0.10 * disease - 0.05 * stem + 0.16 * stripe - 0.02 * sept + noise(),
            0.01, 0.95,
        ),
        "nir": np.clip(
            0.55 - 0.25 * disease - 0.18 * stem - 0.05 * stripe - 0.04 * sept + noise(),
            0.01, 0.95,
        ),
    }
    if fmt == "geotiff":
        _write_geotiff_bands(flight, bands, meta)
    else:
        np.savez_compressed(flight / "flight.npz", **bands)
        (flight / "flight_meta.json").write_text(json.dumps(meta, indent=2))

    model_dir = root / "models" / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "mock_model.json").write_text(json.dumps({"type": "mock", "seed": seed}))

    # --- 60 days of station weather ending on the capture date -----------------
    end = date.fromisoformat(meta["capture_date"])
    rows = []
    tmean = 19.0
    for i in range(60):
        d = end - timedelta(days=59 - i)
        wet = rng.random() < 0.10
        rain = float(rng.uniform(2, 8) if wet else 0.0)
        tmean = float(np.clip(tmean + rng.normal(0, 0.4), 14.0, 23.0))
        lwh = float(np.clip((4 + rain / 2) if wet else rng.uniform(0, 2), 0, 10))
        rh = float(np.clip(52 + rain * 0.9 + rng.normal(0, 5), 30, 100))
        rows.append(
            {
                "date": d.isoformat(),
                "temp_mean": round(tmean, 1),
                "temp_min": round(tmean - rng.uniform(3, 6), 1),
                "temp_max": round(tmean + rng.uniform(4, 8), 1),
                "rh_mean": round(rh, 1),
                "leaf_wetness_hours": round(lwh, 1),
                "rain_mm": round(rain, 1),
                "wind_kph": round(float(rng.uniform(4, 14)), 1),
            }
        )
    with open(root / "weather.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (root / "soil.csv").write_text(
        "site_id,ph,drainage,texture,organic_matter_pct,previous_cereal\n"
        "bishoftu,6.7,moderate,clay_loam,2.1,true\n"
    )
    return {"root": str(root), "bands": list(bands), "days": len(rows)}
