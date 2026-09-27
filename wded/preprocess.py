"""Stages 1-2: multispectral flight loading, tiling and spectral-index extraction."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)

EPS = 1e-6


@dataclass
class FlightData:
    """Preprocessed flight bundle consumed by the model and mapping stages."""

    tiles: np.ndarray      # (N, B, ps, ps) reflectance, float32
    indices: dict          # name -> (N,) per-tile index values
    centroids: np.ndarray  # (N, 2) world coordinates (x, y) of tile centres
    polygons: np.ndarray   # (N, 4, 2) world coordinates of tile corners
    meta: dict             # crs, geotransform (affine), capture_date, site_id, ...


def load_band_stack(flight_dir: Path, band_order):
    """Load the pre-processed band stack for one flight.

    Supported inputs (checked in order):
      1. ``flight.npz`` (one array per band) with ``flight_meta.json`` beside it.
      2. ``bands/<band>.tif`` GeoTIFFs exported from the orthomosaic (needs rasterio).
    """
    flight_dir = Path(flight_dir)
    npz = flight_dir / "flight.npz"
    if npz.exists():
        with np.load(npz) as z:
            missing = [b for b in band_order if b not in z.files]
            if missing:
                raise ValueError(f"flight.npz missing bands: {missing}")
            stack = np.stack([np.asarray(z[b], dtype=np.float32) for b in band_order])
        meta_path = flight_dir / "flight_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("flight_meta.json is required next to flight.npz")
        meta = json.loads(meta_path.read_text())
        return stack, meta

    bands_dir = flight_dir / "bands"
    tifs = sorted(bands_dir.glob("*.tif")) if bands_dir.exists() else []
    if tifs:
        try:
            import rasterio  # optional heavy dependency
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install rasterio to read GeoTIFF flights: pip install 'wded[geo]'"
            ) from exc
        by_name, profile = {}, None
        for t in tifs:
            name = t.stem.lower()
            with rasterio.open(t) as src:
                by_name[name] = src.read(1).astype(np.float32)
                if profile is None:
                    profile = {"crs": src.crs.to_string(), "geotransform": list(src.transform)[:6]}
        missing = [b for b in band_order if b not in by_name]
        if missing:
            raise ValueError(f"GeoTIFF band files missing for: {missing} (found {sorted(by_name)})")
        stack = np.stack([by_name[b] for b in band_order])
        meta = {
            "crs": profile["crs"],
            "geotransform": profile["geotransform"],
            "capture_date": None,
            "site_id": flight_dir.parent.name,
        }
        return stack, meta

    raise FileNotFoundError(
        f"No flight data found under {flight_dir} (expected flight.npz or bands/*.tif)"
    )


def _tile_grid(shape2d, patch):
    h, w = shape2d
    if h % patch or w % patch:
        raise ValueError(f"image size {h}x{w} is not divisible by patch size {patch}")
    for r in range(h // patch):
        for c in range(w // patch):
            yield r, c


def pixel_to_world(gt, row, col):
    """Affine geotransform (a, b, c, d, e, f): x = a*col + b*row + c; y = d*col + e*row + f."""
    a, b, c, d, e, f = gt
    return a * col + b * row + c, d * col + e * row + f


def _glcm(band_tile, levels=8):
    """Compact GLCM texture features (horizontal offset, quantised to `levels`)."""
    q = np.clip((band_tile * levels).astype(np.int64), 0, levels - 1)
    a = q[:, :-1].ravel()
    b = q[:, 1:].ravel()
    P = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(P, (a, b), 1.0)
    s = P.sum()
    if s > 0:
        P /= s
    i, j = np.indices(P.shape)
    d = (i - j).astype(np.float64)
    contrast = float((d ** 2 * P).sum())
    homogeneity = float((P / (1.0 + d ** 2)).sum())
    energy = float(np.sqrt((P ** 2).sum()))
    return contrast, homogeneity, energy


def compute_indices(tiles, band_order):
    """Per-tile spectral indices: NDVI, NDRE, GNDVI, REIP (Guyot & Baret) and GLCM texture."""
    idx = {name: i for i, name in enumerate(band_order)}
    green, red = tiles[:, idx["green"]], tiles[:, idx["red"]]
    redge, nir = tiles[:, idx["red_edge"]], tiles[:, idx["nir"]]

    ndvi = (nir - red) / (nir + red + EPS)
    ndre = (nir - redge) / (nir + redge + EPS)
    gndvi = (nir - green) / (nir + green + EPS)
    reip = 700.0 + 40.0 * (((red + nir) / 2.0 - redge) / (nir - red + EPS))

    n = len(tiles)
    glcm_c = np.empty(n)
    glcm_h = np.empty(n)
    glcm_e = np.empty(n)
    for i in range(n):
        glcm_c[i], glcm_h[i], glcm_e[i] = _glcm(green[i])

    return {
        "ndvi": ndvi.mean(axis=(1, 2)).astype(np.float32),
        "ndre": ndre.mean(axis=(1, 2)).astype(np.float32),
        "gndvi": gndvi.mean(axis=(1, 2)).astype(np.float32),
        "reip": reip.mean(axis=(1, 2)).astype(np.float32),
        "glcm_contrast": glcm_c.astype(np.float32),
        "glcm_homogeneity": glcm_h.astype(np.float32),
        "glcm_energy": glcm_e.astype(np.float32),
    }


def preprocess_new_flight(config: PipelineConfig) -> FlightData:
    """Stage 1-2 entry point: load, tile and index one flight."""
    stack, meta = load_band_stack(config.flight_dir, config.band_order)
    log.info("Loaded band stack %s from %s", stack.shape, config.flight_dir)

    gt = meta.get("geotransform")
    if gt is None or len(gt) != 6:
        raise ValueError("flight metadata must include a 6-element affine 'geotransform'")
    ps = config.patch_size
    h, w = stack.shape[1:]

    tiles, cents, polys = [], [], []
    for r, c in _tile_grid((h, w), ps):
        tiles.append(stack[:, r * ps:(r + 1) * ps, c * ps:(c + 1) * ps])
        x0, y0 = pixel_to_world(gt, r * ps, c * ps)
        x1, y1 = pixel_to_world(gt, r * ps + ps, c * ps + ps)
        cents.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        # corners: SW, SE, NE, NW (image row grows southwards for north-up rasters)
        xa, ya = pixel_to_world(gt, r * ps + ps, c * ps)
        xb, yb = pixel_to_world(gt, r * ps + ps, c * ps + ps)
        xc, yc = pixel_to_world(gt, r * ps, c * ps + ps)
        polys.append([(xa, ya), (xb, yb), (xc, yc), (x0, y0)])

    tiles = np.stack(tiles).astype(np.float32)
    indices = compute_indices(tiles, config.band_order)

    meta = dict(meta)
    meta.setdefault("site_id", config.site_id)
    if not meta.get("capture_date"):
        meta["capture_date"] = config.flight_date
    return FlightData(
        tiles=tiles,
        indices=indices,
        centroids=np.asarray(cents, dtype=float),
        polygons=np.asarray(polys, dtype=float),
        meta=meta,
    )
