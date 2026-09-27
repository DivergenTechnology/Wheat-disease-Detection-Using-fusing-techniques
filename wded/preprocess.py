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

# GeoTIFF tag ids (TIFF 6.0 / GeoTIFF 1.1)
_TAG_MODEL_PIXEL_SCALE = 33550
_TAG_MODEL_TIEPOINT = 33922
_TAG_GEOKEY_DIRECTORY = 34735
_KEY_PROJECTED_CS = 3072  # ProjectedCSTypeGeoKey


@dataclass
class FlightData:
    """Preprocessed flight bundle consumed by the model and mapping stages."""

    tiles: np.ndarray      # (N, B, ps, ps) reflectance, float32
    indices: dict          # name -> (N,) per-tile index values
    centroids: np.ndarray  # (N, 2) world coordinates (x, y) of tile centres
    polygons: np.ndarray   # (N, 4, 2) world coordinates of tile corners
    meta: dict             # crs, geotransform (affine), capture_date, site_id, ...


def _normalize_tiff_datetime(raw) -> str | None:
    """Best-effort ISO date from a TIFF DateTime value ('YYYY:MM:DD HH:MM:SS')."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", "ignore")
    raw = str(raw).strip()
    if len(raw) < 10:
        return None
    return f"{raw[0:4]}-{raw[5:7]}-{raw[8:10]}"


def _read_geotiff_rasterio(tifs):
    """Read single-band GeoTIFFs with rasterio (heaviest, most complete reader)."""
    import rasterio  # optional heavy dependency

    by_name, profile = {}, None
    for t in tifs:
        name = t.stem.lower()
        with rasterio.open(t) as src:
            by_name[name] = src.read(1).astype(np.float32)
            if profile is None:
                crs = src.crs.to_string() if src.crs is not None else None
                if not crs:
                    log.warning("No CRS in %s - assuming EPSG:32637 (Bishoftu UTM 37N)", t.name)
                    crs = "EPSG:32637"
                profile = {
                    "crs": crs,
                    "geotransform": list(src.transform)[:6],
                    "capture_date": _normalize_tiff_datetime(src.tags().get("TIFFTAG_DATETIME")),
                }
    return by_name, profile


def _read_geotiff_tifffile(tifs):
    """Read single-band GeoTIFFs with tifffile (light reader, no GDAL needed).

    Reconstructs the GDAL-style affine geotransform from the ModelPixelScale /
    ModelTiepoint tags and the projected CRS from the GeoKey directory.  Only
    north-up, single-band, pixel-is-area rasters are supported (the usual
    orthomosaic export layout).
    """
    import tifffile  # optional light dependency

    by_name, profile = {}, None
    for t in tifs:
        name = t.stem.lower()
        with tifffile.TiffFile(t) as tif:
            page = tif.pages[0]
            by_name[name] = page.asarray().astype(np.float32)
            if profile is not None:
                continue
            try:
                scale = page.tags[_TAG_MODEL_PIXEL_SCALE].value
                tie = page.tags[_TAG_MODEL_TIEPOINT].value
            except KeyError as exc:
                raise ValueError(
                    f"{t.name} lacks GeoTIFF geo-referencing tags "
                    f"(ModelPixelScale/ModelTiepoint); export with CRS metadata"
                ) from exc
            sx, sy = float(scale[0]), float(scale[1])
            i, j, x_tie, y_tie = float(tie[0]), float(tie[1]), float(tie[3]), float(tie[4])
            # raster (i, j) -> world: x = x_tie + (i - i_tie)*sx, y = y_tie - (j - j_tie)*sy
            x0, y0 = x_tie - i * sx, y_tie + j * sy
            gt = [sx, 0.0, x0, 0.0, -sy, y0]
            crs = None
            gk = page.tags.get(_TAG_GEOKEY_DIRECTORY)
            if gk is not None:
                vals = gk.value
                for n in range(4, len(vals) - 3, 4):
                    if vals[n] == _KEY_PROJECTED_CS:
                        crs = f"EPSG:{int(vals[n + 3])}"
                        break
            if crs is None:
                log.warning("No projected CRS GeoKey in %s - assuming EPSG:32637 (Bishoftu UTM 37N)", t.name)
                crs = "EPSG:32637"
            dt_tag = page.tags.get("DateTime")
            profile = {"crs": crs, "geotransform": gt, "capture_date": _normalize_tiff_datetime(dt_tag.value if dt_tag else None)}
    return by_name, profile


def load_band_stack(flight_dir: Path, band_order):
    """Load the pre-processed band stack for one flight.

    Supported inputs (checked in order):
      1. ``flight.npz`` (one array per band) with ``flight_meta.json`` beside it.
      2. ``bands/<band>.tif`` GeoTIFFs exported from the orthomosaic, read with
         rasterio when available, otherwise with the lightweight tifffile
         (``pip install 'wded[geo]'``).
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
    tifs = sorted(bands_dir.glob("*.tif")) + sorted(bands_dir.glob("*.tiff")) if bands_dir.exists() else []
    if tifs:
        try:
            by_name, profile = _read_geotiff_rasterio(tifs)
        except ImportError:
            log.info("rasterio not available - falling back to tifffile GeoTIFF reader")
            try:
                by_name, profile = _read_geotiff_tifffile(tifs)
            except ImportError as exc:
                raise ImportError(
                    "Install a GeoTIFF reader to read GeoTIFF flights: "
                    "pip install 'wded[geo]' (tifffile, lightweight) or rasterio"
                ) from exc
        missing = [b for b in band_order if b not in by_name]
        if missing:
            raise ValueError(f"GeoTIFF band files missing for: {missing} (found {sorted(by_name)})")
        stack = np.stack([by_name[b] for b in band_order])
        meta = {
            "crs": profile["crs"],
            "geotransform": profile["geotransform"],
            "capture_date": profile.get("capture_date"),
            # site_id intentionally unset -> preprocess falls back to config.site_id
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
