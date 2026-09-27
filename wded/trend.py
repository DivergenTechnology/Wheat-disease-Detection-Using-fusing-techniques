"""Multi-date risk trend analysis across sequential pipeline runs.

Reads one or more pipeline output directories (each containing ``summary.json``
and ``tile_predictions.csv``), orders them by flight date and produces:

- a field-level series (mean risk, tier counts, review queue size per date)
- per-tile series with a risk delta and a coarse trend classification
  (worsening / stable / improving / new), where ``delta`` is the change in
  fused risk between the first and last flight

The JSON artifact feeds the dashboard trend panel and per-tile sparklines.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# |risk delta| below this is treated as stable (fused risk is a 0-1 score;
# +/-0.02 is within run-to-run noise of the MC-Dropout inference).
STABLE_EPS = 0.02


def _classify(delta: float) -> str:
    if delta > STABLE_EPS:
        return "worsening"
    if delta < -STABLE_EPS:
        return "improving"
    return "stable"


def load_run(run_dir: str | Path) -> dict:
    """Load one pipeline run directory into a compact record."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    tiles_path = run_dir / "tile_predictions.csv"
    if not summary_path.exists() or not tiles_path.exists():
        raise FileNotFoundError(
            f"{run_dir} is not a pipeline output (needs summary.json + tile_predictions.csv)"
        )
    summary = json.loads(summary_path.read_text())
    tiles = pd.read_csv(tiles_path)
    return {
        "dir": str(run_dir),
        "date": str(summary.get("flight_date") or summary.get("generated_at", "")[:10]),
        "site": summary.get("site"),
        "mean_risk": float(summary.get("mean_overall_risk", 0.0)),
        "tier_counts": summary.get("tier_counts", {}),
        "expert_review_count": int(summary.get("expert_review_count", 0)),
        "tiles": {
            str(r["tile_id"]): {
                "risk": float(r["overall_risk"]),
                "tier": str(r["overall_tier"]),
                "disease": str(r["dominant_disease"]),
            }
            for _, r in tiles.iterrows()
        },
    }


def build_trend(run_dirs: list[str | Path]) -> dict:
    """Build the multi-date trend artifact from an ordered list of run dirs."""
    runs = [load_run(p) for p in run_dirs]
    if not runs:
        raise ValueError("no runs supplied")
    runs.sort(key=lambda r: r["date"])

    field_series = [
        {
            "date": r["date"],
            "mean_risk": round(r["mean_risk"], 4),
            "tier_counts": r["tier_counts"],
            "expert_review_count": r["expert_review_count"],
        }
        for r in runs
    ]
    field_delta = round(field_series[-1]["mean_risk"] - field_series[0]["mean_risk"], 4)

    # Tile series are anchored on the most recent run (the current map extent).
    tiles_out: dict[str, dict] = {}
    counts = {"worsening": 0, "stable": 0, "improving": 0, "new": 0}
    for tid, last in runs[-1]["tiles"].items():
        series = [(r["date"], r["tiles"][tid]) for r in runs if tid in r["tiles"]]
        if len(series) < 2:
            tiles_out[tid] = {
                "dates": [d for d, _ in series],
                "risks": [round(v["risk"], 4) for _, v in series],
                "delta": None,
                "trend": "new",
                "last_tier": last["tier"],
                "last_disease": last["disease"],
            }
            counts["new"] += 1
            continue
        first = series[0][1]
        delta = round(last["risk"] - first["risk"], 4)
        trend = _classify(delta)
        tiles_out[tid] = {
            "dates": [d for d, _ in series],
            "risks": [round(v["risk"], 4) for _, v in series],
            "delta": delta,
            "trend": trend,
            "last_tier": last["tier"],
            "last_disease": last["disease"],
        }
        counts[trend] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site": runs[-1]["site"],
        "n_dates": len(runs),
        "first_date": field_series[0]["date"],
        "last_date": field_series[-1]["date"],
        "field_series": field_series,
        "field_delta": field_delta,
        "field_trend": _classify(field_delta),
        "tile_trend_counts": counts,
        "tiles": tiles_out,
    }


def write_trend(trend: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trend, indent=2))
    return out
