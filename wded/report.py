"""Stage 7b: standalone field risk report (printable HTML) from a pipeline run.

Reads the artifacts a ``run_pipeline`` execution leaves in the output directory
(``tile_predictions.csv``, ``expert_review_queue.csv``, ``summary.json``) and
renders a self-contained HTML report - inline CSS, no JavaScript, print-ready
(A4).  Agronomists can archive it, mail it, or print it for the field scout
team without opening the dashboard.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_TIER_COLORS = {
    "critical": "#8C3B2E",
    "high": "#D4875A",
    "moderate": "#C9A227",
    "low": "#5F7A5A",
}

_DISEASE_LABELS = {
    "stem_rust": "Stem rust",
    "stripe_rust": "Stripe rust",
    "leaf_rust": "Leaf rust",
    "septoria": "Septoria (STB)",
    "fusarium": "Fusarium (FHB)",
}


def _esc(v) -> str:
    return html.escape(str(v))


def _pct(v) -> str:
    return f"{100.0 * float(v):.1f}%"


_CSS = """
:root{--graphite:#1A2330;--graphite2:#232E40;--orange:#D4875A;--ink:#1E2733;
--muted:#6B7686;--line:#DCE1E8;--paper:#FFFFFF;--wash:#F4F6F8}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',-apple-system,Helvetica,Arial,sans-serif;color:var(--ink);
background:var(--wash);padding:24px}
.page{max-width:900px;margin:0 auto;background:var(--paper);border:1px solid var(--line)}
header{background:var(--graphite);color:#E8ECF1;padding:26px 32px}
header h1{font-size:21px;font-weight:650;letter-spacing:.3px}
header .sub{color:#9AA7B8;font-size:12.5px;margin-top:6px}
header .badge{display:inline-block;background:var(--orange);color:#fff;font-size:11px;
padding:2px 10px;border-radius:10px;margin-left:8px;vertical-align:middle}
section{padding:22px 32px;border-bottom:1px solid var(--line)}
h2{font-size:13px;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);margin-bottom:14px}
.kpis{display:flex;gap:14px;flex-wrap:wrap}
.kpi{flex:1;min-width:150px;background:var(--wash);border:1px solid var(--line);
border-radius:6px;padding:14px 16px}
.kpi .num{font-size:26px;font-weight:700}
.kpi .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:3px}
.bar{display:flex;height:30px;border-radius:5px;overflow:hidden;font-size:12px;color:#fff}
.bar div{display:flex;align-items:center;justify-content:center;min-width:34px}
.legend{display:flex;gap:18px;margin-top:10px;font-size:12px;color:var(--muted);flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;background:var(--graphite2);color:#E8ECF1;padding:7px 10px;font-weight:600}
td{padding:7px 10px;border-bottom:1px solid var(--line)}
tr:nth-child(even) td{background:#FAFBFC}
.tier{display:inline-block;padding:1px 9px;border-radius:9px;color:#fff;font-size:11px;font-weight:600}
.flag{color:#8C3B2E;font-weight:600}
.ok{color:#5F7A5A;font-weight:600}
footer{padding:16px 32px;color:var(--muted);font-size:11.5px;line-height:1.6}
@media print{body{padding:0;background:#fff}.page{border:none}section{page-break-inside:avoid}}
"""


def _kpi(num: str, lbl: str) -> str:
    return f'<div class="kpi"><div class="num">{_esc(num)}</div><div class="lbl">{_esc(lbl)}</div></div>'


def _tier_chip(tier: str) -> str:
    color = _TIER_COLORS.get(str(tier), "#6B7686")
    return f'<span class="tier" style="background:{color}">{_esc(str(tier).upper())}</span>'


def _disease_summary(df: pd.DataFrame, diseases: list[str]) -> str:
    rows = []
    for d in diseases:
        p = df[f"p_{d}"]
        n_dom = int((df["dominant_disease"] == d).sum())
        n_alert = int((p >= 0.50).sum())
        rows.append(
            "<tr>"
            f"<td>{_esc(_DISEASE_LABELS.get(d, d))}</td>"
            f"<td>{p.mean():.3f}</td>"
            f"<td>{p.max():.3f}</td>"
            f"<td>{n_alert} of {len(df)}</td>"
            f"<td>{n_dom}</td>"
            "</tr>"
        )
    return (
        "<table><tr><th>Disease</th><th>Mean p</th><th>Max p</th>"
        "<th>Tiles p &ge; 0.50</th><th>Dominant tiles</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _top_risk_table(top: pd.DataFrame) -> str:
    rows = []
    for _, r in top.iterrows():
        note = str(r.get("recommendation", "") or "")
        if len(note) > 140:
            note = note[:137] + "..."
        rows.append(
            "<tr>"
            f"<td>{_esc(r['tile_id'])}</td>"
            f"<td>{_esc(_DISEASE_LABELS.get(r['dominant_disease'], r['dominant_disease']))}</td>"
            f"<td>{r['max_prob']:.3f}</td>"
            f"<td>{r['overall_risk']:.3f}</td>"
            f"<td>{_tier_chip(r['overall_tier'])}</td>"
            f"<td>{_esc(note)}</td>"
            "</tr>"
        )
    return (
        "<table><tr><th>Tile</th><th>Dominant disease</th><th>Model p</th>"
        "<th>Fused risk</th><th>Tier</th><th>Recommended action</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _review_table(review: pd.DataFrame) -> str:
    if review.empty:
        return '<p class="ok">No tiles in the ambiguous zone - no expert review required.</p>'
    rows = []
    for _, r in review.iterrows():
        rows.append(
            "<tr>"
            f"<td>{_esc(r['tile_id'])}</td>"
            f"<td>{_esc(_DISEASE_LABELS.get(r['dominant_disease'], r['dominant_disease']))}</td>"
            f"<td>{r['max_prob']:.3f}</td>"
            f"<td>{r['mean_uncertainty']:.3f}</td>"
            f"<td>{_tier_chip(r['overall_tier'])}</td>"
            "</tr>"
        )
    return (
        "<table><tr><th>Tile</th><th>Dominant disease</th><th>Model p</th>"
        "<th>MC uncertainty</th><th>Tier</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _tier_bar(tier_counts: dict, n_tiles: int) -> str:
    segs, legend = [], []
    for tier in ("critical", "high", "moderate", "low"):
        c = int(tier_counts.get(tier, 0))
        if c == 0:
            continue
        width = max(100.0 * c / max(n_tiles, 1), 6.0)
        segs.append(
            f'<div style="width:{width:.1f}%;background:{_TIER_COLORS[tier]}" title="{tier}: {c}">{c}</div>'
        )
        legend.append(f'<span><i style="background:{_TIER_COLORS[tier]}"></i>{tier}: {c}</span>')
    bar = '<div class="bar">' + "".join(segs) + "</div>" if segs else '<p>No tiles.</p>'
    return bar + '<div class="legend">' + "".join(legend) + "</div>"


def generate_field_report(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Render the field risk report for one pipeline run; returns the HTML path."""
    run_dir = Path(run_dir)
    preds_path = run_dir / "tile_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} not found - run the pipeline first")

    df = pd.read_csv(preds_path)
    review_path = run_dir / "expert_review_queue.csv"
    review = pd.read_csv(review_path) if review_path.exists() else df.iloc[0:0]
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    diseases = [c[2:] for c in df.columns if c.startswith("p_") and not c.startswith("p_spec")]
    n_tiles = len(df)
    tier_counts = df["overall_tier"].value_counts().to_dict()
    n_alert = int(df["overall_tier"].isin(["high", "critical"]).sum())
    mean_risk = float(df["overall_risk"].mean())

    site = summary.get("site", "unknown site")
    flight_date = summary.get("flight_date") or "n/a"
    generated = summary.get("generated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    fusion = summary.get("fusion_weights", {"model": 0.5, "weather": 0.3, "soil": 0.2})

    title_alert = " ACTION REQUIRED" if n_alert else ""
    top = df.sort_values("overall_risk", ascending=False).head(8)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>WDED Field Risk Report - {_esc(site)} - {_esc(flight_date)}</title>
<style>{_CSS}</style></head><body>
<div class="page">
<header>
  <h1>WDED Field Risk Report<span class="badge">{_esc(site)}</span></h1>
  <div class="sub">Flight date {_esc(flight_date)} &middot; {n_tiles} tiles &middot;
  generated {_esc(generated)} &middot;{_esc(title_alert)}</div>
</header>

<section>
  <h2>Field summary</h2>
  <div class="kpis">
    {_kpi(n_tiles, "tiles analysed")}
    {_kpi(f"{mean_risk:.3f}", "mean fused risk")}
    {_kpi(n_alert, "high / critical tiles")}
    {_kpi(len(review), "expert review queue")}
  </div>
</section>

<section>
  <h2>Tier distribution</h2>
  {_tier_bar(tier_counts, n_tiles)}
</section>

<section>
  <h2>Disease pressure by pathogen</h2>
  {_disease_summary(df, diseases)}
</section>

<section>
  <h2>Top-risk tiles</h2>
  {_top_risk_table(top)}
</section>

<section>
  <h2>Expert review queue ({len(review)} tile{'s' if len(review) != 1 else ''})</h2>
  {_review_table(review)}
  <p style="margin-top:10px;font-size:11.5px;color:var(--muted)">
  Tiles enter this queue when MC-Dropout uncertainty exceeds the threshold or the
  maximum disease probability sits in the open-set ambiguous zone [0.35, 0.55).</p>
</section>

<footer>
  Fusion weights: model {fusion.get('model', 0.5)} &middot; weather {fusion.get('weather', 0.3)} &middot;
  soil {fusion.get('soil', 0.2)}. Risk = weighted blend of SSCNN disease probabilities,
  lagged infection-window weather priors and soil suitability, with asymmetric-cost
  tier escalation (missed infection &asymp; 5&times; false alarm).<br/>
  Generated by WDED (Wheat Disease Early Detection) &middot; decision-support only -
  confirm with ground scouting before chemical intervention.
</footer>
</div>
</body></html>"""

    if out_path is None:
        out_path = run_dir / "field_report.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
