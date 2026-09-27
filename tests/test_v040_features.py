"""Tests for the v0.4.0 field-data submission module (client-side analysis).

The dashboard's field submission tab (docs/field_submission.js) mirrors the
Python pipeline decision logic in the browser. These tests pin that parity:
same infection windows, same fusion weights, same tiering/escalation, and the
required UI wiring in docs/index.html.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from wded.alignment import INFECTION_WINDOWS
from wded.config import FusionWeights, RISK_TIERS

DOCS = Path(__file__).resolve().parents[1] / "docs"
INDEX = DOCS / "index.html"
JS = DOCS / "field_submission.js"


# --------------------------------------------------------------------------
# docs wiring
# --------------------------------------------------------------------------


def test_submission_module_files_exist():
    assert JS.exists(), "docs/field_submission.js missing"
    html = INDEX.read_text(encoding="utf-8")
    assert "./field_submission.js" in html
    for marker in (
        'id="tabSubmit"',
        'id="fsForm"',
        'id="fs_temp"',
        'id="fs_rh"',
        'id="fs_lwh"',
        'id="fs_rain"',
        'id="fs_ph"',
        'id="fs_drainage"',
        'id="fs_green"',
        'id="fs_red_edge"',
        'id="fs_nir"',
        'id="fs_result"',
        'id="fs_history_body"',
        "wded:focus-tile",
    ):
        assert marker in html, f"dashboard markup missing: {marker}"


def test_submission_module_has_required_form_fields_in_js():
    src = JS.read_text(encoding="utf-8")
    for token in (
        "fs_temp", "fs_rh", "fs_lwh", "fs_rain",
        "fs_ph", "fs_drainage", "fs_prev_cereal",
        "fs_green", "fs_red", "fs_red_edge", "fs_nir",
        "localStorage", "wded.field_submission/1",
        "TIER_ACTIONS", "TIER_URGENCY", "TIER_RESCOUT_DAYS", "DISEASE_NOTES",
    ):
        assert token in src, f"engine/token missing: {token}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_submission_js_and_inline_dashboard_js_syntax():
    import tempfile

    subprocess.run(["node", "--check", str(JS)], check=True)
    html = INDEX.read_text(encoding="utf-8")
    inline = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert inline, "no inline dashboard script found"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
        tmp.write(inline[-1])
        tmp_path = tmp.name
    try:
        subprocess.run(["node", "--check", tmp_path], check=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# formula parity: JS module vs Python pipeline
# --------------------------------------------------------------------------


def test_js_infection_windows_match_python():
    src = JS.read_text(encoding="utf-8")
    for disease, cfg in INFECTION_WINDOWS.items():
        block = re.search(
            rf"{disease}:\s*\{{([^}}]*)\}}", src
        )
        assert block, f"INFECTION_WINDOWS[{disease}] missing in JS module"
        body = " ".join(block.group(1).split())  # normalise JS alignment spacing
        assert f"t_lo: {cfg['t_lo']}" in body
        assert f"t_hi: {cfg['t_hi']}" in body
        assert f"leaf_wetness_h: {cfg['leaf_wetness_h']}" in body
        assert f"lag_days: {cfg['lag_days']}" in body


def test_js_fusion_weights_and_tiers_match_python():
    src = JS.read_text(encoding="utf-8")
    fw = FusionWeights()
    assert f"WEIGHTS = {{ spectral: {fw.model}, weather: {fw.weather}, soil: {fw.soil} }}" in src
    for tier, thr in RISK_TIERS:
        if tier == "low":
            continue
        pattern = rf"risk >= {thr:.2f}\) return '{tier}'"
        assert re.search(pattern, src), f"tier boundary {tier}={thr} not mirrored"
    assert "ESCALATION_MARGIN = 0.05" in src


def test_js_recommendation_tables_match_python():
    from wded.recommendation import DISEASE_NOTES, TIER_ACTIONS, TIER_RESCOUT_DAYS, TIER_URGENCY

    src = JS.read_text(encoding="utf-8")
    for tier, text in TIER_ACTIONS.items():
        assert text in src, f"TIER_ACTIONS[{tier}] text drift"
    for tier, text in TIER_URGENCY.items():
        assert text in src, f"TIER_URGENCY[{tier}] text drift"
    for tier, days in TIER_RESCOUT_DAYS.items():
        assert re.search(rf"{tier}: {days}", src), f"TIER_RESCOUT_DAYS[{tier}] drift"
    for disease, note in DISEASE_NOTES.items():
        assert note in src, f"DISEASE_NOTES[{disease}] text drift"


def test_pyproject_version_bumped():
    """Version must be at least 0.4.0 (the field-submission release)."""
    pyproject = (DOCS.parent / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'version = "(\d+)\.(\d+)\.(\d+)"', pyproject)
    assert match, "pyproject version not found"
    version = tuple(int(p) for p in match.groups())
    assert version >= (0, 4, 0), f"version {version} regressed below 0.4.0"
