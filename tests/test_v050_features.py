"""Tests for the v0.5.0 disease knowledge module + front-end redesign.

The knowledge base (docs/knowledge.js) must stay consistent with the Python
pipeline: one entry per target disease, infection-condition chips mirroring
wded/alignment.py INFECTION_WINDOWS, and correlation hooks from the risk-map
popups and the field-submission results into the guide.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from wded.alignment import INFECTION_WINDOWS
from wded.config import DISEASES

DOCS = Path(__file__).resolve().parents[1] / "docs"
INDEX = DOCS / "index.html"
KB = DOCS / "knowledge.js"
FS = DOCS / "field_submission.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_knowledge_js_syntax():
    subprocess.run(["node", "--check", str(KB)], check=True)


def test_knowledge_base_covers_all_target_diseases():
    src = KB.read_text(encoding="utf-8")
    for disease in DISEASES:
        assert re.search(rf"^\s+{disease}: \{{", src, flags=re.M), f"KB entry missing: {disease}"


def test_knowledge_entries_have_required_fields():
    src = KB.read_text(encoding="utf-8")
    for field in ("label:", "pathogen:", "severity:", "summary:", "symptoms:",
                  "conditions:", "spectral:", "riskFactors:", "management:",
                  "scoutingTip:"):
        assert src.count(field) >= len(DISEASES), f"field {field} not present for all diseases"


def test_knowledge_infection_conditions_match_python_windows():
    src = KB.read_text(encoding="utf-8")
    for disease, cfg in INFECTION_WINDOWS.items():
        entry = re.search(rf"{disease}: \{{(.*?)\n  \}}", src, flags=re.S)
        assert entry, f"KB entry missing for {disease}"
        cond = re.search(r"conditions: \{([^}]*)\}", entry.group(1))
        assert cond, f"conditions block missing for {disease}"
        body = cond.group(1)
        assert f"temp_lo: {cfg['t_lo']:g}" in body, f"{disease} temp_lo drift"
        assert f"temp_hi: {cfg['t_hi']:g}" in body, f"{disease} temp_hi drift"
        assert f"lag_days: {cfg['lag_days']:g}" in body, f"{disease} lag drift"
        wet = cfg["leaf_wetness_h"]
        # wetness strings carry the hours, e.g. '8 h+ leaf wetness' / '4 h leaf wetness / dew'
        assert re.search(rf"{wet:g}\s*h", body), f"{disease} leaf-wetness hours drift"


def test_index_redesigned_with_three_tabs_and_kb_wiring():
    html = INDEX.read_text(encoding="utf-8")
    for marker in (
        'data-tab="map"', 'data-tab="submit"', 'data-tab="knowledge"',
        'id="tabKnowledge"', 'id="kbView"',
        "./knowledge.js", "./field_submission.js",
        "wded:summary-loaded", "WDED_KB_OPEN", "kbLink(",
    ):
        assert marker in html, f"redesign marker missing: {marker}"
    # new light theme present, old dark palette gone
    assert "--brand:#3D6B35" in html and "--gold:#C9971C" in html
    assert "--graphite" not in html, "old dark palette still referenced"
    assert "theme-color" in html


def test_submission_result_links_to_disease_guide():
    src = FS.read_text(encoding="utf-8")
    assert 'id="fs_guide"' in src
    assert "WDED_KB_OPEN" in src
    assert "wded:focus-disease" in src


def test_knowledge_index_html_inline_script_still_valid():
    html = INDEX.read_text(encoding="utf-8")
    inline = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert inline, "no inline script found"
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
        tmp.write(inline[-1])
        path = tmp.name
    try:
        subprocess.run(["node", "--check", path], check=True)
    finally:
        Path(path).unlink(missing_ok=True)


def test_pyproject_version_bumped_050():
    """Version must be at least 0.5.0 (the knowledge-module release)."""
    import re

    pyproject = (DOCS.parent / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'version = "(\d+)\.(\d+)\.(\d+)"', pyproject)
    assert match, "pyproject version not found"
    version = tuple(int(p) for p in match.groups())
    assert version >= (0, 5, 0), f"version {version} regressed below 0.5.0"
