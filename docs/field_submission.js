/* WDED — Field Data Submission & Analysis Module (v0.4.0)
 *
 * Farmers / scouts submit locally collected observations from the study area:
 *   - weather aggregates (mean temp, RH, leaf wetness, rainfall),
 *   - soil properties (pH, drainage, residue, moisture, OM, N/P/K),
 *   - canopy reflectance values (green / red / red edge / NIR, 0-1).
 *
 * The browser-side analysis mirrors the Python pipeline exactly:
 *   weather prior   -> wded/alignment.py  (infection-window model, same constants)
 *   soil score      -> wded/alignment.py  (soil_suitability_score)
 *   fusion + tiers  -> wded/fusion.py     (0.5/0.3/0.2, tiers 0.25/0.50/0.75, escalation 0.05)
 *   recommendation  -> wded/recommendation.py (actions, urgency, re-scout cadence, notes)
 *
 * The spectral term is a TRANSPARENT SSCNN PROXY computed from the submitted
 * reflectance (index departures from healthy baselines). When a real model.pt
 * checkpoint becomes available, this term can be replaced by on-device
 * inference without changing the fusion or recommendation layers.
 */
(function () {
'use strict';

/* ------------------------------------------------------------------ *
 * 1. Constants — mirrored from the Python pipeline                    *
 * ------------------------------------------------------------------ */

const DISEASES = ['stem_rust', 'stripe_rust', 'leaf_rust', 'septoria', 'fusarium'];

const DISEASE_LABEL = {
  stem_rust: 'stem rust', stripe_rust: 'stripe rust', leaf_rust: 'leaf rust',
  septoria: 'septoria', fusarium: 'fusarium', healthy: 'healthy background',
};

/* wded/alignment.py :: INFECTION_WINDOWS */
const INFECTION_WINDOWS = {
  stem_rust:   { t_lo: 15.0, t_hi: 35.0, leaf_wetness_h: 4.0, lag_days: 14 },
  stripe_rust: { t_lo:  5.0, t_hi: 20.0, leaf_wetness_h: 4.0, lag_days: 14 },
  leaf_rust:   { t_lo: 15.0, t_hi: 30.0, leaf_wetness_h: 4.0, lag_days: 14 },
  septoria:    { t_lo: 10.0, t_hi: 25.0, leaf_wetness_h: 8.0, lag_days: 21 },
  fusarium:    { t_lo: 20.0, t_hi: 30.0, leaf_wetness_h: 6.0, lag_days: 10 },
};
const WEATHER_WINDOW_DAYS = 21;   // longest lag (septoria)
const RAIN_REF_MM = 25.0;         // e-folding rainfall total per window

/* wded/config.py :: FusionWeights + RISK_TIERS + escalation_margin */
const WEIGHTS = { spectral: 0.5, weather: 0.3, soil: 0.2 };
const TIER_ORDER  = ['low', 'moderate', 'high', 'critical'];
const TIER_BOUNDS = { low: 0.25, moderate: 0.50, high: 0.75 };
const ESCALATION_MARGIN = 0.05;

/* wded/recommendation.py */
const TIER_ACTIONS = {
  critical: 'Apply fungicide immediately; expert on-ground confirmation within 24 h; flag the block for harvest segregation.',
  high:     'Confirmatory scouting within 48 h; prepare the spray plan; re-fly the block in 5-7 days.',
  moderate: 'Intensify monitoring: re-flight within 72 h and scout flagged tiles; watch the 7-day weather forecast for infection windows.',
  low:      'Routine monitoring cadence; no targeted action before the next scheduled flight.',
};
const TIER_URGENCY = {
  critical: 'immediate - act within 24 h',
  high:     'urgent - act within 48 h',
  moderate: 'elevated - monitor within 72 h',
  low:      'routine - next scheduled flight',
};
const TIER_RESCOUT_DAYS = { critical: 3, high: 5, moderate: 7, low: 14 };
const DISEASE_NOTES = {
  stem_rust:   'Ug99-race watch: if severity exceeds 10%, send samples for race typing.',
  stripe_rust: 'Cool-season favourite: prioritise upper-canopy checking after cool, wet spells.',
  leaf_rust:   'Protect the flag leaf; check for post-flushing tillers after rain.',
  septoria:    'Prioritise lower-canopy lesions; protect the flag leaf at T0/T1 timing.',
  fusarium:    'If the crop is at anthesis, assess FHB and DON mycotoxin risk before harvest.',
};

/* Healthy-canopy baselines (calibrated study-area mock: ndvi 0.75, ndre 0.30) */
const BASELINE = { ndvi: 0.75, ndre: 0.30, gndvi: 0.62 };

const GROWTH_STAGES = [
  'seedling', 'tillering', 'stem_elongation', 'booting',
  'heading', 'anthesis', 'grain_fill', 'ripe',
];

const HISTORY_KEY = 'wded_field_submissions_v1';
const SCHEMA = 'wded.field_submission/1';

/* ------------------------------------------------------------------ *
 * 2. Analysis engine (mirror of the Python pipeline)                  *
 * ------------------------------------------------------------------ */

function clip(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }

/* alignment.py :: _temp_suitability */
function tempSuitability(t, lo, hi) {
  const mid = (lo + hi) / 2.0;
  const span = (hi - lo) / 2.0;
  return Math.exp(-Math.pow((t - mid) / span, 2));
}

/* Crude but transparent RH -> wet-hours proxy used when the scout cannot
 * measure leaf wetness directly: 70% RH -> 0 h, 100% RH -> 12 h/day. */
function estimateLeafWetness(rh) {
  if (rh == null || isNaN(rh)) return null;
  return clip(((rh - 70.0) / 30.0) * 12.0, 0, 14);
}

/* alignment.py :: infection_weather_prior — aggregate-input form.
 * Window rainfall is scaled from the submitted 21-day total by lag/21. */
function weatherPrior(w, disease) {
  const cfg = INFECTION_WINDOWS[disease];
  const tempSuit = tempSuitability(w.temp_mean_c, cfg.t_lo, cfg.t_hi);
  const lwh = (w.leaf_wetness_hours != null && !isNaN(w.leaf_wetness_hours))
    ? w.leaf_wetness_hours
    : estimateLeafWetness(w.rh_mean_pct);
  const wetScore = clip((lwh || 0) / cfg.leaf_wetness_h, 0, 1.5) / 1.5;
  const windowRain = w.rain_21d_mm * (cfg.lag_days / WEATHER_WINDOW_DAYS);
  const rainScore = 1.0 - Math.exp(-windowRain / RAIN_REF_MM);
  return clip(0.45 * tempSuit + 0.35 * wetScore + 0.20 * rainScore, 0, 1);
}

/* alignment.py :: soil_suitability_score */
function soilScore(s) {
  const pH = (s.ph == null || isNaN(s.ph)) ? 6.8 : s.ph;
  const pHrisk = 1.0 - Math.exp(-Math.pow((pH - 6.8) / 0.9, 2));
  const drainage = String(s.drainage || 'moderate').toLowerCase();
  const drainageRisk = {
    very_poor: 1.0, poor: 0.8, moderate: 0.5, good: 0.25, excessive: 0.15,
  }[drainage];
  const residue = s.previous_cereal ? 1.0 : 0.0;
  return clip(0.5 * pHrisk + 0.5 * (drainageRisk == null ? 0.5 : drainageRisk) + 0.05 * residue, 0, 1);
}

/* Canopy spectral indices from reflectance values in [0, 1] */
function spectralIndices(b) {
  const safe = (a, c) => ((a + c) === 0 ? 0 : (a - c) / (a + c));
  return {
    ndvi:  safe(b.nir, b.red),
    ndre:  safe(b.nir, b.red_edge),
    gndvi: safe(b.nir, b.green),
  };
}

/* Transparent SSCNN proxy: per-disease pressure from index departures.
 *   stem rust  -> NIR crash (NDVI collapse) dominates
 *   stripe rust -> red-edge rise (NDRE drop) dominates
 *   leaf rust  -> mixed NDVI/GNDVI decline
 *   septoria   -> broad visible+NIR decline (necrotic speckle)
 *   fusarium   -> weak spectral signal (head disease; weather drives risk) */
function spectralScores(ix) {
  const drop = (v, base) => clip((base - v) / base, 0, 1);
  const dN = drop(ix.ndvi, BASELINE.ndvi);
  const dR = drop(ix.ndre, BASELINE.ndre);
  const dG = drop(ix.gndvi, BASELINE.gndvi);
  return {
    stem_rust:   clip(0.70 * dN + 0.20 * dG + 0.10 * dR, 0, 1) * 0.9,
    stripe_rust: clip(0.25 * dN + 0.75 * dR,              0, 1) * 0.9,
    leaf_rust:   clip(0.55 * dN + 0.30 * dG + 0.15 * dR,  0, 1) * 0.9,
    septoria:    clip(0.50 * dN + 0.50 * dG,              0, 1) * 0.9,
    fusarium:    clip(0.30 * dN + 0.20 * dG + 0.10 * dR,  0, 1) * 0.9,
  };
}

/* fusion.py :: tier_of + escalate_tier */
function tierOf(risk) {
  if (risk >= 0.75) return 'critical';
  if (risk >= 0.50) return 'high';
  if (risk >= 0.25) return 'moderate';
  return 'low';
}

function escalateTier(tier, risk) {
  const bound = TIER_BOUNDS[tier];
  if (bound != null && risk >= bound - ESCALATION_MARGIN) {
    return TIER_ORDER[Math.min(TIER_ORDER.indexOf(tier) + 1, TIER_ORDER.length - 1)];
  }
  return tier;
}

/* Full analysis of one submission. Returns the payload.analysis object. */
function analyze(sub) {
  const weatherPriors = {};
  const spec = {};
  const riskPerDisease = {};

  DISEASES.forEach((d) => { weatherPriors[d] = weatherPrior(sub.weather, d); });

  const indices = spectralIndices(sub.reflectance);
  const specRaw = spectralScores(indices);
  const soil = soilScore(sub.soil);

  DISEASES.forEach((d) => {
    spec[d] = specRaw[d];
    riskPerDisease[d] = clip(
      WEIGHTS.spectral * specRaw[d] +
      WEIGHTS.weather * weatherPriors[d] +
      WEIGHTS.soil * soil, 0, 1);
  });

  let overall = 0;
  let dominant = DISEASES[0];
  DISEASES.forEach((d) => {
    if (riskPerDisease[d] > overall) { overall = riskPerDisease[d]; dominant = d; }
  });

  const baseTier = tierOf(overall);
  const tier = escalateTier(baseTier, overall);

  return {
    indices,
    weather_priors: weatherPriors,
    soil_score: soil,
    spectral_scores: spec,
    risk_per_disease: riskPerDisease,
    overall_risk: overall,
    dominant_disease: dominant,
    base_tier: baseTier,
    tier,
    escalated: tier !== baseTier,
    fusion_weights: WEIGHTS,
    recommendation: TIER_ACTIONS[tier],
    urgency: TIER_URGENCY[tier],
    rescout_days: TIER_RESCOUT_DAYS[tier],
    disease_note: DISEASE_NOTES[dominant] || 'Confirm symptoms before treatment.',
  };
}

/* ------------------------------------------------------------------ *
 * 3. Validation                                                       *
 * ------------------------------------------------------------------ */

function validate(sub) {
  const errors = [];
  const req = (cond, field, msg) => { if (!cond) errors.push({ field, msg }); };
  const num = (v) => (v == null || v === '' || isNaN(v) ? null : Number(v));

  const t = num(sub.weather.temp_mean_c);
  req(t != null && t >= -10 && t <= 50, 'fs_temp', 'mean temperature must be a number between -10 and 50 °C');
  const rh = num(sub.weather.rh_mean_pct);
  req(rh != null && rh > 0 && rh <= 100, 'fs_rh', 'mean relative humidity must be between 0 and 100 %');
  const rain = num(sub.weather.rain_21d_mm);
  req(rain != null && rain >= 0 && rain <= 1000, 'fs_rain', '21-day rainfall must be between 0 and 1000 mm');
  const lwh = num(sub.weather.leaf_wetness_hours);
  req(lwh == null || (lwh >= 0 && lwh <= 24), 'fs_lwh', 'leaf wetness must be between 0 and 24 h/day');

  const ph = num(sub.soil.ph);
  req(ph != null && ph >= 3 && ph <= 10, 'fs_ph', 'soil pH must be between 3 and 10');
  req(!!sub.soil.drainage, 'fs_drainage', 'select a drainage class');

  ['green', 'red', 'red_edge', 'nir'].forEach((b) => {
    const v = num(sub.reflectance[b]);
    req(v != null && v >= 0 && v <= 1, 'fs_' + b, b.replace('_', ' ') + ' reflectance must be between 0 and 1');
  });

  const lat = num(sub.field.lat), lon = num(sub.field.lon);
  req(lat == null || (lat >= -90 && lat <= 90), 'fs_lat', 'latitude must be between -90 and 90');
  req(lon == null || (lon >= -180 && lon <= 180), 'fs_lon', 'longitude must be between -180 and 180');

  const moist = num(sub.soil.moisture_pct);
  req(moist == null || (moist >= 0 && moist <= 100), 'fs_moisture', 'soil moisture must be between 0 and 100 %');
  const om = num(sub.soil.organic_matter_pct);
  req(om == null || (om >= 0 && om <= 25), 'fs_om', 'organic matter must be between 0 and 25 %');
  ['n_mg_kg', 'p_mg_kg', 'k_mg_kg'].forEach((k, i) => {
    const v = num(sub.soil[k]);
    req(v == null || (v >= 0 && v <= 5000), ['fs_n', 'fs_p', 'fs_k'][i], 'nutrient values must be between 0 and 5000 mg/kg');
  });

  return errors;
}

/* ------------------------------------------------------------------ *
 * 4. DOM helpers                                                      *
 * ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;',
  }[c]));
}
function numVal(id) {
  const v = $(id).value.trim();
  return v === '' ? null : Number(v);
}
function badge(tier) {
  return '<span class="badge b-' + esc(tier) + '">' + esc(tier) + '</span>';
}

/* ------------------------------------------------------------------ *
 * 5. Form <-> payload                                                 *
 * ------------------------------------------------------------------ */

function readForm() {
  const lat = numVal('fs_lat'), lon = numVal('fs_lon');
  const sub = {
    schema: SCHEMA,
    submitted_at: new Date().toISOString(),
    observation_date: $('fs_date').value || new Date().toISOString().slice(0, 10),
    field: {
      field_id: $('fs_field_id').value.trim() || null,
      site_id: (window.WDED_SUMMARY && window.WDED_SUMMARY.site) || 'bishoftu',
      lat: lat, lon: lon,
      growth_stage: $('fs_stage').value || null,
      observer: $('fs_observer').value.trim() || null,
    },
    weather: {
      window_days: WEATHER_WINDOW_DAYS,
      temp_mean_c: numVal('fs_temp'),
      rh_mean_pct: numVal('fs_rh'),
      leaf_wetness_hours: numVal('fs_lwh'),
      leaf_wetness_estimated: numVal('fs_lwh') == null,
      rain_21d_mm: numVal('fs_rain'),
    },
    soil: {
      ph: numVal('fs_ph'),
      drainage: $('fs_drainage').value || null,
      previous_cereal: $('fs_prev_cereal').value === 'yes',
      moisture_pct: numVal('fs_moisture'),
      organic_matter_pct: numVal('fs_om'),
      n_mg_kg: numVal('fs_n'),
      p_mg_kg: numVal('fs_p'),
      k_mg_kg: numVal('fs_k'),
    },
    reflectance: {
      green: numVal('fs_green'),
      red: numVal('fs_red'),
      red_edge: numVal('fs_red_edge'),
      nir: numVal('fs_nir'),
      bands_nm: { green: 560, red: 650, red_edge: 730, nir: 840 },
      source: $('fs_refl_source').value || 'handheld',
    },
  };
  sub.analysis = analyze(sub);
  return sub;
}

function fillErrors(errors) {
  const box = $('fs_errors');
  if (!errors.length) { box.style.display = 'none'; box.innerHTML = ''; return; }
  errors.forEach((e) => {
    const el = $(e.field);
    if (el) { el.classList.add('fsBad'); }
  });
  box.style.display = 'block';
  box.innerHTML = '<b>Please fix ' + errors.length + ' field' + (errors.length > 1 ? 's' : '') + ':</b><ul>' +
    errors.map((e) => '<li>' + esc(e.msg) + '</li>').join('') + '</ul>';
}

function clearErrors() {
  const box = $('fs_errors');
  box.style.display = 'none'; box.innerHTML = '';
  document.querySelectorAll('.fsBad').forEach((el) => el.classList.remove('fsBad'));
}

/* ------------------------------------------------------------------ *
 * 6. Results rendering                                                *
 * ------------------------------------------------------------------ */

function barRow(label, pct, color, suffix) {
  return '<div class="bar-row"><span style="width:118px">' + esc(label) + '</span>' +
    '<span class="bar" style="background:' + color + ';width:' + Math.max(2, Math.round(1.6 * pct)) + 'px"></span>' +
    '<span>' + suffix + '</span></div>';
}

function renderResult(sub) {
  const a = sub.analysis;
  const driverLabel = a.overall_risk < 0.10 ? 'healthy background' : DISEASE_LABEL[a.dominant_disease];

  const parts = [];
  parts.push('<div class="fsHead">' + badge(a.tier) +
    (a.escalated ? ' <span class="trendchip tw" title="asymmetric-cost escalation (FN:FP = 5:1)">▲ escalated</span>' : '') +
    '<span class="fsRisk">overall risk <b>' + (100 * a.overall_risk).toFixed(1) + '%</b></span>' +
    '<span class="fsDriver">driver: <b>' + esc(driverLabel) + '</b></span></div>');

  parts.push('<h3>Evidence contribution</h3>');
  const contrib = [
    ['spectral (0.5×)', a.spectral_scores[a.dominant_disease], WEIGHTS.spectral, '#D4875A'],
    ['weather (0.3×)', a.weather_priors[a.dominant_disease], WEIGHTS.weather, '#6fa8dc'],
    ['soil (0.2×)', a.soil_score, WEIGHTS.soil, '#8fd694'],
  ];
  contrib.forEach(([label, raw, w, color]) => {
    parts.push(barRow(label, 100 * raw * w, color, (100 * raw * w).toFixed(1) + ' pp of ' + (100 * raw).toFixed(1) + '%'));
  });

  parts.push('<h3 style="margin-top:14px">Per-disease fused risk</h3>');
  DISEASES.slice().sort((x, y) => a.risk_per_disease[y] - a.risk_per_disease[x]).forEach((d) => {
    const pct = 100 * a.risk_per_disease[d];
    const color = a.risk_per_disease[d] >= 0.75 ? 'var(--critical)'
      : a.risk_per_disease[d] >= 0.50 ? 'var(--high)'
      : a.risk_per_disease[d] >= 0.25 ? 'var(--moderate)' : 'var(--low)';
    parts.push(barRow(DISEASE_LABEL[d], pct, color, pct.toFixed(1) + '%'));
  });

  const ix = a.indices;
  parts.push('<div class="fsIdx">' +
    idxChip('NDVI', ix.ndvi, BASELINE.ndvi) + idxChip('NDRE', ix.ndre, BASELINE.ndre) +
    idxChip('GNDVI', ix.gndvi, BASELINE.gndvi) + '</div>');

  parts.push('<div class="fsRec"><h3>Recommendation for the farmer</h3>' +
    '<p><b>' + esc(TIER_URGENCY[a.tier]) + '</b> · re-scout / re-fly ≤ ' + a.rescout_days + ' days</p>' +
    '<p>' + esc(a.recommendation) + '</p>' +
    '<p><i>' + esc(a.disease_note) + '</i></p>' +
    '<p class="fsFine">Fusion: ' + WEIGHTS.spectral + ' · spectral signature (SSCNN proxy) + ' +
    WEIGHTS.weather + ' · lagged-weather prior + ' + WEIGHTS.soil + ' · soil score. Decision support only — ' +
    'confirm on the ground before any treatment.</p></div>');

  parts.push('<div class="fsBtns">' +
    '<button class="act" id="fs_save">💾 Save to field log</button>' +
    '<button class="act" id="fs_download">⬇ JSON payload</button>' +
    ((sub.field.lat != null || sub.field.field_id) ? '<button class="act" id="fs_map">📍 Show on map</button>' : '') +
    '</div>');

  $('fs_result').innerHTML = parts.join('');
  $('fs_result').style.display = 'block';

  $('fs_save').addEventListener('click', () => saveSubmission(sub));
  $('fs_download').addEventListener('click', () => downloadJSON(sub, fileNameFor(sub)));
  const mapBtn = $('fs_map');
  if (mapBtn) {
    mapBtn.addEventListener('click', () => {
      window.dispatchEvent(new CustomEvent('wded:focus-tile', {
        detail: { tileId: sub.field.field_id, lat: sub.field.lat, lon: sub.field.lon },
      }));
    });
  }
}

function idxChip(name, v, base) {
  const delta = v - base;
  const cls = delta <= -0.10 ? 'tw' : delta < 0 ? 'ts' : 'ti';
  const arrow = delta <= -0.10 ? '▼' : delta < 0 ? '▾' : delta > 0 ? '▴' : '▬';
  return '<span class="fsIdxChip"><b>' + name + '</b> ' + v.toFixed(3) +
    ' <span class="trendchip ' + cls + '">' + arrow + ' ' + (delta >= 0 ? '+' : '') + delta.toFixed(3) +
    ' vs healthy ' + base.toFixed(2) + '</span></span>';
}

/* ------------------------------------------------------------------ *
 * 7. History (localStorage) + exports                                 *
 * ------------------------------------------------------------------ */

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch (e) { return []; }
}
function persistHistory(list) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); }
  catch (e) { /* storage unavailable — session-only */ }
}
function saveSubmission(sub) {
  const list = loadHistory();
  list.unshift(sub);
  persistHistory(list.slice(0, 200));
  renderHistory();
  const btn = $('fs_save');
  if (btn) { btn.textContent = '✓ saved'; setTimeout(() => { btn.textContent = '💾 Save to field log'; }, 1500); }
}

function fileNameFor(sub) {
  const d = (sub.observation_date || '').replace(/-/g, '');
  const f = (sub.field.field_id || 'field').replace(/[^A-Za-z0-9_-]+/g, '_');
  return 'wded_submission_' + f + '_' + d + '.json';
}

function downloadText(text, name, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
function downloadJSON(obj, name) {
  downloadText(JSON.stringify(obj, null, 2), name, 'application/json');
}

function historyCSV(list) {
  const cols = ['submitted_at', 'observation_date', 'field_id', 'lat', 'lon', 'growth_stage',
    'temp_mean_c', 'rh_mean_pct', 'leaf_wetness_hours', 'rain_21d_mm',
    'ph', 'drainage', 'previous_cereal', 'ndvi', 'ndre', 'gndvi',
    'overall_risk', 'tier', 'dominant_disease', 'urgency', 'rescout_days'];
  const get = (sub, c) => {
    const a = sub.analysis || {};
    const map = {
      submitted_at: sub.submitted_at, observation_date: sub.observation_date,
      field_id: sub.field.field_id, lat: sub.field.lat, lon: sub.field.lon,
      growth_stage: sub.field.growth_stage,
      temp_mean_c: sub.weather.temp_mean_c, rh_mean_pct: sub.weather.rh_mean_pct,
      leaf_wetness_hours: sub.weather.leaf_wetness_hours, rain_21d_mm: sub.weather.rain_21d_mm,
      ph: sub.soil.ph, drainage: sub.soil.drainage, previous_cereal: sub.soil.previous_cereal,
      ndvi: a.indices && a.indices.ndvi, ndre: a.indices && a.indices.ndre,
      gndvi: a.indices && a.indices.gndvi,
      overall_risk: a.overall_risk, tier: a.tier, dominant_disease: a.dominant_disease,
      urgency: a.urgency, rescout_days: a.rescout_days,
    };
    const v = map[c];
    return v == null ? '' : String(v).replace(/[,\n;]/g, ' ');
  };
  return [cols.join(',')].concat(list.map((s) => cols.map((c) => get(s, c)).join(','))).join('\n');
}

function renderHistory() {
  const list = loadHistory();
  const tbody = $('fs_history_body');
  const wrap = $('fs_history');
  if (!list.length) {
    wrap.style.display = 'none';
    tbody.innerHTML = '';
    return;
  }
  wrap.style.display = 'block';
  tbody.innerHTML = list.map((s, i) => {
    const a = s.analysis || {};
    return '<tr>' +
      '<td>' + esc((s.observation_date || '').slice(0, 10)) + '</td>' +
      '<td>' + esc(s.field.field_id || '—') + '</td>' +
      '<td>' + (a.overall_risk != null ? (100 * a.overall_risk).toFixed(1) + '%' : '—') + '</td>' +
      '<td>' + (a.tier ? badge(a.tier) : '—') + '</td>' +
      '<td>' + esc(DISEASE_LABEL[a.dominant_disease] || '—') + '</td>' +
      '<td class="fsRowBtns">' +
      '<button class="mini" data-act="dl" data-i="' + i + '" title="download JSON payload">⬇</button>' +
      '<button class="mini" data-act="del" data-i="' + i + '" title="delete entry">✕</button></td>' +
      '</tr>';
  }).join('');

  tbody.querySelectorAll('button.mini').forEach((b) => {
    b.addEventListener('click', () => {
      const i = Number(b.dataset.i);
      const list2 = loadHistory();
      if (b.dataset.act === 'dl') { downloadJSON(list2[i], fileNameFor(list2[i])); return; }
      if (b.dataset.act === 'del') { list2.splice(i, 1); persistHistory(list2); renderHistory(); }
    });
  });
}

/* ------------------------------------------------------------------ *
 * 8. Example values + reset                                           *
 * ------------------------------------------------------------------ */

const EXAMPLE = {
  fs_field_id: 'BISH-NE-07', fs_temp: '16.5', fs_rh: '78', fs_lwh: '',
  fs_rain: '38', fs_ph: '7.2', fs_drainage: 'poor', fs_prev_cereal: 'yes',
  fs_moisture: '24', fs_om: '1.8', fs_n: '34', fs_p: '12', fs_k: '180',
  fs_green: '0.09', fs_red: '0.075', fs_red_edge: '0.27', fs_nir: '0.40',
  fs_lat: '8.75', fs_lon: '39.0', fs_stage: 'heading',
};

function loadExample() {
  Object.keys(EXAMPLE).forEach((id) => { const el = $(id); if (el) el.value = EXAMPLE[id]; });
  $('fs_lwh').placeholder = 'auto from RH ≈ 3.2 h';
  clearErrors();
}

function resetForm() {
  $('fsForm').reset();
  $('fs_date').value = new Date().toISOString().slice(0, 10);
  $('fs_lwh').placeholder = 'blank = estimate from RH';
  $('fs_result').style.display = 'none';
  clearErrors();
}

/* ------------------------------------------------------------------ *
 * 9. Wiring                                                           *
 * ------------------------------------------------------------------ */

function initFieldSubmission() {
  if (!$('fsForm')) return;
  $('fs_date').value = new Date().toISOString().slice(0, 10);
  const stage = $('fs_stage');
  GROWTH_STAGES.forEach((s) => {
    const o = document.createElement('option');
    o.value = s; o.textContent = s.replace(/_/g, ' ');
    stage.appendChild(o);
  });

  $('fs_example').addEventListener('click', (e) => { e.preventDefault(); loadExample(); });
  $('fs_reset').addEventListener('click', (e) => { e.preventDefault(); resetForm(); });
  $('fsForm').addEventListener('submit', (e) => {
    e.preventDefault();
    clearErrors();
    const sub = readForm();
    const errors = validate(sub);
    fillErrors(errors);
    if (errors.length) return;
    renderResult(sub);
  });

  $('fs_csv').addEventListener('click', () => {
    const list = loadHistory();
    if (list.length) downloadText(historyCSV(list), 'wded_field_log.csv', 'text/csv');
  });
  $('fs_json').addEventListener('click', () => {
    const list = loadHistory();
    if (list.length) downloadJSON({ schema: SCHEMA, n: list.length, submissions: list }, 'wded_field_log.json');
  });
  $('fs_clear').addEventListener('click', () => {
    if (confirm('Delete all locally stored field submissions?')) { persistHistory([]); renderHistory(); }
  });

  renderHistory();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initFieldSubmission);
} else {
  initFieldSubmission();
}

/* Test hooks (also usable from the console) */
window.WDEDSubmit = {
  analyze, weatherPrior, soilScore, spectralIndices, spectralScores,
  tierOf, escalateTier, estimateLeafWetness, validate,
  INFECTION_WINDOWS, WEIGHTS, BASELINE, DISEASES,
};
})();
