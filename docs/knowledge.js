/* WDED — Disease Knowledge Module (v0.5.0)
 *
 * Agronomic knowledge base for the five target diseases, correlated with the
 * model's prediction results:
 *   - the Disease guide tab lists which diseases the latest flight flagged
 *     as dominant driver and in how many tiles (window.WDED_SUMMARY),
 *   - risk-map tile popups and field-submission results deep-link into the
 *     guide for their dominant disease via window.WDED_KB_OPEN(id),
 *   - infection-condition chips mirror wded/alignment.py INFECTION_WINDOWS
 *     so farmers see exactly what the weather prior evaluates.
 * Content is decision-support information for the Ethiopian study area
 * (Bishoftu / Debre Zeit, Asella, Ambo); validate locally before spraying.
 */
(function () {
'use strict';

const DISEASES = ['stem_rust', 'stripe_rust', 'leaf_rust', 'septoria', 'fusarium'];

const KB = {
  stem_rust: {
    label: 'Stem rust (black rust)',
    pathogen: 'Puccinia graminis f. sp. tritici',
    hue: '#A63A2B',
    icon: '🟥',
    severity: 'The most destructive wheat disease worldwide — epidemics can claim the whole crop in weeks. The Ug99 race group defeats many common resistance genes, which is why race-typing matters.',
    summary: 'A fast-moving fungal rust that attacks stems, leaf sheaths and heads. Brick-red powdery pustules rupture the stem epidermis, girdling the culm so it lodges and shrivels grain. Wind-borne spores travel hundreds of kilometres, so a single hot spot threatens the whole study area.',
    symptoms: [
      'Brick-red to dark red-brown, oval-to-elongated pustules (uredinia) on stems and leaf sheaths — rough to the touch, torn epidermis frays at the edges.',
      'Pustules can also appear on glume bases and awns near heading; "snap" symptom: tissue splits visibly around the pustule.',
      'Late in the season pustules turn black-brown (telia) and do not powder.',
      'Severe infection: stems girdled, plants lodge, grain shrivels with poor test weight.',
    ],
    conditions: {
      temp_lo: 15, temp_hi: 35, temp_note: 'warm conditions; optimum 20–28 °C',
      wetness: '4 h leaf wetness / heavy dew', lag_days: 14,
      other: 'Urediniospores are wind-dispersed over long distances; dense, lush canopies and lush late N favour epidemics.',
    },
    spectral: 'The SSCNN reads stem rust as a canopy-structure collapse: a sharp NIR reflectance crash drives NDVI down first, with a smaller GNDVI/NDRE response — usually strongest over pustule clusters and lodged patches.',
    riskFactors: [
      'Warm days with dewy nights and 4+ h leaf wetness in the past two weeks',
      'Susceptible variety without effective Sr resistance gene',
      'Neighbouring fields already flagged high/critical (spore load builds locally)',
      'Lodge-prone, over-fertilised dense canopy',
    ],
    management: {
      cultural: [
        'Scout flagged blocks within 24–48 h; walk the pustule clusters flagged by the risk map.',
        'Avoid excess nitrogen and very dense sowing that keeps canopies humid.',
        'Plan harvest order: cut high-risk blocks first and segregate shrivelled grain.',
      ],
      chemical: [
        'Triazole + strobilurin premixes (e.g. tebuconazole or propiconazole + azoxystrobin) at first sign; repeat per label interval if pressure persists.',
        'Spray coverage must reach stems — use adequate water volume; act before pustules erupt on >10% of stems.',
      ],
      resistance: [
        'Use varieties carrying Ug99-effective stem-rust resistance (e.g. Sr22, Sr26, Sr35, Sr50 combinations); ask your research centre for the current local recommendation.',
        'Send stem samples for race typing when severity exceeds 10% (Ug99 surveillance).',
      ],
    },
    scoutingTip: 'Rub a white cloth along the stem: brick-red powder confirms active stem rust. Check the same flagged tiles every 3–5 days during warm spells.',
  },

  stripe_rust: {
    label: 'Stripe rust (yellow rust)',
    pathogen: 'Puccinia striiformis Westend. f. sp. tritici',
    hue: '#D9A441',
    icon: '🟨',
    severity: 'Can erase 40–100% of yield in cool, wet seasons when it reaches the upper canopy and flag leaf before grain fill.',
    summary: 'A cool-loving rust that forms characteristic yellow stripes between leaf veins. It builds quietly during long cool, humid spells and then erupts across the upper canopy, stealing photosynthate from the flag leaf exactly when grain fill needs it most.',
    symptoms: [
      'Small yellow-orange pustules arranged in neat stripes/lines between leaf veins — the signature symptom.',
      'Prefers the upper canopy: flag leaf and F-1 get hit first after a cool wet spell.',
      'Powdery pustules that rub off as a yellow dust; older leaves may show black telia underneath late season.',
      'In heavy attacks, stripes merge, leaves yellow and wither from the tips.',
    ],
    conditions: {
      temp_lo: 5, temp_hi: 20, temp_note: 'cool conditions; optimum 10–15 °C',
      wetness: '4 h leaf wetness / dew', lag_days: 14,
      other: 'Thrives on cool nights, dew and drizzle; hot days (>22 °C sustained) slow it down. Can cycle repeatedly within one season.',
    },
    spectral: 'Stripe rust shows up as a red-edge anomaly: the model reads a red-edge band rise / REIP blue-shift that drags NDRE down while NDVI is still only mildly affected — an early signal before visible striping.',
    riskFactors: [
      'Mean temperature inside the 5–20 °C window with dewy nights over the past two weeks',
      'Susceptible variety lacking Yr resistance',
      'Cool, wet spell forecast for the coming week (infection windows ahead of symptoms)',
      'Early-sown, thick stands that hold dew',
    ],
    management: {
      cultural: [
        'Prioritise upper-canopy checking on flagged tiles within 48 h — stripes are easiest to see in morning side-light.',
        'Avoid overly early sowing and excessive seed rates that create dense, dew-holding canopies.',
        'Balanced N: very lush canopies prolong wetness and rust multiplication.',
      ],
      chemical: [
        'Triazoles (tebuconazole, propiconazole) or DMI+QoI premixes at first stripe appearance; a timely first spray protecting the flag leaf gives the best return.',
        'Re-check treated tiles after 7–10 days — new leaf layers can be re-infected.',
      ],
      resistance: [
        'Grow Yr-gene varieties recommended for the Arsi/Bale/Bishoftu pathotype mix; stack adult-plant resistance where possible.',
        'Seed treatment (e.g. triadimenol/flutriafol where registered) gives early-season cover in known hotspots.',
      ],
    },
    scoutingTip: 'Hold the leaf against the light: yellow dust in parallel lines = stripe rust. After every cool, wet 5-day spell, walk the upper canopy of flagged tiles first.',
  },

  leaf_rust: {
    label: 'Leaf rust (brown rust)',
    pathogen: 'Puccinia triticina',
    hue: '#B4632C',
    icon: '🟫',
    severity: 'Typically 10–30% yield loss on susceptible varieties, but flag-leaf infection during grain fill pushes losses higher.',
    summary: 'The most common wheat rust. Round orange-brown pustules scatter randomly across the upper leaf surface, building through the season on dewy warm nights. It nibbles at the flag leaf during grain fill, lightening grain and weakening straw.',
    symptoms: [
      'Round to oval, orange-brown pustules scattered randomly on the upper leaf surface — no stripe pattern, no stem involvement.',
      'Pustules powder easily (orange spore dust); dark telia appear on the underside of older leaves late season.',
      'Starts low in the canopy and moves upward with each wet night.',
      'Post-rain flushing tillers often show fresh infections first.',
    ],
    conditions: {
      temp_lo: 15, temp_hi: 30, temp_note: 'mild-to-warm; optimum 20–25 °C',
      wetness: '4 h leaf wetness / dew', lag_days: 14,
      other: 'Warm days + dewy nights accelerate cycles; infection resets with every rainy period, so watch after each rain event.',
    },
    spectral: 'Leaf rust produces a mixed canopy decline: moderate NDVI drop with a visible-band (GNDVI) component — weaker and patchier than the stem-rust NIR crash, so the model leans on its temporal spread across tiles.',
    riskFactors: [
      'Dewy warm nights (15–30 °C) with 4 h wetness in the past fortnight',
      'Susceptible variety (no effective Lr gene)',
      'Rain followed by flushing tillers; lush late canopy',
      'Local spore build-up on nearby flagged tiles',
    ],
    management: {
      cultural: [
        'Check flagged tiles within 48 h, focusing on the flag leaf and newly flushed tillers after rain.',
        'Avoid very dense stands and excessive late N that prolong leaf wetness.',
      ],
      chemical: [
        'Protect the flag leaf: triazole or triazole+strobilurin at T1/T2 timing (or immediately when pustules reach the flag leaf); cover newly emerged leaves after rain.',
        'Follow label pre-harvest intervals before harvest segregation decisions.',
      ],
      resistance: [
        'Use varieties with durable adult-plant leaf-rust resistance (slow-rusting Lr34/Lr46-type background plus seedling genes where available).',
        'Rotate resistance sources between seasons — leaf rust adapts quickly.',
      ],
    },
    scoutingTip: 'Random round pustules (not lines, not on stems) = leaf rust. Count pustules on 10 flag leaves per flagged tile; >1% flag-leaf area means spray economics usually turn positive.',
  },

  septoria: {
    label: 'Septoria leaf blotch',
    pathogen: 'Zymoseptoria tritici (Septoria tritici)',
    hue: '#8A7F5C',
    icon: '🟧',
    severity: 'Rain-fed seasons commonly lose 20–40% of usable green canopy; flag-leaf loss at grain fill is the costly phase.',
    summary: 'A residue-borne leaf blotch that starts on the lowest leaves after rain-splash dispersal and climbs the canopy during wet spells. Its long invisible latency (~3 weeks) means today\u2019s weather already decided much of the next outbreak — which is exactly what the lagged weather prior captures.',
    symptoms: [
      'Irregular grey-brown necrotic lesions running parallel to leaf veins, edged by the leaf\u2019s yellowing.',
      'Tiny black pepper-dot fruiting bodies (pycnidia) embedded inside the lesions — key discriminator from other blotches.',
      'Starts on the lowest, oldest leaves; moves upward with each splash-dispersing rain.',
      'Severe flag-leaf phase: leaf tip dieback and large merged necrotic areas during grain fill.',
    ],
    conditions: {
      temp_lo: 10, temp_hi: 25, temp_note: 'cool-to-mild; optimum 15–20 °C',
      wetness: '8 h+ leaf wetness', lag_days: 21,
      other: 'Rain splash spreads spores upward through the canopy; dense canopies and old cereal residue on the surface multiply risk. Long ~21-day latency hides infections already underway.',
    },
    spectral: 'Septoria\u2019s necrotic speckle reads as a broad visible-band decline: NDVI and GNDVI both sag while the red edge stays comparatively stable — the model pairs this with texture (speckle) cues in full SSCNN mode.',
    riskFactors: [
      'Multiple wet days (8 h+ wetness) within the past three weeks',
      'Cereal residue from the previous season left on the soil surface',
      'Dense canopy with low air movement; frequent light rain rather than one heavy storm',
      'Variety with weak septoria standing',
    ],
    management: {
      cultural: [
        'Walk the LOWER canopy of flagged tiles — lesions start low; strip leaves to read the progress up the stem (bottom three leaves infected = treatable window may already be closing).',
        'Manage residue: plough-in or remove cereal stubble, rotate out of wheat for a season in hotspot blocks.',
        'Delay sowing slightly in severe-pressure fields to shorten the autumn splash window.',
      ],
      chemical: [
        'T0/T1 protectant timing pays most: triazole (prothioconazole/tebuconazole) plus a multisite partner (chlorothalonil where registered) — treat BEFORE the flag leaf is infected, not after.',
        'Once >50% of the flag leaf is necrotic, fungicide returns are marginal — protect the next leaf layer instead.',
      ],
      resistance: [
        'Choose varieties with good septoria ratings; they shift spray timings later and buy margin.',
        'Rotate fungicide mode-of-action groups — septoria resistance to single-site chemistry is widespread.',
      ],
    },
    scoutingTip: 'Black pepper-dots inside grey lesions confirm septoria (not rust). Because symptoms lag infection by ~3 weeks, act on the weather-driven risk flag, not on visible disease alone.',
  },

  fusarium: {
    label: 'Fusarium head blight (scab)',
    pathogen: 'Fusarium graminearum / F. culmorum',
    hue: '#C2527B',
    icon: '🩷',
    severity: 'Head-level disease: direct yield loss plus DON (vomitoxin) contamination that can reject an entire grain lot at market.',
    summary: 'Head blight infects the spike at flowering. One rain event at anthesis can bleach a whole head in days. Because the damage is at head level and patchy, canopy reflectance barely sees it — the model therefore leans on the weather-driven infection window, and the crop stage decides everything.',
    symptoms: [
      'One or more spikelets bleaching prematurely while the rest of the head stays green — often starting where a glume was wetted.',
      'Pink/salmon spore masses at the base of infected glumes in humid mornings.',
      'Shrivelled, chalk-white "tombstone" grains at harvest; kernels lightweight and rough.',
      'Stem-base browning (crown rot) in the same fields signals residue-borne inoculum.',
    ],
    conditions: {
      temp_lo: 20, temp_hi: 30, temp_note: 'warm; optimum ~25 °C',
      wetness: '6 h+ wetness on the head', lag_days: 10,
      other: 'Rain or very humid days DURING ANTHESIS are the decisive trigger; maize or cereal residue on the surface supplies the inoculum. Irrigated fields carry extra risk.',
    },
    spectral: 'FHB gives the weakest canopy signal of the five (head-level, patchy): the spectral proxy deliberately weighs it low and lets the warm, wet anthesis window dominate the risk — check the growth stage.',
    riskFactors: [
      'Crop at anthesis (heading/flowering) when warm wet weather hits',
      'Maize or cereal residue from the previous season on the surface',
      'Warm (20–30 °C) humid spell with 6 h+ head wetness in the past 10 days',
      'Sprinkler irrigation during flowering; susceptible variety',
    ],
    management: {
      cultural: [
        'If the crop is AT anthesis and the risk flag is high, a pre-rain protective spray is the single most effective action — timing beats product choice.',
        'Rotate away from maize; incorporate or remove infested residue; avoid irrigating during flowering.',
        'Plan harvest segregation: combine-blend raises DON averages — keep hotspots separate and test.',
      ],
      chemical: [
        'At-anthesis triazoles with FHB activity (prothioconazole, metconazole, tebuconazole) within the first 48–72 h of flowering onset; strobilurins alone are NOT recommended (can raise DON).',
        'Ensure head coverage (higher water volume, forward nozzles).',
      ],
      resistance: [
        'Grow moderately FHB-tolerant varieties (Type II spread resistance) where available.',
        'At harvest, test DON on lots from flagged blocks before marketing; clean/tombstone-separate grain.',
      ],
    },
    scoutingTip: 'Count bleached spikelets on 10 heads per flagged tile during grain fill; >10% symptomatic heads means the lot needs a DON test before sale. At anthesis, watch the forecast, not the symptoms.',
  },
};

const HEALTHY_NOTE = 'A tile with no dominant disease signal means the canopy spectra sit inside the healthy baseline and weather pressure is low. Keep the routine monitoring cadence — absence of signal is not absence of risk in the next infection window.';

/* ------------------------------------------------------------------ *
 * Rendering                                                           *
 * ------------------------------------------------------------------ */

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;',
  }[c]));
}

function counts() {
  const s = window.WDED_SUMMARY || {};
  return s.dominant_disease_counts || {};
}

function chip(text, hue) {
  return '<span class="kbChip" style="border-color:' + hue + ';color:' + hue + '">' + esc(text) + '</span>';
}

function card(id) {
  const d = KB[id];
  const n = counts()[id] || 0;
  const flag = n > 0
    ? '<span class="kbFlag on">flagged in ' + n + ' tile' + (n > 1 ? 's' : '') + ' this flight</span>'
    : '<span class="kbFlag">not dominant in latest flight</span>';
  return '<button class="kbCard" data-d="' + id + '" style="border-top:4px solid ' + d.hue + '">' +
    '<span class="kbIcon">' + d.icon + '</span>' +
    '<span class="kbName">' + esc(d.label) + '</span>' +
    '<span class="kbPath">' + esc(d.pathogen) + '</span>' +
    '<span class="kbSev">' + esc(d.severity.split('—')[0].split(':')[0]) + '</span>' +
    flag + '</button>';
}

function renderList() {
  const view = document.getElementById('kbView');
  if (!view) return;
  const c = counts();
  const order = DISEASES.slice().sort((a, b) => (c[b] || 0) - (c[a] || 0));
  view.innerHTML =
    '<div class="kbIntro"><h2>Disease guide</h2>' +
    '<p>Field reference for the five diseases the SSCNN model distinguishes, correlated with the latest ' +
    'flight\u2019s predictions: each card shows whether the disease was the <b>dominant risk driver</b> in ' +
    'flagged tiles. Open a card for symptoms, the exact infection conditions the weather prior evaluates, ' +
    'what the multispectral model sees, and management options.</p>' +
    (Object.keys(c).length ? '<p class="kbFine">Latest flight: ' +
      order.filter((d) => c[d]).map((d) => esc(KB[d].label.split(' (')[0]) + ' × ' + c[d]).join(' · ') +
      '</p>' : '') + '</div>' +
    '<div class="kbGrid">' + order.map(card).join('') + '</div>' +
    '<div class="kbHealthy">' + esc(HEALTHY_NOTE) + '</div>';
  view.querySelectorAll('.kbCard').forEach((el) => {
    el.addEventListener('click', () => renderDetail(el.dataset.d));
  });
}

function renderDetail(id) {
  const d = KB[id];
  if (!d) { renderList(); return; }
  const n = counts()[id] || 0;
  const corr = n > 0
    ? '<div class="kbCorr on"><b>Model correlation:</b> dominant risk driver in <b>' + n + '</b> tile' + (n > 1 ? 's' : '') +
      ' of the latest flight — compare its flagged locations on the risk map. ' +
      '<button class="act" id="kb_map">open risk map</button></div>'
    : '<div class="kbCorr"><b>Model correlation:</b> not the dominant driver in the latest flight — keep it in mind after the next warm/wet infection window.</div>';

  document.getElementById('kbView').innerHTML =
    '<button class="ghost" id="kb_back">← all diseases</button>' +
    '<article class="kbDetail" style="border-top:6px solid ' + d.hue + '">' +
    '<header class="kbHead"><span class="kbIcon lg">' + d.icon + '</span><div>' +
    '<h2>' + esc(d.label) + '</h2><div class="kbPath">' + esc(d.pathogen) + '</div></div></header>' +
    '<p class="kbSevLine">' + esc(d.severity) + '</p>' +
    '<p>' + esc(d.summary) + '</p>' +
    corr +

    '<h3>Symptoms — what to look for</h3><ul>' + d.symptoms.map((s) => '<li>' + esc(s) + '</li>').join('') + '</ul>' +

    '<h3>Infection conditions (what the weather prior evaluates)</h3>' +
    '<div class="kbConds">' +
    chip('temp ' + d.conditions.temp_lo + '–' + d.conditions.temp_hi + ' °C · ' + d.conditions.temp_note, '#B4632C') +
    chip('leaf wetness: ' + esc(d.conditions.wetness), '#4A7FB5') +
    chip('lag to symptoms ≈ ' + d.conditions.lag_days + ' days', '#3D6B35') +
    '</div><p class="kbOther">' + esc(d.conditions.other) + '</p>' +

    '<h3>What the multispectral model sees</h3><p>' + esc(d.spectral) + '</p>' +

    '<h3>Local risk factors</h3><ul>' + d.riskFactors.map((s) => '<li>' + esc(s) + '</li>').join('') + '</ul>' +

    '<h3>Management</h3>' +
    '<div class="kbMgmt">' +
    '<div><h4>Cultural & scouting</h4><ul>' + d.management.cultural.map((s) => '<li>' + esc(s) + '</li>').join('') + '</ul></div>' +
    '<div><h4>Chemical</h4><ul>' + d.management.chemical.map((s) => '<li>' + esc(s) + '</li>').join('') + '</ul></div>' +
    '<div><h4>Resistance & longer term</h4><ul>' + d.management.resistance.map((s) => '<li>' + esc(s) + '</li>').join('') + '</ul></div>' +
    '</div>' +

    '<div class="kbTip"><b>Scouting tip — </b>' + esc(d.scoutingTip) + '</div>' +
    '<p class="kbFine">Decision support only. Always confirm symptoms on the ground and follow the product label and ' +
    'local extension advice before any treatment.</p>' +
    '</article>';

  document.getElementById('kb_back').addEventListener('click', renderList);
  const mapBtn = document.getElementById('kb_map');
  if (mapBtn) {
    mapBtn.addEventListener('click', () => {
      if (window.WDED_SHOW_TAB) window.WDED_SHOW_TAB('map');
    });
  }
  if (window.WDED_SHOW_TAB) window.WDED_SHOW_TAB('knowledge');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* External entry point: map popups + submission results */
window.WDED_KB_OPEN = function (id) {
  if (KB[id]) renderDetail(id);
  else renderList();
  return false; // for href="#" links
};

window.addEventListener('wded:focus-disease', (e) => {
  if (e.detail && KB[e.detail.disease]) renderDetail(e.detail.disease);
});

function initKnowledge() {
  if (!document.getElementById('kbView')) return;
  renderList();
  /* re-render badges when a new run summary arrives */
  window.addEventListener('wded:summary-loaded', renderList);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initKnowledge);
} else {
  initKnowledge();
}

/* Test hooks */
window.WDED_KB = { KB, DISEASES };
})();
