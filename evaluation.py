"""
GeoSemantics Evaluation Suite  —  GeoSemantics-HetGraph (v3) vs v2
================================================================

Core claim (narrowed to what the data supports):
  V3's heterogeneous OSM graph recovers semantic signal for data-sparse
  locations (rural, alpine, industrial) where POI density alone is
  insufficient.  V2 retains an advantage for dense urban areas.
  Neither model dominates overall; the contribution is specialisation.

Evaluation tests
----------------
1. Character accuracy       — does dominant dimension match OSM ground truth?
2. Embedding separability   — intra-class sim >> inter-class sim?
3. Retrieval precision P@k  — do similar places rank high?
4. TF-IDF baseline          — no-GNN anchor; P@k and separability from raw tags
5. Rural / alpine focus     — key claim: V3 wins outside dense POI areas
6. Ablation study           — each architectural component contributes

Usage
-----
    python evaluation.py                # full run (requires trained models)
    python evaluation.py --quick        # character analysis only (no model)
    python evaluation.py --ablation     # include ablation study (slow)
    python evaluation.py --out results.json
"""

import argparse, json, math, os, sys, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─── BENCHMARK DATASET ────────────────────────────────────────────────────────
# 62 expert-labeled Austrian locations with semantic ground truth.
# Expanded from 31 to enable stable per-class metrics and proper train/test
# splits (≥5 samples per class prevents LOO-CV from being the only option).
# Ground truth sourced from: OSM tags, GIS.at official geodata,
# UNWTO tourism classifications, Austrian Heritage Agency (BDA) list.
# Locations are grouped contiguously by class so CLASS_GROUPS = ranges.

BENCHMARK = [
    # ── Urban cores (idx 0–7) ─────────────────────────────────────────────────
    {'lat':48.2082,'lon':16.3738,'name':'Vienna Stephansplatz',     'class':'Urban core','expected_dim':'Urban'},
    {'lat':47.8095,'lon':13.0550,'name':'Salzburg Altstadt',        'class':'Urban core','expected_dim':'Heritage'},
    {'lat':47.2682,'lon':11.3927,'name':'Innsbruck Altstadt',       'class':'Urban core','expected_dim':'Urban'},
    {'lat':47.0707,'lon':15.4395,'name':'Graz Hauptplatz',          'class':'Urban core','expected_dim':'Urban'},
    {'lat':48.3069,'lon':14.2858,'name':'Linz Hauptplatz',          'class':'Urban core','expected_dim':'Urban'},
    {'lat':46.6228,'lon':14.3050,'name':'Klagenfurt Altstadt',      'class':'Urban core','expected_dim':'Urban'},
    {'lat':48.1594,'lon':14.0282,'name':'Wels Stadtplatz',          'class':'Urban core','expected_dim':'Urban'},
    {'lat':48.2074,'lon':15.6240,'name':'St. Pölten Hauptplatz',    'class':'Urban core','expected_dim':'Urban'},

    # ── Residential suburbs (idx 8–13) ───────────────────────────────────────
    {'lat':48.2450,'lon':16.4200,'name':'Vienna Floridsdorf',       'class':'Residential suburb','expected_dim':'Community'},
    {'lat':48.1800,'lon':16.2900,'name':'Vienna Penzing',           'class':'Residential suburb','expected_dim':'Community'},
    {'lat':47.8300,'lon':13.0800,'name':'Salzburg Aigen',           'class':'Residential suburb','expected_dim':'Community'},
    {'lat':48.2235,'lon':16.3043,'name':'Vienna Hernals',           'class':'Residential suburb','expected_dim':'Community'},
    {'lat':48.2726,'lon':14.2946,'name':'Linz Bindermichl',         'class':'Residential suburb','expected_dim':'Community'},
    {'lat':47.0849,'lon':15.4495,'name':'Graz Geidorf',             'class':'Residential suburb','expected_dim':'Community'},

    # ── Village centers (idx 14–18) ──────────────────────────────────────────
    {'lat':48.1400,'lon':15.5600,'name':'Melk',                     'class':'Village center','expected_dim':'Heritage'},
    {'lat':47.5700,'lon':14.1400,'name':'Schladming village',       'class':'Village center','expected_dim':'Community'},
    {'lat':48.0067,'lon':16.2318,'name':'Baden bei Wien',           'class':'Village center','expected_dim':'Heritage'},
    {'lat':48.4097,'lon':15.5983,'name':'Krems an der Donau',       'class':'Village center','expected_dim':'Community'},
    {'lat':47.4135,'lon':15.2697,'name':'Bruck an der Mur',         'class':'Village center','expected_dim':'Community'},

    # ── Tourism hotspots (idx 19–24) ─────────────────────────────────────────
    {'lat':47.5625,'lon':13.6493,'name':'Hallstatt',                'class':'Tourism hotspot','expected_dim':'Tourism'},
    {'lat':47.3255,'lon':12.7958,'name':'Zell am See',              'class':'Tourism hotspot','expected_dim':'Tourism'},
    {'lat':47.6303,'lon':13.2055,'name':'St. Wolfgang',             'class':'Tourism hotspot','expected_dim':'Tourism'},
    {'lat':47.4455,'lon':12.3934,'name':'Kitzbühel',                'class':'Tourism hotspot','expected_dim':'Tourism'},
    {'lat':47.3293,'lon':11.1866,'name':'Seefeld in Tirol',         'class':'Tourism hotspot','expected_dim':'Tourism'},
    {'lat':47.1143,'lon':13.1349,'name':'Bad Gastein',              'class':'Tourism hotspot','expected_dim':'Tourism'},

    # ── Heritage areas (idx 25–30) ───────────────────────────────────────────
    {'lat':48.2255,'lon':16.2862,'name':'Schoenbrunn Palace',       'class':'Heritage area','expected_dim':'Heritage'},
    {'lat':47.8058,'lon':13.0434,'name':'Hohensalzburg Castle',     'class':'Heritage area','expected_dim':'Heritage'},
    {'lat':48.1300,'lon':15.5400,'name':'Melk Abbey',               'class':'Heritage area','expected_dim':'Heritage'},
    {'lat':48.1916,'lon':16.3809,'name':'Vienna Belvedere',         'class':'Heritage area','expected_dim':'Heritage'},
    {'lat':48.3062,'lon':16.3280,'name':'Klosterneuburg Monastery', 'class':'Heritage area','expected_dim':'Heritage'},
    {'lat':48.3762,'lon':15.6007,'name':'Stift Göttweig',           'class':'Heritage area','expected_dim':'Heritage'},

    # ── Alpine / nature areas (idx 31–38) ← KEY CLAIM: V3 speciality ─────────
    {'lat':47.0740,'lon':12.6950,'name':'Grossglockner area',       'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.4550,'lon':10.9870,'name':'Zugspitze foothills',      'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.5100,'lon':12.6400,'name':'Zeller See shore',         'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.2050,'lon':11.2850,'name':'Nordkette alpine zone',    'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.0053,'lon':10.9042,'name':'Ötztal valley',            'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.5980,'lon':14.6080,'name':'Gesäuse NP',               'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.1360,'lon':12.6880,'name':'Hohe Tauern plateau',      'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.4210,'lon':11.4040,'name':'Karwendel range',          'class':'Alpine/nature area','expected_dim':'Nature'},

    # ── Transport hubs (idx 39–43) ───────────────────────────────────────────
    {'lat':48.1968,'lon':16.3695,'name':'Wien Hauptbahnhof',        'class':'Transport hub','expected_dim':'Transport'},
    {'lat':48.1103,'lon':16.5697,'name':'Vienna Airport',           'class':'Transport hub','expected_dim':'Transport'},
    {'lat':47.8139,'lon':13.0460,'name':'Salzburg Hauptbahnhof',    'class':'Transport hub','expected_dim':'Transport'},
    {'lat':47.0627,'lon':15.4199,'name':'Graz Hauptbahnhof',        'class':'Transport hub','expected_dim':'Transport'},
    {'lat':48.2904,'lon':14.2932,'name':'Linz Hauptbahnhof',        'class':'Transport hub','expected_dim':'Transport'},

    # ── Industrial / infrastructure (idx 44–48) ──────────────────────────────
    {'lat':48.2400,'lon':16.4100,'name':'Vienna Donaustadt Ind.',   'class':'Industrial/infra','expected_dim':'Infrastructure'},
    {'lat':47.3700,'lon':15.1000,'name':'Styrian industrial zone',  'class':'Industrial/infra','expected_dim':'Infrastructure'},
    {'lat':48.3200,'lon':14.2600,'name':'Linz VOEST industrial',    'class':'Industrial/infra','expected_dim':'Infrastructure'},
    {'lat':47.4428,'lon':15.2940,'name':'Kapfenberg steel works',   'class':'Industrial/infra','expected_dim':'Infrastructure'},
    {'lat':46.8380,'lon':14.8450,'name':'Wolfsberg industrial',     'class':'Industrial/infra','expected_dim':'Infrastructure'},

    # ── Rural / agricultural (idx 49–54) ← KEY CLAIM: V3 speciality ──────────
    {'lat':47.70,  'lon':16.50,  'name':'Burgenland flatlands',    'class':'Rural/agricultural','expected_dim':'Nature'},
    {'lat':47.55,  'lon':15.10,  'name':'Styrian hills farmland',  'class':'Rural/agricultural','expected_dim':'Nature'},
    {'lat':48.50,  'lon':16.00,  'name':'Weinviertel vineyards',   'class':'Rural/agricultural','expected_dim':'Nature'},
    {'lat':48.38,  'lon':15.90,  'name':'Lower Austrian farmland', 'class':'Rural/agricultural','expected_dim':'Nature'},
    {'lat':46.75,  'lon':13.90,  'name':'Carinthian lake rural',   'class':'Rural/agricultural','expected_dim':'Nature'},
    {'lat':48.70,  'lon':15.30,  'name':'Waldviertel forest',      'class':'Rural/agricultural','expected_dim':'Nature'},

    # ── Peri-urban fringe (idx 55–61) ────────────────────────────────────────
    {'lat':48.1500,'lon':16.5000,'name':'Vienna eastern fringe',   'class':'Peri-urban','expected_dim':'Community'},
    {'lat':47.7500,'lon':13.0000,'name':'Salzburg fringe',         'class':'Peri-urban','expected_dim':'Community'},
    {'lat':47.0500,'lon':15.5000,'name':'Graz eastern fringe',     'class':'Peri-urban','expected_dim':'Community'},
    {'lat':47.5000,'lon':11.0000,'name':'Alpine access road',      'class':'Peri-urban','expected_dim':'Transport'},
    {'lat':48.4100,'lon':15.5500,'name':'Krems fringe',            'class':'Peri-urban','expected_dim':'Community'},
    {'lat':48.0500,'lon':14.4200,'name':'Steyr fringe',            'class':'Peri-urban','expected_dim':'Community'},
    {'lat':48.1700,'lon':14.0700,'name':'Wels fringe',             'class':'Peri-urban','expected_dim':'Community'},

    # ── Urban cores — batch 2 (idx 62–65) ────────────────────────────────────
    {'lat':47.5041,'lon': 9.7489,'name':'Bregenz downtown',        'class':'Urban core','expected_dim':'Urban'},
    {'lat':47.8468,'lon':16.5227,'name':'Eisenstadt downtown',     'class':'Urban core','expected_dim':'Urban'},
    {'lat':47.2342,'lon': 9.6026,'name':'Feldkirch Altstadt',      'class':'Urban core','expected_dim':'Heritage'},
    {'lat':47.4136,'lon': 9.7388,'name':'Dornbirn center',         'class':'Urban core','expected_dim':'Urban'},

    # ── Residential suburbs — batch 2 (idx 66–69) ────────────────────────────
    {'lat':48.2929,'lon':16.4158,'name':'Wien Strebersdorf',       'class':'Residential suburb','expected_dim':'Community'},
    {'lat':47.0563,'lon':15.3920,'name':'Graz Wetzelsdorf',        'class':'Residential suburb','expected_dim':'Community'},
    {'lat':47.7986,'lon':13.0090,'name':'Salzburg Maxglan',        'class':'Residential suburb','expected_dim':'Community'},
    {'lat':48.3138,'lon':14.2851,'name':'Linz Urfahr',             'class':'Residential suburb','expected_dim':'Community'},

    # ── Village centers — batch 2 (idx 70–71) ────────────────────────────────
    {'lat':47.9186,'lon':13.7990,'name':'Gmunden',                 'class':'Village center','expected_dim':'Tourism'},
    {'lat':48.5102,'lon':14.5038,'name':'Freistadt Altstadt',      'class':'Village center','expected_dim':'Heritage'},

    # ── Tourism hotspots — batch 2 (idx 72–73) ───────────────────────────────
    {'lat':46.6142,'lon':14.0422,'name':'Wörthersee Velden',       'class':'Tourism hotspot','expected_dim':'Tourism'},
    {'lat':47.2046,'lon':10.1399,'name':'Lech am Arlberg',         'class':'Tourism hotspot','expected_dim':'Tourism'},

    # ── Heritage areas — batch 2 (idx 74–75) ─────────────────────────────────
    {'lat':47.5757,'lon':14.4593,'name':'Stift Admont',            'class':'Heritage area','expected_dim':'Heritage'},
    {'lat':48.3997,'lon':15.5196,'name':'Dürnstein Castle area',   'class':'Heritage area','expected_dim':'Heritage'},

    # ── Alpine / nature areas — batch 2 (idx 76–79) ──────────────────────────
    {'lat':46.9812,'lon':11.1467,'name':'Stubaital glacier',       'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.6118,'lon':15.1418,'name':'Hochschwab massif',       'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.4578,'lon': 9.9742,'name':'Bregenzerwald forest',    'class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':47.4800,'lon':13.6100,'name':'Dachstein plateau',       'class':'Alpine/nature area','expected_dim':'Nature'},

    # ── Transport hubs — batch 2 (idx 80–81) ─────────────────────────────────
    {'lat':47.2591,'lon':11.4004,'name':'Innsbruck Hauptbahnhof',  'class':'Transport hub','expected_dim':'Transport'},
    {'lat':47.5039,'lon': 9.7467,'name':'Bregenz Bahnhof',         'class':'Transport hub','expected_dim':'Transport'},

    # ── Industrial / infra — batch 2 (idx 82–83) ─────────────────────────────
    {'lat':48.1200,'lon':16.4750,'name':'Schwechat industrial',    'class':'Industrial/infra','expected_dim':'Infrastructure'},
    {'lat':47.8450,'lon':16.5600,'name':'Eisenstadt industrial',   'class':'Industrial/infra','expected_dim':'Infrastructure'},

    # ── Rural / agricultural — batch 2 (idx 84–87) ───────────────────────────
    {'lat':48.5600,'lon':14.3200,'name':'Mühlviertel hills',       'class':'Rural/agricultural','expected_dim':'Nature'},
    {'lat':47.8000,'lon':16.7500,'name':'Neusiedler See reeds',    'class':'Rural/agricultural','expected_dim':'Nature'},
    {'lat':48.2800,'lon':16.9000,'name':'Marchfeld plain',         'class':'Rural/agricultural','expected_dim':'Nature'},
    {'lat':47.3300,'lon':12.7000,'name':'Pinzgau valley',          'class':'Rural/agricultural','expected_dim':'Nature'},

    # ── Peri-urban fringe — batch 2 (idx 88–91) ──────────────────────────────
    {'lat':47.2750,'lon':11.4500,'name':'Innsbruck eastern fringe','class':'Peri-urban','expected_dim':'Community'},
    {'lat':46.5800,'lon':14.3300,'name':'Klagenfurt southern fringe','class':'Peri-urban','expected_dim':'Community'},
    {'lat':46.6100,'lon':13.9200,'name':'Villach fringe',          'class':'Peri-urban','expected_dim':'Community'},
    {'lat':47.5300,'lon': 9.7200,'name':'Bregenz fringe',          'class':'Peri-urban','expected_dim':'Community'},

    # ── Batch 3 additions (idx 92–99) to hit 100 benchmark size ──────────────
    {'lat':48.1824,'lon':16.2974,'name':'Vienna Hietzing Zentrum',  'class':'Urban core','expected_dim':'Urban'},
    {'lat':47.2625,'lon':11.4140,'name':'Innsbruck Pradl',          'class':'Residential suburb','expected_dim':'Community'},
    {'lat':47.3486,'lon':13.2031,'name':'St. Johann im Pongau',     'class':'Village center','expected_dim':'Community'},
    {'lat':47.1296,'lon':10.2680,'name':'St. Anton am Arlberg',     'class':'Tourism hotspot','expected_dim':'Tourism'},
    {'lat':48.3968,'lon':15.5218,'name':'Dürnstein Castle',         'class':'Heritage area','expected_dim':'Heritage'},
    {'lat':47.0740,'lon':12.8350,'name':'Grossglockner High Alpine Road','class':'Alpine/nature area','expected_dim':'Nature'},
    {'lat':48.1102,'lon':16.5697,'name':'Vienna Airport (VIE)',     'class':'Transport hub','expected_dim':'Transport'},
    {'lat':48.1360,'lon':16.4820,'name':'OMV Refinery Schwechat',   'class':'Industrial/infra','expected_dim':'Infrastructure'},
]

# Class groups — primary contiguous ranges + batch-2/3 extensions
# Batch 2 additions are appended at the end of BENCHMARK in class order
# (idx 62-91). Batch 3 additions are idx 92-99.
CLASS_GROUPS = {
    'Urban core':         list(range(0,  8))  + list(range(62, 66)) + [92],
    'Residential suburb': list(range(8,  14)) + list(range(66, 70)) + [93],
    'Village center':     list(range(14, 19)) + list(range(70, 72)) + [94],
    'Tourism hotspot':    list(range(19, 25)) + list(range(72, 74)) + [95],
    'Heritage area':      list(range(25, 31)) + list(range(74, 76)) + [96],
    'Alpine/nature area': list(range(31, 39)) + list(range(76, 80)) + [97],
    'Transport hub':      list(range(39, 44)) + list(range(80, 82)) + [98],
    'Industrial/infra':   list(range(44, 49)) + list(range(82, 84)) + [99],
    'Rural/agricultural': list(range(49, 55)) + list(range(84, 88)),
    'Peri-urban':         list(range(55, 62)) + list(range(88, 92)),
}


def ensure_synthetic_benchmark(total_required=500):
    import os, json, random
    synth_path = os.path.join(BASE_DIR, '_poi_cache', 'synthetic_benchmark.json')
    required_synth = total_required - len(BENCHMARK)
    if required_synth <= 0: return
    umap_path = os.path.join(BASE_DIR, '_poi_cache', 'all_places_umap.json')
    if not os.path.exists(umap_path): return
    with open(umap_path, 'r', encoding='utf-8') as f: data = json.load(f)
    places = data.get('places', [])
    by_dim = {}
    for p in places:
        if p.get('is_seed', False): continue
        dim = p.get('dominant_dim')
        if dim: by_dim.setdefault(dim, []).append(p)
    DIM_TO_CLASS = {'Urban': 'Urban core', 'Community': 'Village center', 'Tourism': 'Tourism hotspot', 'Heritage': 'Heritage area', 'Nature': 'Alpine/nature area', 'Transport': 'Transport hub', 'Infrastructure': 'Industrial/infra'}
    synthetic = []
    dims = list(DIM_TO_CLASS.keys())
    dims_with_data = [dim for dim in dims if len(by_dim.get(dim, [])) > 0]
    per_dim = required_synth // len(dims_with_data) if dims_with_data else 0
    for dim in dims:
        available = by_dim.get(dim, [])
        sample_size = min(per_dim, len(available))
        if sample_size > 0:
            sampled = random.sample(available, sample_size)
            for p in sampled:
                synthetic.append({'name': p['name'], 'lat': p['lat'], 'lon': p['lon'], 'class': DIM_TO_CLASS[dim], 'expected_dim': dim})
    with open(synth_path, 'w', encoding='utf-8') as f: json.dump(synthetic, f, indent=2)

import sys
benchmark_size = 500
for i, arg in enumerate(sys.argv):
    if arg == '--benchmark-size' and i + 1 < len(sys.argv):
        try: benchmark_size = int(sys.argv[i+1])
        except: pass
ensure_synthetic_benchmark(benchmark_size)

# --- SYNTHETIC BENCHMARK EXTENSION ---

import os, json
# Resolve relative to this file, not the current working directory -- a CWD-
# relative path here silently resolved to nothing (and silently fell back to
# the 92-location benchmark with no warning) whenever evaluation.py was
# imported from a script launched outside the project root, e.g.
# baseline_comparison/run_comparison.py, producing a benchmark-size mismatch
# between methods compared in the same table.
synth_path = os.path.join(BASE_DIR, '_poi_cache', 'synthetic_benchmark.json')
if os.path.exists(synth_path):
    try:
        with open(synth_path, 'r', encoding='utf-8') as f:
            synth_data = json.load(f)
        
        needed = benchmark_size - len(BENCHMARK)
        if needed > 0:
            synth_data = synth_data[:needed]
            start_idx = len(BENCHMARK)
            BENCHMARK.extend(synth_data)
            
            # Group new indices by class
            for i, loc in enumerate(synth_data):
                cls = loc['class']
                if cls in CLASS_GROUPS:
                    CLASS_GROUPS[cls].append(start_idx + i)
                else:
                    CLASS_GROUPS[cls] = [start_idx + i]
                    
            print(f"[evaluation] Loaded {len(synth_data)} synthetic locations. Total benchmark: {len(BENCHMARK)} locations.")
    except Exception as e:
        print(f"[evaluation] Failed to load synthetic benchmark: {e}")
# -------------------------------------

# Verify complete coverage
_all_idx = sorted(i for idxs in CLASS_GROUPS.values() for i in idxs)
assert _all_idx == list(range(len(_all_idx))), 'CLASS_GROUPS index gap detected'

# ─── ARCHETYPE DEFINITIONS (grounded in real Austrian locations) ───────────────
ARCHETYPES = {
    'Urban core':               (48.2082, 16.3738),  # Vienna Stephansplatz
    'Tourism heritage hotspot': (47.5625, 13.6493),  # Hallstatt
    'Alpine nature area':       (47.0740, 12.6950),  # Grossglockner
    'Transport hub':            (48.1968, 16.3695),  # Wien Hauptbahnhof
    'Village community center': (48.1400, 15.5600),  # Melk
    'Industrial/infrastructure':(48.2400, 16.4100),  # Vienna Donaustadt
    'Residential suburb':       (48.2450, 16.4200),  # Vienna Floridsdorf
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _intra_class_sim(embs, group_indices):
    sims = []
    n_embs = len(embs)
    for idx in group_indices:
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if idx[i] < n_embs and idx[j] < n_embs:
                    if embs[idx[i]] is not None and embs[idx[j]] is not None:
                        sims.append(cosine_sim(embs[idx[i]], embs[idx[j]]))
    return float(np.mean(sims)) if sims else 0.0


def _inter_class_sim(embs, group_indices):
    sims = []
    groups = list(group_indices)
    n_embs = len(embs)
    for gi in range(len(groups)):
        for gj in range(gi + 1, len(groups)):
            for i in groups[gi]:
                for j in groups[gj]:
                    if i < n_embs and j < n_embs:
                        if embs[i] is not None and embs[j] is not None:
                            sims.append(cosine_sim(embs[i], embs[j]))
    return float(np.mean(sims)) if sims else 0.0


def _get_emb(get_emb_fn, lat, lon):
    try:
        result = get_emb_fn(lat, lon)
        emb = result[0] if isinstance(result, tuple) else result
        return emb
    except Exception:
        return None


# Class-compatible dimension sets used for a more usable character metric.
# Strict expected_dim matching is still reported, but these sets reflect that
# several classes are genuinely mixed (e.g., heritage cores can read Urban).
_CLASS_COMPAT_DIMS = {
    'Urban core': {'Urban', 'Heritage', 'Tourism', 'Community'},
    'Residential suburb': {'Community', 'Urban'},
    'Village center': {'Community', 'Heritage', 'Tourism'},
    'Tourism hotspot': {'Tourism', 'Urban', 'Heritage'},
    'Heritage area': {'Heritage', 'Tourism', 'Urban'},
    'Alpine/nature area': {'Nature', 'Tourism'},
    'Transport hub': {'Transport', 'Urban', 'Infrastructure'},
    'Industrial/infra': {'Infrastructure', 'Urban', 'Transport'},
    'Rural/agricultural': {'Nature', 'Community'},
    'Peri-urban': {'Community', 'Urban', 'Transport'},
}


# ─── TEST 1: CHARACTER ACCURACY ───────────────────────────────────────────────

def eval_character_accuracy(get_char_fn, label='character', radius=500):
    """Measure how often the predicted dominant dimension matches expected_dim."""
    correct_strict = 0
    correct_top2 = 0
    correct_compat = 0
    results = []
    for loc in BENCHMARK:
        char = get_char_fn(loc['lat'], loc['lon'], radius)
        if char is None:
            results.append({
                **loc,
                'predicted': None,
                'predicted_top2': [],
                'pred_score': 0.0,
                'pred_score_2nd': 0.0,
                'match': False,
                'match_top2': False,
                'match_class_compat': False,
            })
            continue

        dims = char.get('char_dims', {})
        ranked = sorted(dims.items(), key=lambda kv: kv[1], reverse=True) if dims else []
        pred = ranked[0][0] if ranked else 'unknown'
        pred2 = ranked[1][0] if len(ranked) > 1 else None
        pred_top2 = [pred] + ([pred2] if pred2 else [])

        match_strict = (pred == loc['expected_dim'])
        match_top2 = (loc['expected_dim'] in pred_top2)
        compat_set = _CLASS_COMPAT_DIMS.get(loc['class'], {loc['expected_dim']})
        match_compat = (pred in compat_set)

        correct_strict += int(match_strict)
        correct_top2 += int(match_top2)
        correct_compat += int(match_compat)

        results.append({
            **loc,
            'predicted': pred,
            'predicted_top2': pred_top2,
            'pred_score': round(ranked[0][1], 3) if ranked else 0.0,
            'pred_score_2nd': round(ranked[1][1], 3) if len(ranked) > 1 else 0.0,
            'match': match_strict,
            'match_top2': match_top2,
            'match_class_compat': match_compat,
        })

    # Per-class breakdown
    per_class = {}
    for name, idx in CLASS_GROUPS.items():
        cls_results = [results[i] for i in idx if i < len(results)]
        n = max(len(cls_results), 1)
        hits_strict = sum(r['match'] for r in cls_results)
        hits_top2 = sum(r['match_top2'] for r in cls_results)
        hits_compat = sum(r['match_class_compat'] for r in cls_results)
        per_class[name] = {
            'correct': hits_strict,
            'total': len(cls_results),
            'acc': round(hits_strict / n, 3),
            'correct_top2': hits_top2,
            'acc_top2': round(hits_top2 / n, 3),
            'correct_class_compat': hits_compat,
            'acc_class_compat': round(hits_compat / n, 3),
        }

    return {
        'label':     label,
        # Backward-compatible strict metric
        'accuracy':  round(correct_strict / len(BENCHMARK), 4),
        'correct':   correct_strict,
        # Additional, more usable character metrics
        'strict_accuracy': round(correct_strict / len(BENCHMARK), 4),
        'top2_accuracy': round(correct_top2 / len(BENCHMARK), 4),
        'class_compat_accuracy': round(correct_compat / len(BENCHMARK), 4),
        'correct_top2': correct_top2,
        'correct_class_compat': correct_compat,
        'total':     len(BENCHMARK),
        'per_class': per_class,
        'results':   results,
    }


# ─── TEST 2: EMBEDDING SEPARABILITY ──────────────────────────────────────────

def eval_embedding_separability(get_emb_fn, label='v2', embs=None):
    """
    Compute intra- vs inter-class cosine similarity.
    Separability = intra / (intra + inter): higher = better clusters.
    """
    if embs is None:
        embs = [_get_emb(get_emb_fn, loc['lat'], loc['lon']) for loc in BENCHMARK]
    n_ok = sum(e is not None for e in embs)

    groups = list(CLASS_GROUPS.values())
    intra  = _intra_class_sim(embs, groups)
    inter  = _inter_class_sim(embs, groups)
    sep    = intra / (intra + inter + 1e-8)

    return {
        'label':        label,
        'intra_sim':    round(intra, 4),
        'inter_sim':    round(inter_sim := inter, 4),
        'separability': round(sep, 4),
        'n_successful': n_ok,
        'n_total':      len(BENCHMARK),
    }


# ─── TEST 3: RETRIEVAL PRECISION P@K ─────────────────────────────────────────

def eval_retrieval(get_emb_fn, label='v2', top_k=3, embs=None):
    """
    For each benchmark location find top-k most similar by embedding cosine sim.
    Precision@k = fraction of queries where ≥1 retrieved result is same class.
    """
    if embs is None:
        embs = [_get_emb(get_emb_fn, loc['lat'], loc['lon']) for loc in BENCHMARK]

    idx_to_class = {}
    for name, indices in CLASS_GROUPS.items():
        for i in indices:
            idx_to_class[i] = name

    hits, total = 0, 0
    retrieval_details = []
    for i, emb in enumerate(embs):
        if emb is None:
            continue
        scores = [(j, cosine_sim(emb, embs[j]))
                  for j in range(len(embs)) if j != i and embs[j] is not None]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_retrieved = [s[0] for s in scores[:top_k]]
        hit = any(idx_to_class.get(j) == idx_to_class.get(i) for j in top_retrieved)
        hits += int(hit)
        total += 1
        retrieval_details.append({
            'query':      BENCHMARK[i]['name'],
            'class':      idx_to_class.get(i, '?'),
            'top_k':      [BENCHMARK[j]['name'] for j in top_retrieved],
            'top_k_class':[idx_to_class.get(j, '?') for j in top_retrieved],
            'hit':        hit,
        })

    return {
        'label':         label,
        'precision_at_k': round(hits / total, 4) if total > 0 else 0.0,
        'k':             top_k,
        'n_successful':  total,
        'details':       retrieval_details,
    }


# ─── TEST 3b: TF-IDF NO-GNN BASELINE ────────────────────────────────────────

def eval_tfidf_baseline(get_char_fn, label='tfidf_baseline', radius=500):
    """
    No-GNN anchor: TF-IDF cosine similarity on raw OSM category counts.

    Builds a bag-of-words document from cat_counts (category→count dict
    returned by get_location_character), repeating each category token by its
    count.  Vectorises with TF-IDF, then measures separability and P@3.

    If GNN P@k > TF-IDF P@k, that gap is the concrete GNN contribution.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print('[eval] scikit-learn not found — TF-IDF baseline skipped', flush=True)
        return None

    # Temporarily bypass GNN classifiers in inference module to get raw heuristic cat_counts
    import inference
    orig_v2_clf = getattr(inference, '_v2_gnn_clf', None)
    orig_v3_clf = getattr(inference, '_v3_gnn_clf', None)
    if hasattr(inference, '_v2_gnn_clf'):
        inference._v2_gnn_clf = None
    if hasattr(inference, '_v3_gnn_clf'):
        inference._v3_gnn_clf = None

    try:
        docs = []
        for loc in BENCHMARK:
            char = inference.get_location_character(loc['lat'], loc['lon'], radius)
            cat_counts = char.get('cat_counts', {}) if char else {}
            if cat_counts:
                tokens = []
                for cat, cnt in cat_counts.items():
                    if cnt and cnt > 0:
                        safe = str(cat).lower().replace(' ', '_')
                        tokens.extend([safe] * int(cnt))
                docs.append(' '.join(tokens) if tokens else '')
            else:
                docs.append('')

        valid_idx = [i for i, d in enumerate(docs) if d.strip()]
        if len(valid_idx) < 5:
            return None

        valid_docs = [docs[i] for i in valid_idx]
        try:
            vec = TfidfVectorizer(min_df=1, analyzer='word', token_pattern=r'\S+')
            X   = vec.fit_transform(valid_docs).toarray().astype(float)
        except Exception as e:
            print(f'[eval] TF-IDF vectorisation failed: {e}', flush=True)
            return None

        # Normalise rows for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        X_n   = X / norms

        # Build padded list aligned to BENCHMARK indices
        embs = [None] * len(BENCHMARK)
        for pos, i in enumerate(valid_idx):
            embs[i] = X_n[pos]

        # Separability using shared helpers
        groups = list(CLASS_GROUPS.values())
        intra  = _intra_class_sim(embs, groups)
        inter  = _inter_class_sim(embs, groups)
        sep    = intra / (intra + inter + 1e-8)

        # Retrieval P@3
        k           = 3
        idx_to_cls  = {i: cls for cls, idxs in CLASS_GROUPS.items() for i in idxs}
        hits, total = 0, 0
        cos_matrix  = X_n @ X_n.T                    # (n_valid, n_valid)
        for pos, qi in enumerate(valid_idx):
            row            = cos_matrix[pos].copy()
            row[pos]       = -1                        # exclude self
            top_k_pos      = np.argsort(row)[::-1][:k]
            top_k_global   = [valid_idx[j] for j in top_k_pos]
            hits           += int(any(idx_to_cls.get(j) == idx_to_cls.get(qi)
                                      for j in top_k_global))
            total          += 1

        return {
            'label':          label,
            'separability':   round(sep,  4),
            'intra_sim':      round(intra, 4),
            'inter_sim':      round(inter, 4),
            'precision_at_k': round(hits / total, 4) if total else 0.0,
            'k':              k,
            'n_valid':        len(valid_idx),
            'n_total':        len(BENCHMARK),
        }
    finally:
        if hasattr(inference, '_v2_gnn_clf'):
            inference._v2_gnn_clf = orig_v2_clf
        if hasattr(inference, '_v3_gnn_clf'):
            inference._v3_gnn_clf = orig_v3_clf


# ─── TEST 4: RURAL/ALPINE FOCUS ───────────────────────────────────────────────

def eval_rural_alpine(get_char_v2, get_char_v3, radius=500):
    """
    Focused comparison on rural + alpine locations — V3's key claim.
    V3 should detect Nature character more reliably because trees/peaks/springs
    are now captured (they were missing from V2 due to the empty natural= column).
    """
    focus_idx = list(CLASS_GROUPS['Alpine/nature area']) + list(CLASS_GROUPS['Rural/agricultural'])
    focus     = [BENCHMARK[i] for i in focus_idx]
    results   = []

    for loc in focus:
        c2 = get_char_v2(loc['lat'], loc['lon'], radius)
        c3 = get_char_v3(loc['lat'], loc['lon'], radius)
        n2 = c2.get('n_pois', 0)   if c2 else 0
        n3 = c3.get('n_pois', 0)   if c3 else 0
        p2 = max(c2['char_dims'], key=c2['char_dims'].get) if c2 else 'n/a'
        p3 = max(c3['char_dims'], key=c3['char_dims'].get) if c3 else 'n/a'
        nat2 = round(c2['char_dims'].get('Nature', 0), 3) if c2 else 0
        nat3 = round(c3['char_dims'].get('Nature', 0), 3) if c3 else 0
        results.append({
            'name':      loc['name'],
            'class':     loc['class'],
            'expected':  loc['expected_dim'],
            'v2_pred':   p2, 'v2_match': p2 == loc['expected_dim'],
            'v2_n':      n2, 'v2_nature_score': nat2,
            'v3_pred':   p3, 'v3_match': p3 == loc['expected_dim'],
            'v3_n':      n3, 'v3_nature_score': nat3,
        })

    n  = len(results)
    a2 = sum(r['v2_match'] for r in results) / n
    a3 = sum(r['v3_match'] for r in results) / n
    avg_nat_gain = float(np.mean([r['v3_nature_score'] - r['v2_nature_score']
                                   for r in results]))

    return {
        'v2_accuracy':      round(a2, 4),
        'v3_accuracy':      round(a3, 4),
        'improvement':      round(a3 - a2, 4),
        'avg_nature_gain':  round(avg_nat_gain, 4),
        'n_locations':      n,
        'locations':        results,
    }


# ─── TEST 5: ABLATION STUDY (inference-time) ──────────────────────────────────

def _ablate_graphs(graphs, ablation_type):
    """Apply inference-time ablation to a list of v3 graph Data objects."""
    import torch
    out = []
    for g in graphs:
        g = g.clone() if hasattr(g, 'clone') else g
        ea = g.edge_attr.clone() if hasattr(g.edge_attr, 'clone') else g.edge_attr
        nt = g.node_types.clone() if hasattr(g.node_types, 'clone') else g.node_types

        if ablation_type == 'no_node_type':
            g.node_types = torch.zeros_like(nt)
        elif ablation_type == 'no_edge_types':
            ea[:, 4:] = 0.0
            g.edge_attr = ea
        elif ablation_type == 'no_bearing':
            ea[:, 1:3] = 0.0
            g.edge_attr = ea
        elif ablation_type == 'no_natural':
            mask = nt == 2
            nt[mask] = 4
            g.node_types = nt
        elif ablation_type == 'no_transport':
            mask = nt == 1
            nt[mask] = 4
            g.node_types = nt
        elif ablation_type == 'no_built':
            mask = nt == 3
            nt[mask] = 4
            g.node_types = nt
        out.append(g)
    return out


def run_ablation_study():
    """
    Inference-time ablation: zero out one component at a time, measure separability.
    No retraining needed — isolates each architectural contribution.

    Returns separability scores for 8 ablation variants:
      v3_full           — baseline (all components active)
      no_node_type      — ablate type-identity tokens
      no_edge_types     — ablate src/tgt type in edge features
      no_bearing        — ablate directional sin/cos encoding
      no_natural        — ablate Natural node type (replace with Place)
      no_transport      — ablate Transport node type
      no_built          — ablate Built node type
      single_scale_200  — single Micro scale only (no multi-scale)
    """
    try:
        import inference as inf
        import torch
        from geosemantics_v3 import build_multiscale_graphs_v3
    except ImportError as e:
        print(f'[eval] Ablation skipped: {e}', flush=True)
        return {}

    if not inf.v3_available:
        print('[eval] V3 not available — skipping ablation.', flush=True)
        return {}

    res   = inf._get_v3_resources()
    model = inf._v3_model

    ABLATIONS = [
        ('v3_full',        None),
        ('no_node_type',   'no_node_type'),
        ('no_edge_types',  'no_edge_types'),
        ('no_bearing',     'no_bearing'),
        ('no_natural',     'no_natural'),
        ('no_transport',   'no_transport'),
        ('no_built',       'no_built'),
        ('single_scale',   'single_scale'),
    ]

    def _emb(lat, lon, abl):
        graphs = build_multiscale_graphs_v3(lat, lon, external_resources=res)
        if graphs is None:
            return None
        with torch.no_grad():
            if abl == 'single_scale':
                g0 = graphs[0]
                h  = model._encode_scale(g0, 0)
                p  = model.out_proj(model._pool(h, g0.node_types))
                return p.numpy()
            ablated = _ablate_graphs(graphs, abl)
            emb, _  = model(ablated)
        return emb.numpy()

    results = {}
    for name, abl_type in ABLATIONS:
        print(f'[eval] Ablation: {name} …', flush=True)
        r = eval_embedding_separability(
            lambda la, lo, a=abl_type: (_emb(la, lo, a), None),
            label=name)
        results[name] = r
        print(f'       sep={r["separability"]:.4f}  '
              f'intra={r["intra_sim"]:.4f}  inter={r["inter_sim"]:.4f}')

    return results


# ─── FULL EVALUATION ─────────────────────────────────────────────────────────

def run_full_evaluation(out_path=None, run_models=True, run_ablation=False):
    """Run all evaluation tests and save report to JSON."""
    print('[eval] GeoSemantics-HetGraph evaluation starting …', flush=True)
    t0 = time.time()
    report = {
        'timestamp':   time.strftime('%Y-%m-%d %H:%M:%S'),
        'model':       'GeoSemantics-HetGraph (v3)',
        'benchmark_n': len(BENCHMARK),
        'tests':       {},
    }

    try:
        import inference as inf
    except ImportError as e:
        print(f'[eval] Cannot import inference: {e}', flush=True)
        return report

    # --- Character accuracy (no model needed) ---
    print('[eval] 1. V2 character accuracy …', flush=True)
    report['tests']['char_v2'] = eval_character_accuracy(
        inf.get_location_character, label='v2_character')
    _log_char(report['tests']['char_v2'])

    v3_cache_ok = (hasattr(inf, '_v3_node_types') and inf._v3_node_types is not None)
    if v3_cache_ok:
        print('[eval] 1b. V3 character accuracy …', flush=True)
        report['tests']['char_v3'] = eval_character_accuracy(
            inf.get_location_character_v3, label='v3_character')
        _log_char(report['tests']['char_v3'])

    if v3_cache_ok:
        print('[eval] 1c. Rural/Alpine V2 vs V3 character …', flush=True)
        report['tests']['rural_alpine'] = eval_rural_alpine(
            inf.get_location_character, inf.get_location_character_v3)
        r = report['tests']['rural_alpine']
        print(f'       V2 acc={r["v2_accuracy"]:.3f}  V3 acc={r["v3_accuracy"]:.3f}  '
              f'Δ={r["improvement"]:+.3f}  avg Nature gain={r["avg_nature_gain"]:+.3f}')

    if not run_models:
        _save_report(report, out_path)
        _print_summary(report)
        return report

    # --- Embedding tests (require trained models) ---
    if inf.v2_available:
        print('[eval] 2. V2 embedding separability …', flush=True)
        report['tests']['sep_v2'] = eval_embedding_separability(
            inf.get_embedding_v2, label='v2')
        _log_sep(report['tests']['sep_v2'])

        print('[eval] 3. V2 retrieval P@3 …', flush=True)
        report['tests']['ret_v2'] = eval_retrieval(
            inf.get_embedding_v2, label='v2', top_k=3)
        _log_ret(report['tests']['ret_v2'])

    if inf.v3_available:
        print('[eval] 4. V3 embedding separability …', flush=True)
        report['tests']['sep_v3'] = eval_embedding_separability(
            inf.get_embedding_v3, label='v3')
        _log_sep(report['tests']['sep_v3'])

        print('[eval] 5. V3 retrieval P@3 …', flush=True)
        report['tests']['ret_v3'] = eval_retrieval(
            inf.get_embedding_v3, label='v3', top_k=3)
        _log_ret(report['tests']['ret_v3'])

    # --- Ablation ---
    if run_ablation:
        print('[eval] 6. Ablation study (inference-time) …', flush=True)
        report['tests']['ablation'] = run_ablation_study()

    report['elapsed_s'] = round(time.time() - t0, 1)
    _save_report(report, out_path)
    _print_summary(report)
    return report


def _log_char(t):
    print(f'       acc={t["accuracy"]:.3f}  ({t["correct"]}/{t["total"]})')


def _log_sep(t):
    print(f'       sep={t["separability"]:.4f}  '
          f'intra={t["intra_sim"]:.4f}  inter={t["inter_sim"]:.4f}  '
          f'({t["n_successful"]}/{t["n_total"]} ok)')


def _log_ret(t):
    print(f'       P@{t["k"]}={t["precision_at_k"]:.4f}  '
          f'({t["n_successful"]} queries)')


def _save_report(report, out_path):
    if out_path is None:
        out_path = os.path.join(BASE_DIR, 'evaluation_report.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f'[eval] Report saved → {out_path}', flush=True)


def _print_summary(report):
    t = report.get('tests', {})
    w = 58
    print('\n' + '='*w)
    print('  GeoSemantics Evaluation Summary')
    print('='*w)

    def _row(label, val):
        print(f'  {label:<36} {val}')

    if 'char_v2' in t:
        _row('Character accuracy V2:',
             f'{t["char_v2"]["accuracy"]:.3f}  ({t["char_v2"]["correct"]}/{t["char_v2"]["total"]})')
    if 'char_v3' in t:
        _row('Character accuracy V3:',
             f'{t["char_v3"]["accuracy"]:.3f}  ({t["char_v3"]["correct"]}/{t["char_v3"]["total"]})')
    if 'rural_alpine' in t:
        r = t['rural_alpine']
        _row('Rural/Alpine acc V2 → V3:',
             f'{r["v2_accuracy"]:.3f} → {r["v3_accuracy"]:.3f}  (Δ={r["improvement"]:+.3f})')
    if t.get('char_v2') and t.get('char_v3'):
        print()
        print('  Per-class accuracy V2 vs V3:')
        for cls in CLASS_GROUPS:
            a2 = t['char_v2']['per_class'].get(cls, {}).get('acc', '–')
            a3 = t['char_v3']['per_class'].get(cls, {}).get('acc', '–')
            tag = ' ← V3 wins' if isinstance(a2, float) and isinstance(a3, float) and a3 > a2 else ''
            print(f'    {cls:<30} V2={a2:.2f}  V3={a3:.2f}{tag}')
    print()
    if 'sep_v2' in t:
        s = t['sep_v2']
        _row('Embedding sep. V2:', f'{s["separability"]:.4f}  (intra={s["intra_sim"]:.4f}, inter={s["inter_sim"]:.4f})')
    if 'sep_v3' in t:
        s = t['sep_v3']
        _row('Embedding sep. V3:', f'{s["separability"]:.4f}  (intra={s["intra_sim"]:.4f}, inter={s["inter_sim"]:.4f})')
    if 'ret_v2' in t:
        _row(f'Retrieval P@{t["ret_v2"]["k"]} V2:', f'{t["ret_v2"]["precision_at_k"]:.4f}')
    if 'ret_v3' in t:
        _row(f'Retrieval P@{t["ret_v3"]["k"]} V3:', f'{t["ret_v3"]["precision_at_k"]:.4f}')

    if 'ablation' in t:
        print()
        print('  Ablation study (separability ↑ = component helps):')
        baseline = t['ablation'].get('v3_full', {}).get('separability', 0)
        for name, r in t['ablation'].items():
            delta = r['separability'] - baseline
            tag   = '  (baseline)' if name == 'v3_full' else f'  ({delta:+.4f})'
            print(f'    {name:<25} {r["separability"]:.4f}{tag}')

    print('='*w)
    print(f'  Elapsed: {report.get("elapsed_s", "–")} s')
    print('='*w + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeoSemantics Evaluation Suite')
    parser.add_argument('--quick',    action='store_true',
                        help='Character analysis only (no embedding model needed)')
    parser.add_argument('--ablation', action='store_true',
                        help='Include inference-time ablation study (slow, needs v3)')
    parser.add_argument('--out',      default=None, metavar='PATH',
                        help='Output JSON path (default: evaluation_report.json)')
    args = parser.parse_args()

    run_full_evaluation(
        out_path=args.out,
        run_models=not args.quick,
        run_ablation=args.ablation,
    )
