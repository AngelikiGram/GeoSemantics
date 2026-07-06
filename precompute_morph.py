"""
precompute_morph.py
────────────────────────────────────────────────────────────────
Generate morph_data.json for GeoSemantics-Morph.

Run once before starting morph_app.py:
    python precompute_morph.py

Requires  : trained saliency_gnn.pt  (always required)
            optional geosemantics_v2.pt (uses v1 fallback if absent)
Output    : _poi_cache/morph_data.json  (~400–600 KB)
Duration  : ~3–8 min depending on hardware
────────────────────────────────────────────────────────────────
"""
import os, json, math, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference as inf

RADIUS   = 600
OUT_PATH = os.path.join('_poi_cache', 'morph_data.json')

ARCHETYPES = {
    'Vienna Center':    (48.2082, 16.3738),
    'Salzburg':         (47.7987, 13.0468),
    'Innsbruck':        (47.2682, 11.3927),
    'Alpine Resort':    (47.3255, 12.7958),
    'Rural Burgenland': (47.8500, 16.6000),
    'Vienna Airport':   (48.1103, 16.5697),
}

SEED_LOCATIONS = [
    # ── Vienna ─────────────────────────────────────────────────────
    {"name": "Vienna · Innere Stadt",    "lat": 48.2082, "lon": 16.3738},
    {"name": "Vienna · Naschmarkt",      "lat": 48.1990, "lon": 16.3650},
    {"name": "Vienna · Prater",          "lat": 48.2167, "lon": 16.3944},
    {"name": "Vienna · Mariahilf",       "lat": 48.1979, "lon": 16.3555},
    {"name": "Vienna · Belvedere",       "lat": 48.1908, "lon": 16.3808},
    {"name": "Vienna · Schönbrunn",      "lat": 48.1845, "lon": 16.3122},
    {"name": "Vienna · Donaustadt",      "lat": 48.2394, "lon": 16.4470},
    {"name": "Vienna · Airport",         "lat": 48.1103, "lon": 16.5697},
    {"name": "Vienna · Floridsdorf",     "lat": 48.2551, "lon": 16.3993},
    {"name": "Vienna · Leopoldstadt",    "lat": 48.2167, "lon": 16.3834},
    {"name": "Vienna · Hietzing",        "lat": 48.1777, "lon": 16.2917},
    {"name": "Vienna · Favoriten",       "lat": 48.1639, "lon": 16.3714},
    # ── Salzburg ───────────────────────────────────────────────────
    {"name": "Salzburg · Altstadt",      "lat": 47.7987, "lon": 13.0468},
    {"name": "Salzburg · Festung",       "lat": 47.7946, "lon": 13.0501},
    {"name": "Salzburg · Neustadt",      "lat": 47.8045, "lon": 13.0437},
    {"name": "Salzburg · Hellbrunn",     "lat": 47.7622, "lon": 13.0596},
    # ── Innsbruck ──────────────────────────────────────────────────
    {"name": "Innsbruck · Altstadt",     "lat": 47.2682, "lon": 11.3927},
    {"name": "Innsbruck · Hungerburg",   "lat": 47.2832, "lon": 11.4007},
    {"name": "Innsbruck · Bahnhof",      "lat": 47.2629, "lon": 11.4002},
    # ── Graz ───────────────────────────────────────────────────────
    {"name": "Graz · Hauptplatz",        "lat": 47.0708, "lon": 15.4386},
    {"name": "Graz · Schlossberg",       "lat": 47.0770, "lon": 15.4387},
    {"name": "Graz · Liebenau",          "lat": 47.0351, "lon": 15.4477},
    # ── Linz ───────────────────────────────────────────────────────
    {"name": "Linz · Hauptplatz",        "lat": 48.3069, "lon": 14.2858},
    {"name": "Linz · Hafen",             "lat": 48.3019, "lon": 14.3097},
    {"name": "Linz · Pöstlingberg",      "lat": 48.3215, "lon": 14.2717},
    # ── Other cities ───────────────────────────────────────────────
    {"name": "Klagenfurt · Zentrum",     "lat": 46.6228, "lon": 14.3070},
    {"name": "Klagenfurt · Wörthersee",  "lat": 46.6117, "lon": 14.2866},
    {"name": "St. Pölten · Zentrum",     "lat": 48.2052, "lon": 15.6257},
    {"name": "Bregenz · Altstadt",       "lat": 47.5031, "lon":  9.7471},
    {"name": "Dornbirn · Zentrum",       "lat": 47.4129, "lon":  9.7417},
    {"name": "Villach · Zentrum",        "lat": 46.6103, "lon": 13.8558},
    {"name": "Wels · Zentrum",           "lat": 48.1570, "lon": 14.0237},
    {"name": "Steyr · Zentrum",          "lat": 48.0404, "lon": 14.4208},
    {"name": "Krems an der Donau",       "lat": 48.4097, "lon": 15.6049},
    {"name": "Wiener Neustadt",          "lat": 47.8096, "lon": 16.2439},
    {"name": "Baden bei Wien",           "lat": 48.0048, "lon": 16.2321},
    {"name": "Feldkirch",                "lat": 47.2372, "lon":  9.6009},
    {"name": "Kapfenberg",               "lat": 47.4492, "lon": 15.2925},
    {"name": "Amstetten",                "lat": 48.1217, "lon": 14.8692},
    # ── Alpine / tourist ───────────────────────────────────────────
    {"name": "Zell am See",              "lat": 47.3255, "lon": 12.7958},
    {"name": "Hallstatt",                "lat": 47.5622, "lon": 13.6493},
    {"name": "Seefeld in Tirol",         "lat": 47.3324, "lon": 11.1855},
    {"name": "Kitzbühel",                "lat": 47.4463, "lon": 12.3939},
    {"name": "Bad Gastein",              "lat": 47.1110, "lon": 13.1329},
    {"name": "Alpbach",                  "lat": 47.3866, "lon": 11.9680},
    {"name": "Mariazell",                "lat": 47.7731, "lon": 15.3164},
    {"name": "Lech am Arlberg",          "lat": 47.2083, "lon": 10.1390},
    {"name": "Mayrhofen",                "lat": 47.1669, "lon": 11.8666},
    {"name": "Großglockner",             "lat": 47.0742, "lon": 12.6939},
    # ── Rural / Burgenland ─────────────────────────────────────────
    {"name": "Neusiedl am See",          "lat": 47.9481, "lon": 16.8414},
    {"name": "Eisenstadt",               "lat": 47.8451, "lon": 16.5270},
    {"name": "Rust",                     "lat": 47.8025, "lon": 16.6734},
    {"name": "Frauenkirchen",            "lat": 47.8378, "lon": 16.9218},
    # ── Remote / nature ────────────────────────────────────────────
    {"name": "Waldviertel",              "lat": 48.5821, "lon": 15.3372},
    {"name": "Weinviertel",              "lat": 48.5477, "lon": 16.5673},
    {"name": "Salzkammergut",            "lat": 47.6500, "lon": 13.5000},
]

SCALE_LABELS = ['Micro (200 m)', 'Meso (700 m)', 'Macro (2 km)']


def _confidence(n_pois, avg_sal):
    return round(min(1.0, n_pois / 200) * 0.45 + float(avg_sal) * 0.55, 3)


def _compute_arch_embeddings():
    print('[precompute] Computing archetype embeddings …')
    result = {'v1': {}, 'v2': {}}
    for name, (lat, lon) in ARCHETYPES.items():
        v1, _ = inf.get_embedding(lat, lon, RADIUS)
        result['v1'][name] = v1

        if inf.v2_available:
            v2, _ = inf.get_embedding_v2(lat, lon)
            result['v2'][name] = v2

        print(f'  ✓ {name}')
    return result


def _process_location(loc, arch_embs):
    lat, lon = loc['lat'], loc['lon']

    sal, sel_df, _ = inf.get_saliency(lat, lon, RADIUS)
    if sal is None or len(sal) < 3:
        return None

    char = inf.get_location_character(lat, lon, RADIUS)

    s_min   = sal.min()
    s_range = (sal.max() - s_min) + 1e-8
    sal_norm = (sal - s_min) / s_range
    avg_sal  = float(sal_norm.mean())

    # Top POIs
    pois = []
    for i, (_, row) in enumerate(sel_df.iterrows()):
        cat, val = inf.get_poi_label(row)
        if cat == 'unknown':
            continue
        pois.append({
            'name':     str(row.get('name', '') or val),
            'category': cat,
            'value':    val,
            'saliency': round(float(sal_norm[i]) if i < len(sal_norm) else 0.0, 4),
            'lat':      float(row['lat']),
            'lon':      float(row['lon']),
        })
    pois.sort(key=lambda p: p['saliency'], reverse=True)

    # Embedding (v2 preferred, v1 fallback)
    embedding    = None
    scale_attn   = [1/3, 1/3, 1/3]
    used_v2      = False
    if inf.v2_available:
        emb_v2, attn = inf.get_embedding_v2(lat, lon)
        if emb_v2 is not None:
            embedding  = emb_v2.tolist()
            scale_attn = attn.tolist() if attn is not None else scale_attn
            used_v2 = True
    if embedding is None:
        emb_v1, _ = inf.get_embedding(lat, lon, RADIUS)
        if emb_v1 is not None:
            embedding = emb_v1.tolist()

    # Archetype similarities
    arch_sims = {}
    if embedding and arch_embs:
        e = np.array(embedding)
        arches = arch_embs.get('v2', {}) if used_v2 else arch_embs.get('v1', {})
        for name, ae in arches.items():
            if ae is not None:
                arch_sims[name] = round(float(inf.cosine_sim(e, ae)), 4)

    n_pois = char.get('n_pois', len(sal))

    return {
        'name':                   loc['name'],
        'lat':                    lat,
        'lon':                    lon,
        'label':                  char.get('label', '–'),
        'embedding':              embedding,
        'ux':                     0.5,   # filled after UMAP
        'uy':                     0.5,
        'semantic_dims':          char.get('char_dims', {}),
        'scale_attention':        [round(x, 4) for x in scale_attn],
        'archetype_similarities': arch_sims,
        'top_pois':               pois,
        'n_pois':                 n_pois,
        'avg_saliency':           round(avg_sal, 4),
        'confidence':             _confidence(n_pois, avg_sal),
    }


def _run_umap(embeddings):
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, metric='cosine', random_state=42,
                       n_neighbors=min(15, len(embeddings) - 1), min_dist=0.1)
        coords = reducer.fit_transform(np.array(embeddings))
    except ImportError:
        print('[precompute] umap-learn not found — using PCA fallback')
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(np.array(embeddings))

    mn, mx = coords.min(axis=0), coords.max(axis=0)
    rng = mx - mn + 1e-8
    return ((coords - mn) / rng).tolist()


def main():
    os.makedirs('_poi_cache', exist_ok=True)

    arch_embs = _compute_arch_embeddings()

    print(f'\n[precompute] Processing {len(SEED_LOCATIONS)} locations …')
    locations = []
    for i, loc in enumerate(SEED_LOCATIONS):
        t0 = time.time()
        result = _process_location(loc, arch_embs)
        elapsed = time.time() - t0
        tag = f'[{i+1:02d}/{len(SEED_LOCATIONS)}]'
        if result is None:
            print(f'  {tag} SKIP {loc["name"]} (no POIs)')
            continue
        print(f'  {tag} {loc["name"]:35s}  '
              f'pois={result["n_pois"]:3d}  sal={result["avg_saliency"]:.3f}  '
              f'label={result["label"]:<26s}  {elapsed:.1f}s')
        locations.append(result)

    # UMAP over all embeddings that exist
    locs_with_emb = [l for l in locations if l['embedding']]
    if locs_with_emb:
        # Avoid mixed embedding dimensions in UMAP
        max_dim = max(len(l['embedding']) for l in locs_with_emb)
        locs_with_emb = [l for l in locs_with_emb if len(l['embedding']) == max_dim]

    if len(locs_with_emb) >= 4:
        print(f'\n[precompute] Running UMAP on {len(locs_with_emb)} embeddings (dim={max_dim}) …')
        coords = _run_umap([l['embedding'] for l in locs_with_emb])
        for l, (ux, uy) in zip(locs_with_emb, coords):
            l['ux'] = round(float(ux), 4)
            l['uy'] = round(float(uy), 4)

    for i, l in enumerate(locations):
        l['id'] = i

    out = {'locations': locations, 'n_locations': len(locations)}
    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(',', ':'))

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f'\n[precompute] Done — {len(locations)} locations saved to {OUT_PATH} ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()
