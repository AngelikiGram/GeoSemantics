"""
precompute_all_places.py
────────────────────────────────────────────────────────────────
Generates _poi_cache/all_places_umap.json containing UMAP (ux, uy)
coordinates and dominant semantic dimensions for ALL Austrian
cities / towns / villages in df.parquet.

Strategy
--------
1.  Load df.parquet → filter to place-type nodes with a name.
2.  Deduplicate (same name + similar lat/lon).
3.  For each place try GNN character inference (if inference module
    is available); otherwise fall back to a geography-based
    approximation that assigns plausible semantic_dims from the
    place_type and coordinates.
4.  Load the 53 existing precomputed locations from morph_data.json
    so the UMAP is fit JOINTLY — the 53 seed locations keep their
    relative positions, and new places land in the same space.
5.  Run UMAP (or PCA fallback) over the combined set.
6.  Save slim output — no embeddings, just name/lat/lon/ux/uy/dominant_dim.

Run once before starting morph_app.py:
    python precompute_all_places.py [--fast]

    --fast   Skip GNN inference entirely (seconds vs. minutes).

Output: _poi_cache/all_places_umap.json  (~200–600 KB)
────────────────────────────────────────────────────────────────
"""
import os, json, sys, math, argparse, time
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--fast', action='store_true',
                    help='Skip GNN; use geographic approximation for all new places')
parser.add_argument('--limit', type=int, default=0,
                    help='Cap total new places (0 = no cap, useful for testing)')
parser.add_argument('--place-types', type=str,
                    default='city,town,village,suburb,neighbourhood',
                    help='Comma-separated place types to include')
args = parser.parse_args()

FAST_MODE    = args.fast
LIMIT        = args.limit
PLACE_TYPES  = set(args.place_types.split(','))

OUT_PATH     = os.path.join('_poi_cache', 'all_places_umap.json')
MORPH_PATH   = os.path.join('_poi_cache', 'morph_data.json')
PARQUET_PATH = os.path.join('_poi_cache', 'df.parquet')
RADIUS       = 600

# Austria bounding box (loose)
LAT_MIN, LAT_MAX = 46.3, 49.1
LON_MIN, LON_MAX = 9.5,  17.2

# ── Helpers ───────────────────────────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p  = math.radians(lat1); q  = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p) * math.cos(q) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _geo_dims(lat, lon, place_type):
    """
    Fast geographic approximation of semantic_dims.
    Uses lat/lon and place_type to make a plausible character estimate.
    This is intentionally simple — it just ensures new places can land
    somewhere sensible in UMAP space without running GNN inference.
    """
    # Altitude proxy — rough heuristic from longitude + lat
    # Tyrol/Salzburg alpine zone: lon 10–14, lat 46.5–47.7
    alpine_score = max(0.0, min(1.0,
        (1 - abs(lon - 12.0) / 3.5) * 0.5 +
        (1 - abs(lat - 47.1) / 1.0) * 0.5
    ))
    urban_score = {'city': 0.75, 'town': 0.50, 'suburb': 0.45,
                   'neighbourhood': 0.40, 'village': 0.15}.get(place_type, 0.25)
    # Eastern plains (Burgenland/Weinviertel) → Nature/Community
    rural_east = max(0.0, (lon - 15.5) / 1.5) * (1 - urban_score)
    nature_score = alpine_score * 0.6 + rural_east * 0.3 + max(0, 0.4 - urban_score)

    dims = {
        'Urban':          round(urban_score, 3),
        'Nature':         round(min(1.0, nature_score), 3),
        'Tourism':        round(alpine_score * 0.4, 3),
        'Heritage':       round(urban_score * 0.2 + 0.05, 3),
        'Transport':      round(urban_score * 0.15, 3),
        'Community':      round((1 - urban_score) * 0.3 + rural_east * 0.2, 3),
        'Infrastructure': round(urban_score * 0.1, 3),
    }
    total = sum(dims.values()) + 1e-9
    dims  = {k: round(v / total, 4) for k, v in dims.items()}
    dominant = max(dims, key=dims.get)
    return dims, dominant


def _run_umap_joint(seed_embeddings, new_features):
    """
    Fit UMAP on seed_embeddings (real GNN), then transform new_features.
    Both inputs are numpy arrays of the same column dimension.
    Falls back to PCA if umap-learn is not installed.
    """
    all_feats = np.vstack([seed_embeddings, new_features])
    n_total   = len(all_feats)
    try:
        from umap import UMAP
        n_neighbors = min(15, n_total - 1)
        reducer = UMAP(n_components=2, metric='cosine',
                       random_state=42, n_neighbors=n_neighbors,
                       min_dist=0.08)
        coords  = reducer.fit_transform(all_feats)
        method  = 'UMAP'
    except ImportError:
        print('[all_places] umap-learn not found — using PCA fallback')
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(all_feats)
        method = 'PCA'

    mn, mx = coords.min(axis=0), coords.max(axis=0)
    rng    = mx - mn + 1e-8
    normed = ((coords - mn) / rng)
    return normed[:len(seed_embeddings)], normed[len(seed_embeddings):], method


def _dims_to_feature(dims):
    """Convert semantic_dims dict to a fixed-length feature vector."""
    KEYS = ['Urban', 'Tourism', 'Heritage', 'Nature',
            'Transport', 'Infrastructure', 'Community']
    return np.array([dims.get(k, 0.0) for k in KEYS], dtype=float)


# ── 1. Load existing morph_data.json (53 seeds) ──────────────────────────────
print('[all_places] Loading existing seed locations …')
with open(MORPH_PATH, 'r', encoding='utf-8') as fh:
    morph = json.load(fh)
seed_locs = morph['locations']  # already have ux/uy + semantic_dims
print(f'  → {len(seed_locs)} seed locations loaded')

# ── 2. Load df.parquet ───────────────────────────────────────────────────────
print(f'[all_places] Loading df.parquet …')
try:
    import pandas as pd
    df_all = pd.read_parquet(PARQUET_PATH)
except ImportError:
    print('[all_places] ERROR: pandas/pyarrow not installed. Run: pip install pandas pyarrow')
    sys.exit(1)

# Filter to wanted place types within Austria
mask = (
    df_all['place'].isin(PLACE_TYPES) &
    df_all['name'].notna() &
    (df_all['name'] != '') &
    df_all['lat'].between(LAT_MIN, LAT_MAX) &
    df_all['lon'].between(LON_MIN, LON_MAX)
)
df_places = df_all[mask].copy()
print(f'  → {len(df_places)} place rows after filtering')

# De-duplicate: keep one entry per (name, rounded-lat, rounded-lon)
seen    = set()
rows    = []
seed_coords = {(round(l['lat'], 3), round(l['lon'], 3)) for l in seed_locs}

for _, row in df_places.iterrows():
    name  = str(row['name']).strip()
    lat   = float(row['lat'])
    lon   = float(row['lon'])
    ptype = str(row['place'])
    key   = (name.lower(), round(lat, 3), round(lon, 3))
    if key in seen:
        continue
    seen.add(key)
    # Skip if this is essentially one of the existing 53 seeds (within 200 m)
    is_seed = any(_haversine(lat, lon, sl['lat'], sl['lon']) < 200 for sl in seed_locs)
    if is_seed:
        continue
    rows.append({'name': name, 'lat': lat, 'lon': lon, 'place_type': ptype})

if LIMIT > 0:
    rows = rows[:LIMIT]

print(f'  → {len(rows)} unique new places (after removing seed overlaps)')

# ── 3. Optional GNN inference ─────────────────────────────────────────────────
_inf = None
if not FAST_MODE:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import inference as _inf
        print('[all_places] Inference module loaded — will attempt live GNN character')
    except Exception as e:
        print(f'[all_places] Inference not available ({e}) — using fast geo approximation')
        _inf = None

# ── 4. Process new places ─────────────────────────────────────────────────────
print(f'\n[all_places] Processing {len(rows)} new places (fast={FAST_MODE}) …')
new_records = []
for i, row in enumerate(rows):
    lat, lon   = row['lat'], row['lon']
    ptype      = row['place_type']
    dims       = None

    # Try live GNN inference
    if _inf is not None and not FAST_MODE:
        try:
            char = None
            if hasattr(_inf, 'get_location_character_v3') and _inf._v3_node_types is not None:
                char = _inf.get_location_character_v3(lat, lon, RADIUS)
            if char is None and _inf.v2_available:
                char = _inf.get_location_character(lat, lon, RADIUS)
            if char and char.get('char_dims'):
                dims     = char['char_dims']
                dominant = char.get('label', max(dims, key=dims.get))
        except Exception:
            pass

    # Fallback: geographic approximation
    if dims is None:
        dims, dominant = _geo_dims(lat, lon, ptype)

    new_records.append({
        'name':         row['name'],
        'lat':          lat,
        'lon':          lon,
        'place_type':   ptype,
        'semantic_dims': dims,
        'dominant_dim': dominant,
        'source':       'geo_approx' if _inf is None or FAST_MODE else 'gnn',
    })

    if (i + 1) % 200 == 0 or (i + 1) == len(rows):
        print(f'  [{i+1:4d}/{len(rows)}] … last: {row["name"]}')

print(f'[all_places] Done processing — {len(new_records)} new records')

# ── 5. Build joint feature matrix and run UMAP ───────────────────────────────
print('\n[all_places] Building feature matrix for UMAP …')

# Seed feature vectors (from existing semantic_dims, consistent with new_records)
seed_feats = np.array([_dims_to_feature(s.get('semantic_dims', {})) for s in seed_locs])
new_feats  = np.array([_dims_to_feature(r['semantic_dims']) for r in new_records])

print(f'  seed shape: {seed_feats.shape}, new shape: {new_feats.shape}')
print('[all_places] Running UMAP jointly …')
seed_coords_new, new_coords, method = _run_umap_joint(seed_feats, new_feats)

print(f'  UMAP done ({method}). Updating coordinates …')

# Update seed ux/uy with the new joint-UMAP positions
# (the relative ordering within seeds is preserved, absolute positions may shift slightly)
for loc, (ux, uy) in zip(seed_locs, seed_coords_new):
    loc['ux'] = round(float(ux), 4)
    loc['uy'] = round(float(uy), 4)

# Assign new UMAP coords
for rec, (ux, uy) in zip(new_records, new_coords):
    rec['ux'] = round(float(ux), 4)
    rec['uy'] = round(float(uy), 4)

# ── 6. Build output: seeds + new places (slim — no embeddings) ───────────────
out_records = []

# Seeds: keep all fields but strip embedding (bandwidth)
for loc in seed_locs:
    out_records.append({
        'name':         loc['name'],
        'lat':          loc['lat'],
        'lon':          loc['lon'],
        'place_type':   'precomputed_seed',
        'dominant_dim': max(loc.get('semantic_dims', {'Community': 1}),
                            key=loc.get('semantic_dims', {'Community': 1}).get),
        'semantic_dims': loc.get('semantic_dims', {}),
        'label':        loc.get('label', ''),
        'ux':           loc['ux'],
        'uy':           loc['uy'],
        'is_seed':      True,
    })

# New places
for rec in new_records:
    out_records.append({
        'name':         rec['name'],
        'lat':          rec['lat'],
        'lon':          rec['lon'],
        'place_type':   rec['place_type'],
        'dominant_dim': rec['dominant_dim'],
        'semantic_dims': rec['semantic_dims'],
        'ux':           rec['ux'],
        'uy':           rec['uy'],
        'source':       rec['source'],
        'is_seed':      False,
    })

# ── 7. Save ───────────────────────────────────────────────────────────────────
os.makedirs('_poi_cache', exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as fh:
    json.dump({'places': out_records, 'n_places': len(out_records),
               'method': method}, fh, ensure_ascii=False, separators=(',', ':'))

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f'\n[all_places] ✓ Saved {len(out_records)} places to {OUT_PATH} ({size_kb:.0f} KB)')
print(f'  Seeds: {sum(1 for r in out_records if r["is_seed"])}')
print(f'  New  : {sum(1 for r in out_records if not r["is_seed"])}')
print(f'  Method: {method}')
