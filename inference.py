"""
GeoSemantics inference module.

First run: builds _poi_cache/ (~60 s).
Subsequent runs: loads from cache (~5 s).
"""
import json
import math
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '_poi_cache')

K_NEIGHBORS   = 8
IN_CHANNELS   = 256   # 2 coords + 222 cat OHE + 32 other-tags
HIDDEN        = 64
EMBED_DIM     = 32
N_OHE_FEATURES = IN_CHANNELS - 2 - 32   # 222 – must match trained model

# Regex to extract hive-format pairs  "key"=>"value"
_KEY_RE = re.compile(r'"([^"]+)"=>"')          # keys only (for CountVectorizer)
_KV_RE  = re.compile(r'"([^"]+)"=>"([^"]*)"')  # key+value (for labels)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _other_tags_keys(arr):
    """Extract space-joined key names from an array of other_tags strings."""
    return [' '.join(_KEY_RE.findall(t)) if t else '' for t in arr]


# Keys in other_tags that carry no semantic meaning for place characterisation.
# Note: 'emergency' is intentionally NOT here — emergency=fire_hydrant IS semantic.
_LABEL_BLOCKLIST = frozenset({
    'check_date', 'source', 'note', 'survey:date', 'created_by',
    'fixme', 'FIXME', 'description', 'start_date', 'end_date',
    'image', 'url', 'website', 'phone', 'opening_hours', 'email',
    'ref', 'old_name', 'alt_name', 'operator', 'brand',
    'wikipedia', 'wikidata', 'brand:wikidata', 'inscription',
    'attribution', 'comment', 'last_edit_user_id',
    'colour', 'couplings', 'fire_hydrant:type', 'fire_hydrant:diameter',
    'fire_hydrant:position', 'fire_hydrant:pressure', 'pillar:type',
    'water_source', 'bonnet:colour', 'cap:colour', 'manufacturer',
    'noexit', 'entrance', 'access', 'circumference', 'height', 'width', 'length', 'depth', 'distance', 'diameter', 'ele', 'elevation', 'maxspeed', 'capacity', 'lanes', 'levels',   # these appear as keys with value "yes"; not useful alone
    'layer', 'oneway', 'lit', 'surface', 'smoothness', 'tracktype', 'bicycle', 'foot', 'horse',  # routing/geometry metadata, not place character
})

# highway tag values that are routing/geometry artifacts, not place types
_HIGHWAY_SKIP = frozenset({
    'noexit', 'give_way', 'traffic_signals', 'turning_circle',
    'mini_roundabout', 'stop', 'motorway_junction', 'passing_place',
    'speed_camera', 'milestone', 'elevator', 'street_lamp',
    'traffic_calming', 'turning_loop', 'emergency_bay',
})

# Values that mean "yes" or "no" and carry no extra info beyond the key name
_YES_VALUES = frozenset({'yes', 'y', '1', 'true'})
_NO_VALUES  = frozenset({'no', 'n', '0', 'false'})
_DATE_RE    = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}$|^\d{4}$')


def _fmt(key: str, val: str) -> str:
    """Return a human-readable label for a tag value.
    'yes' → key name;  'fire_hydrant' → 'fire hydrant'; etc.
    """
    val = val.strip()
    if val.lower() in _YES_VALUES:
        return key.replace('_', ' ')
    if val.lower() in _NO_VALUES:
        return 'no ' + key.replace('_', ' ')
    return val.replace('_', ' ')


def get_poi_label(row):
    """Return (category, value) for display.  Works on pd.Series or dict."""
    get = row.get if hasattr(row, 'get') else lambda k, d='': row[k]
    # Dedicated columns first; clean semicolons and normalise values
    for col in ('highway', 'place', 'man_made', 'natural'):
        v = get(col, '')
        if v:
            v = v.split(';')[0].strip()
            if v and (col != 'highway' or v not in _HIGHWAY_SKIP):
                return col, _fmt(col, v)
    tags_str = get('other_tags', '') or ''
    tags = {m.group(1): m.group(2) for m in _KV_RE.finditer(tags_str)}
    # High-value semantic keys — ordered by interpretive richness
    for key in ('amenity', 'tourism', 'shop', 'leisure', 'historic',
                'sport', 'emergency', 'healthcare', 'office',
                'barrier', 'waterway', 'building', 'power',
                'public_transport', 'railway', 'landuse', 'aeroway'):
        if key in tags and tags[key]:
            v = _fmt(key, tags[key])
            if v:
                return key, v
    # Fall back to first non-noise, non-metadata tag
    for k, v in tags.items():
        if (k not in _LABEL_BLOCKLIST
                and not k.startswith(('seamark:', 'source:', 'addr:', 'contact:', 'name:'))
                and v and not _DATE_RE.match(v)):
            return k, _fmt(k, v)
    return 'unknown', 'point'


# ---------------------------------------------------------------------------
# Cache build / load
# ---------------------------------------------------------------------------

def _build_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print('[cache] Parsing GeoJSON … (one-time, ~60 s)', flush=True)
    with open(os.path.join(BASE_DIR, 'austrian-pois.geojson'), 'r', encoding='utf-8') as f:
        geo = json.load(f)

    lons, lats = [], []
    names, highways, naturals, places, man_mades, other_tags_list = [], [], [], [], [], []
    for feat in geo['features']:
        lo, la = feat['geometry']['coordinates']
        p = feat['properties']
        lons.append(lo);         lats.append(la)
        names.append(p.get('name')      or '')
        highways.append(p.get('highway')  or '')
        naturals.append(p.get('natural')  or '')
        places.append(p.get('place')    or '')
        man_mades.append(p.get('man_made') or '')
        other_tags_list.append(p.get('other_tags') or '')
    del geo

    df = pd.DataFrame({
        'lon': np.array(lons, dtype=np.float64),
        'lat': np.array(lats, dtype=np.float64),
        'name':       names,
        'highway':    highways,
        'natural':    naturals,
        'place':      places,
        'man_made':   man_mades,
        'other_tags': other_tags_list,
    })
    print(f'[cache] {len(df):,} POIs loaded.', flush=True)
    df.to_parquet(os.path.join(CACHE_DIR, 'df.parquet'), index=False)

    # Category label encoding (vectorised, no row-wise Python loops)
    h = np.asarray(highways);  n = np.asarray(naturals)
    p_arr = np.asarray(places); m = np.asarray(man_mades)
    cat_primary = np.where(h != '', h,
                  np.where(n != '', n,
                  np.where(p_arr != '', p_arr,
                  np.where(m != '', m, ''))))
    del h, n, p_arr, m, lons, lats, names, highways, naturals, places, man_mades

    le = LabelEncoder()
    cat_labels = le.fit_transform(cat_primary).astype(np.int32)
    assert len(le.classes_) == N_OHE_FEATURES, (
        f"Expected {N_OHE_FEATURES} categories, got {len(le.classes_)}. "
        "Check that IN_CHANNELS matches the trained model.")
    np.save(os.path.join(CACHE_DIR, 'cat_labels.npy'), cat_labels)
    with open(os.path.join(CACHE_DIR, 'le.pkl'), 'wb') as fh:
        pickle.dump(le, fh)
    print(f'[cache] {len(le.classes_)} category classes encoded.', flush=True)

    # CountVectorizer – fit on extracted keys (C-level regex, fast)
    print('[cache] Fitting CountVectorizer …', flush=True)
    keys_list = _other_tags_keys(other_tags_list)
    del other_tags_list
    vect = CountVectorizer(max_features=32)
    vect.fit(keys_list)
    del keys_list
    with open(os.path.join(CACHE_DIR, 'vectorizer.pkl'), 'wb') as fh:
        pickle.dump(vect, fh)
    print(f'[cache] Vectorizer done. Vocab: {list(vect.vocabulary_)[:8]} …', flush=True)

    # Spatial index
    print('[cache] Building BallTree …', flush=True)
    coords_rad = np.radians(df[['lat', 'lon']].values)
    tree = BallTree(coords_rad, metric='haversine')
    with open(os.path.join(CACHE_DIR, 'btree.pkl'), 'wb') as fh:
        pickle.dump(tree, fh)
    print('[cache] Cache built and saved.', flush=True)

    return df, le, cat_labels, vect, tree


def _load_cache():
    print('[cache] Loading from cache …', flush=True)
    df = pd.read_parquet(os.path.join(CACHE_DIR, 'df.parquet'))
    cat_labels = np.load(os.path.join(CACHE_DIR, 'cat_labels.npy'))
    with open(os.path.join(CACHE_DIR, 'le.pkl'), 'rb') as fh:
        le = pickle.load(fh)
    with open(os.path.join(CACHE_DIR, 'vectorizer.pkl'), 'rb') as fh:
        vect = pickle.load(fh)
    with open(os.path.join(CACHE_DIR, 'btree.pkl'), 'rb') as fh:
        tree = pickle.load(fh)
    print(f'[cache] Loaded {len(df):,} POIs from cache.', flush=True)
    return df, le, cat_labels, vect, tree


_CACHE_FILES = ['df.parquet', 'cat_labels.npy', 'le.pkl', 'vectorizer.pkl', 'btree.pkl']
_cache_ok    = all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in _CACHE_FILES)

if _cache_ok:
    df, _le, _cat_labels, _vectorizer, btree = _load_cache()
else:
    print('[cache] No cache found – building from scratch (one-time ~60 s) …', flush=True)
    df, _le, _cat_labels, _vectorizer, btree = _build_cache()

_n_categories = len(_le.classes_)   # 222
print(f'[inference] Ready. n_cat={_n_categories}', flush=True)

# Public aliases – lets geosemantics_v2.py reuse already-loaded data
le               = _le
cat_labels       = _cat_labels
vectorizer       = _vectorizer
n_poi_categories = _n_categories
_v3_type_les = None

_v3_gnn_clf = None
_v2_gnn_clf = None

def _try_load_gnn_clf():
    """Load (or reload) the GNN-supervised character classifiers.

    Pickled as {'clf': Pipeline, 'le': LabelEncoder} by train_classifier.py.
    Stored in the module-level _v2_gnn_clf / _v3_gnn_clf variables as the
    same bundle dict so call sites can do clf['clf'].predict_proba(feat) and
    clf['le'].classes_ to map integer columns back to string class names.
    Falls back to the heuristic _CHAR_MAP path if loading fails.
    """
    global _v3_gnn_clf, _v2_gnn_clf

    for key, path_stem in [('v3', 'gnn_clf_v3'), ('v2', 'gnn_clf_v2')]:
        current = _v3_gnn_clf if key == 'v3' else _v2_gnn_clf
        if current is not None:
            continue
        clf_path = os.path.join(BASE_DIR, '_poi_cache', f'{path_stem}.pkl')
        if not os.path.exists(clf_path):
            continue
        try:
            import pickle
            with open(clf_path, 'rb') as fh:
                bundle = pickle.load(fh)
            # Support both the old bare-clf format and the new {'clf':..,'le':..} bundle
            if isinstance(bundle, dict) and 'clf' in bundle and 'le' in bundle:
                if key == 'v3':
                    _v3_gnn_clf = bundle
                else:
                    _v2_gnn_clf = bundle
            else:
                # Legacy bare Pipeline — synthesise a fake classes_ list from
                # the model's own output nodes so call sites work uniformly
                fake_bundle = {'clf': bundle, 'le': None}
                fake_bundle['clf'].classes_ = getattr(bundle, 'classes_', None)
                if key == 'v3':
                    _v3_gnn_clf = fake_bundle
                else:
                    _v2_gnn_clf = fake_bundle
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Saliency Maps & Helper Loaders
# ---------------------------------------------------------------------------

_V3_CACHE_FILES = [
    'v3_node_types.npy', 'v3_type_labels.npy',
    'v3_type_les.pkl',   'v3_type_vocabs.json',
]
_v3_cache_ok = all(os.path.exists(os.path.join(CACHE_DIR, f))
                   for f in _V3_CACHE_FILES)

_v3_node_types  = None
_v3_type_labels = None
_v3_type_les    = None
_v3_type_vocabs = None

if _v3_cache_ok:
    try:
        _v3_node_types  = np.load(os.path.join(CACHE_DIR, 'v3_node_types.npy'))
        _v3_type_labels = np.load(os.path.join(CACHE_DIR, 'v3_type_labels.npy'))
        with open(os.path.join(CACHE_DIR, 'v3_type_les.pkl'), 'rb') as fh:
            _v3_type_les = pickle.load(fh)
        with open(os.path.join(CACHE_DIR, 'v3_type_vocabs.json'), 'r', encoding='utf-8') as fh:
            _v3_type_vocabs = json.load(fh)
        print('[inference] v3 node-type cache loaded.', flush=True)
    except Exception as _e:
        print(f'[inference] v3 cache load failed: {_e}', flush=True)
else:
    print('[inference] v3 cache not found – run: python geosemantics_v3.py', flush=True)


# ---------------------------------------------------------------------------
# Model definitions  (must match trained weights exactly)
# ---------------------------------------------------------------------------

class _EmbedGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(IN_CHANNELS, HIDDEN)
        self.conv2 = GCNConv(HIDDEN, EMBED_DIM)

    def forward(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)


class _SaliencyGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(IN_CHANNELS, HIDDEN)
        self.conv2 = GCNConv(HIDDEN, 1)

    def forward(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index).squeeze(-1)


# v1 embedding model (optional – file may have been deleted)
_embed_model = None
_v1_embed_path = os.path.join(BASE_DIR, 'spatial_semantic_gnn.pt')
if os.path.exists(_v1_embed_path):
    try:
        _embed_model = _EmbedGNN()
        _embed_model.load_state_dict(
            torch.load(_v1_embed_path, map_location='cpu'))
        _embed_model.eval()
        print('[inference] v1 embedding model loaded.', flush=True)
    except Exception as _e:
        print(f'[inference] v1 embedding model skipped: {_e}', flush=True)
        _embed_model = None
else:
    print('[inference] v1 embedding model not found (spatial_semantic_gnn.pt) – skipping.', flush=True)

# Saliency model (required for heatmap and per-click saliency scores)
_saliency_model = _SaliencyGNN()
_saliency_model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, 'saliency', 'saliency_gnn.pt'),
               map_location='cpu'))
_saliency_model.eval()

print('[inference] Models loaded.', flush=True)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _hav(lon1, lat1, lon2, lat2):
    R = 6_371_000.0
    lo1, la1, lo2, la2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return R * 2 * math.asin(math.sqrt(max(a, 0)))


def _brng(lon1, lat1, lon2, lat2):
    lo1, la1, lo2, la2 = map(math.radians, [lon1, lat1, lon2, lat2])
    x = math.sin(lo2-lo1) * math.cos(la2)
    y = math.cos(la1)*math.sin(la2) - math.sin(la1)*math.cos(la2)*math.cos(lo2-lo1)
    return math.atan2(x, y)


def build_local_graph(query_lat, query_lon, radius=500):
    """Return (Data, sel_df) or (None, None) if < 3 POIs nearby (after expansion)."""
    q   = np.radians([[query_lat, query_lon]])
    
    # Dynamically expand radius in rural areas to find context
    current_radius = radius
    idx = btree.query_radius(q, r=current_radius / 6_371_000.0)[0]
    while len(idx) < 5 and current_radius < 5000:
        current_radius += 500
        idx = btree.query_radius(q, r=current_radius / 6_371_000.0)[0]

    if len(idx) < 3:
        return None, None

    sel_df  = df.iloc[idx].reset_index(drop=True)
    coords  = sel_df[['lat', 'lon']].values

    # --- Category features (one-hot, on-the-fly) ---
    sel_labels = _cat_labels[idx]
    sel_cat    = np.zeros((len(idx), _n_categories), dtype=np.float32)
    sel_cat[np.arange(len(idx)), sel_labels] = 1.0

    # --- Other-tags features (vectorise only this small subset) ---
    sub_keys  = _other_tags_keys(sel_df['other_tags'].tolist())
    sel_other = _vectorizer.transform(sub_keys).toarray().astype(np.float32)

    # --- Coordinate features (normalised) ---
    coords_rel = StandardScaler().fit_transform(
        coords - np.array([query_lat, query_lon]))

    node_feats = np.hstack([coords_rel, sel_cat, sel_other]).astype(np.float32).copy()  # (N, 256)

    # --- KNN edges ---
    k = min(K_NEIGHBORS, len(sel_df) - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbr.kneighbors(coords)

    edges, eattr = [], []
    for i, neighs in enumerate(indices):
        for n in neighs[1:]:
            d = _hav(coords[i, 1], coords[i, 0], coords[n, 1], coords[n, 0])
            b = _brng(coords[i, 1], coords[i, 0], coords[n, 1], coords[n, 0])
            same = int(sel_labels[i] == sel_labels[n])
            edges.append([i, n])
            eattr.append([d, math.sin(b), math.cos(b), same])

    if not edges:
        return None, None

    g = Data(
        x          = torch.tensor(node_feats),
        edge_index = torch.tensor(np.array(edges), dtype=torch.long).t().contiguous(),
        edge_attr  = torch.tensor(np.array(eattr, dtype=np.float32)),
    )
    return g, sel_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embedding(query_lat, query_lon, radius=500):
    """Return (32-d ndarray, sel_df) or (None, None). Returns None if v1 model missing."""
    if _embed_model is None:
        return None, None
    g, sel_df = build_local_graph(query_lat, query_lon, radius)
    if g is None:
        return None, None
    with torch.no_grad():
        emb = _embed_model(g.x, g.edge_index).mean(dim=0).numpy()
    return emb, sel_df


def get_saliency(query_lat, query_lon, radius=500):
    """Return (per-node ndarray, sel_df, graph) or (None, None, None)."""
    g, sel_df = build_local_graph(query_lat, query_lon, radius)
    if g is None:
        return None, None, None
    with torch.no_grad():
        sal = _saliency_model(g.x, g.edge_index).numpy()
    return sal, sel_df, g


def cosine_sim(a, b):
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# GeoSemantics v2 (optional – loaded only if geosemantics_v2.pt exists)
# ---------------------------------------------------------------------------

_v2_model     = None
v2_available  = False


import threading as _threading
_v2_load_lock = _threading.Lock()


def _try_load_v2():
    global _v2_model, v2_available
    v2_path = os.path.join(BASE_DIR, 'geosemantics_v2.pt')
    if not os.path.exists(v2_path):
        print('[inference] geosemantics_v2.pt not found – run: python geosemantics_v2.py', flush=True)
        return
    try:
        from geosemantics_v2 import GeoSemanticsV2
        model = GeoSemanticsV2(n_poi_types=_n_categories)
        state = torch.load(v2_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _v2_model    = model
        v2_available = True
        print('[inference] GeoSemantics v2 loaded.', flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f'[inference] v2 load FAILED: {exc}', flush=True)


def ensure_v2_loaded():
    """Idempotent hot-reload: re-attempts loading if the checkpoint has appeared since startup."""
    if v2_available:
        return True
    with _v2_load_lock:
        if not v2_available:
            _try_load_v2()
    return v2_available


_try_load_v2()

# Shared resource tuple for v2 graph builder (avoids loading data twice)
_v2_resources = (df, _cat_labels, _le, _vectorizer, btree)


def get_embedding_v2(query_lat, query_lon):
    """Return (64-d ndarray, (n_scales,) attn_weights) or (None, None)."""
    if _v2_model is None:
        return None, None
    from geosemantics_v2 import build_multiscale_graphs
    graphs = build_multiscale_graphs(query_lat, query_lon,
                                     external_resources=_v2_resources)
    if graphs is None:
        return None, None
    with torch.no_grad():
        emb, attn = _v2_model(graphs)
    return emb.numpy(), attn.numpy()


# ---------------------------------------------------------------------------
# POI Distribution Analysis  (always meaningful, no ML required)
# ---------------------------------------------------------------------------

# Map fine-grained POI categories → interpretable character dimensions
_CHAR_MAP = {
    'amenity':          'Urban',
    'shop':             'Urban',
    'building':         'Infrastructure',  # generic structures, not distinctive urban signal
    'landuse':          'Urban',
    'office':           'Urban',
    'tourism':          'Tourism',
    'historic':         'Heritage',
    'natural':          'Nature',
    'leisure':          'Community',       # parks, playgrounds, stadiums → community
    'waterway':         'Nature',
    'highway':          'Transport',
    'railway':          'Transport',
    'public_transport': 'Transport',
    'aeroway':          'Transport',
    'man_made':         'Infrastructure',
    'power':            'Infrastructure',
    'barrier':          'Infrastructure',
    'emergency':        'Infrastructure',
    'healthcare':       'Community',       # hospitals, pharmacies → community service
    'place':            'Community',
    'sport':            'Community',       # sports facilities → community activity
    'craft':            'Urban',           # artisan workshops, trades
    'club':             'Community',       # social and sports clubs
    'gambling':         'Urban',
    'vending':          'Urban',
    # 'unknown' is excluded from analysis — metadata-only POIs don't count
}

_CHAR_COLORS = {
    'Urban':          '#f59e0b',
    'Tourism':        '#ec4899',
    'Heritage':       '#8b5cf6',
    'Nature':         '#10b981',
    'Transport':      '#3b82f6',
    'Infrastructure': '#64748b',
    'Community':      '#94a3b8',
}

def _apply_contextual_suppression(char_raw, urban_indicator_count):
    """
    Suppresses 'Nature' dimension in heavily built environments to prevent
    urban greenery (e.g., street trees, parks) from dominating the character.
    """
    if urban_indicator_count >= 10:
        # If there are >10 urban POIs/buildings, it's definitely a built environment.
        # Suppress Nature heavily.
        if char_raw.get('Nature', 0) > 0:
            char_raw['Nature'] *= 0.05
    elif urban_indicator_count >= 3:
        # Moderate built environment (e.g. small village).
        if char_raw.get('Nature', 0) > 0:
            char_raw['Nature'] *= 0.2
    return char_raw



_CHAR_KEYS = list(_CHAR_COLORS.keys())   # fixed order for feature vectors


def _saliency_dims_vector(lat, lon, radius=500):
    """Return a 7-d L1-normalised saliency character vector (no clf path,
    no recursion risk) for use as a second-channel feature alongside the GNN
    embedding when calling the supervised classifier head."""
    sal, sel_df, _ = get_saliency(lat, lon, radius)
    if sel_df is None or len(sel_df) == 0:
        return np.zeros(len(_CHAR_KEYS), dtype=np.float32)
    if sal is not None and len(sal) == len(sel_df):
        s_min, s_range = sal.min(), (sal.max() - sal.min()) + 1e-8
        sal_norm = (sal - s_min) / s_range
    else:
        sal_norm = np.ones(len(sel_df))
    raw = {k: 0.0 for k in _CHAR_COLORS}
    for i, (_, row) in enumerate(sel_df.iterrows()):
        cat, _ = get_poi_label(row)
        if cat == 'unknown':
            continue
        raw[_CHAR_MAP.get(cat, 'Community')] += float(sal_norm[i]) * 0.7 + 0.3
    total = sum(raw.values()) + 1e-8
    return np.array([raw[k] / total for k in _CHAR_KEYS], dtype=np.float32)


def get_location_character(query_lat, query_lon, radius=500):
    """
    Analyse POI composition around a point using saliency-weighted counts.
    High-saliency (distinctive) POIs contribute more to the character profile.
    Skips metadata-only POIs so they don't bloat Community.
    Returns category counts, normalised character dimensions, a plain-English
    location label, and a character-dimension vector for comparison.
    """
    global _v2_gnn_clf

    _try_load_gnn_clf()
    if _v2_gnn_clf is not None:
        emb_res = get_embedding_v2(query_lat, query_lon)
        if emb_res is not None and emb_res[0] is not None:
            emb = emb_res[0]
            # Build 71-d combined feature: GNN embedding (64) + saliency (7)
            sal_vec = _saliency_dims_vector(query_lat, query_lon, radius)
            feat = np.concatenate([emb, sal_vec]).astype(np.float64).reshape(1, -1)
            bundle = _v2_gnn_clf
            probs = bundle['clf'].predict_proba(feat)[0]
            classes = bundle['le'].classes_ if bundle['le'] is not None else bundle['clf'].classes_

            char_norm = {k: 0.0 for k in _CHAR_COLORS}
            char_dims = {k: 0.0 for k in _CHAR_COLORS}

            for i, cls in enumerate(classes):
                char_norm[cls] = round(float(probs[i]), 3)
                char_dims[cls] = round(float(probs[i]), 3)

            return {
                'char_dims': char_dims,
                'char_norm': char_norm
            }

    # Fallback to heuristic
    sal, sel_df, _ = get_saliency(query_lat, query_lon, radius)
    if sel_df is None or len(sel_df) == 0:
        return None

    # Normalise saliency for weighting (blend 70% saliency + 30% uniform)
    if sal is not None and len(sal) == len(sel_df):
        s_min   = sal.min()
        s_range = (sal.max() - s_min) + 1e-8
        sal_norm = (sal - s_min) / s_range
    else:
        sal_norm = np.ones(len(sel_df))

    cat_counts: dict = {}
    char_raw: dict   = {k: 0.0 for k in _CHAR_COLORS}

    for i, (_, row) in enumerate(sel_df.iterrows()):
        cat, _ = get_poi_label(row)
        if cat == 'unknown':
            continue
        weight = float(sal_norm[i]) * 0.7 + 0.3  # 70% saliency + 30% floor
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        dim = _CHAR_MAP.get(cat, 'Community')
        char_raw[dim] += weight

    if not cat_counts:
        return None

    urban_indicator_count = sum(cat_counts.get(k, 0) for k in ['amenity', 'shop', 'office', 'leisure', 'healthcare', 'craft', 'tourism'])
    char_raw = _apply_contextual_suppression(char_raw, urban_indicator_count)

    total = sum(char_raw.values()) + 1e-8
    char_norm = {k: round(v / total, 3) for k, v in char_raw.items()}

    return {
        'cat_counts':  cat_counts,
        'char_dims':   char_norm,
        'char_colors': _CHAR_COLORS,
        'label':       _location_label(char_norm, len(cat_counts)),
        'n_pois':      len(sel_df),
        'n_labelled':  sum(cat_counts.values()),
    }


def _location_label(d, n_labelled):
    if n_labelled < 3:
        return 'Remote Area'
    u   = d.get('Urban', 0)
    t   = d.get('Tourism', 0)
    h   = d.get('Heritage', 0)
    nat = d.get('Nature', 0)
    tr  = d.get('Transport', 0)
    inf = d.get('Infrastructure', 0)
    com = d.get('Community', 0)
    # Combined signals — most specific first
    if h > 0.10 and t > 0.10:
        return 'Historic / Tourist'
    if t > 0.20:
        return 'Tourist Destination'
    if h > 0.18:
        return 'Heritage Site'
    if nat > 0.40 and t > 0.08:
        return 'Nature Tourism'
    if nat > 0.45:
        return 'Natural / Alpine Area'
    if tr > 0.38:
        return 'Transport Hub'
    if u > 0.50:
        return 'Urban Transport Hub' if tr > 0.15 else 'Urban Center'
    if inf > 0.35:
        return 'Industrial / Infrastructure'
    if nat > 0.25:
        return 'Natural Fringe'
    if com > 0.35:
        return 'Village / Community'
    if u > 0.30:
        return 'Mixed Urban'
    return 'Peri-Urban'


def character_dimension_similarity(dims1, dims2):
    """
    Euclidean similarity on the 7 character dimensions (normalised vectors).
    Far more discriminative than JS on raw POI categories because highway
    noise has already been collapsed into Transport, etc.
    Returns [0, 1] where 1 = identical character profile.

    Contrastive penalty: if one place is Nature-dominant and the other is
    Urban-dominant (or vice-versa), apply an additional penalty so that
    alpine/rural areas are never ranked similar to city centres.
    """
    keys = list(_CHAR_COLORS.keys())
    d1 = np.array([dims1.get(k, 0.0) for k in keys], dtype=float)
    d2 = np.array([dims2.get(k, 0.0) for k in keys], dtype=float)
    # Max Euclidean distance between two unit-sum vectors ≈ sqrt(2)
    dist = float(np.linalg.norm(d1 - d2)) / 1.414
    sim = max(0.0, 1.0 - dist)

    # Contrastive penalty for semantically incompatible dominant dimensions.
    # Each pair (A, B): positive when one place is A-dominant and the other is B-dominant.
    _OPPOSING = [
        ('Nature',    'Urban'),
        ('Nature',    'Transport'),
        ('Nature',    'Infrastructure'),
        ('Heritage',  'Transport'),
        ('Heritage',  'Infrastructure'),
        ('Tourism',   'Infrastructure'),
    ]
    total_penalty = 0.0
    for a, b in _OPPOSING:
        a1, b1 = dims1.get(a, 0.0), dims1.get(b, 0.0)
        a2, b2 = dims2.get(a, 0.0), dims2.get(b, 0.0)
        contrast = (a1 - b1) * (b2 - a2)   # positive when they are semantic opposites
        if contrast > 0.05:
            total_penalty += min(0.40, contrast * 0.65)
    if total_penalty > 0:
        sim = max(0.0, sim - min(total_penalty, 0.55))

    return round(sim, 4)


def poi_distribution_similarity(counts1, counts2):
    """
    Jensen-Shannon similarity on raw POI category counts (kept for reference).
    Prefer character_dimension_similarity for UI display — it is more discriminative.
    """
    from scipy.spatial.distance import jensenshannon
    all_cats = sorted(set(list(counts1.keys()) + list(counts2.keys())))
    p = np.array([counts1.get(c, 0) for c in all_cats], dtype=float)
    q = np.array([counts2.get(c, 0) for c in all_cats], dtype=float)
    p /= (p.sum() + 1e-8)
    q /= (q.sum() + 1e-8)
    js_dist = float(jensenshannon(p, q, base=2))
    return round(1.0 - js_dist, 4)


def get_saliency_profile(query_lat, query_lon, radius=500):
    """
    Per-category saliency profile for model validation / interpretability.

    Returns a list of {category, char_dim, avg_saliency, count} records
    sorted by avg_saliency descending.  If the saliency model is working
    correctly, semantically rich categories (tourism, historic, amenity)
    should score higher than infrastructure noise (highway, power).
    """
    sal, sel_df, _ = get_saliency(query_lat, query_lon, radius)
    if sal is None:
        return []

    s_min   = sal.min()
    s_range = (sal.max() - s_min) + 1e-8
    sal_norm = (sal - s_min) / s_range

    buckets: dict = {}   # category → [saliency_values]
    for i, (_, row) in enumerate(sel_df.iterrows()):
        cat, _ = get_poi_label(row)
        if cat == 'unknown':
            continue
        s = float(sal_norm[i]) if i < len(sal_norm) else 0.0
        buckets.setdefault(cat, []).append(s)

    result = []
    for cat, vals in buckets.items():
        result.append({
            'category':    cat,
            'char_dim':    _CHAR_MAP.get(cat, 'Community'),
            'avg_saliency': round(float(np.mean(vals)), 4),
            'max_saliency': round(float(np.max(vals)), 4),
            'count':        len(vals),
        })
    result.sort(key=lambda x: x['avg_saliency'], reverse=True)
    return result


# ---------------------------------------------------------------------------
# GeoSemantics v3 (optional – loaded only if geosemantics_v3.pt exists)
# ---------------------------------------------------------------------------

_v3_model    = None
v3_available = False

_v3_load_lock = _threading.Lock()

# Node type → character dimension mapping (mirrors geosemantics_v3.py constants)
_V3_NODE_TYPE_POI = 0
_V3_TYPE_CHAR_DIM = {1: 'Transport', 2: 'Nature', 3: 'Infrastructure', 4: 'Community'}
_V3_POI_CHAR_MAP  = {
    'amenity':    'Urban',
    'shop':       'Urban',
    'tourism':    'Tourism',
    'leisure':    'Community',
    'historic':   'Heritage',
    'healthcare': 'Community',
    'sport':      'Community',
    'office':     'Urban',
    'craft':      'Urban',
    'club':       'Community',
    'gambling':   'Urban',
    'vending':    'Urban',
}


def _try_load_v3():
    global _v3_model, v3_available
    v3_path   = os.path.join(BASE_DIR, 'geosemantics_v3.pt')
    meta_path = os.path.join(BASE_DIR, 'geosemantics_v3_meta.json')
    if not os.path.exists(v3_path) or not os.path.exists(meta_path):
        print('[inference] geosemantics_v3.pt not found – run: python geosemantics_v3.py',
              flush=True)
        return
    try:
        from geosemantics_v3 import GeoSemanticsV3
        with open(meta_path, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)
        model = GeoSemanticsV3(type_vocabs=meta['type_vocabs'])
        state = torch.load(v3_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'[inference] v3: {len(missing)} missing keys (new arch, needs retrain): {missing[:4]}',
                  flush=True)
        model.eval()
        _v3_model    = model
        v3_available = True
        print('[inference] GeoSemantics v3 loaded.', flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f'[inference] v3 load FAILED: {exc}', flush=True)


def ensure_v3_loaded():
    """Idempotent hot-reload: re-attempts loading if checkpoint appeared since startup."""
    if v3_available:
        return True
    with _v3_load_lock:
        if not v3_available:
            _try_load_v3()
    return v3_available


_try_load_v3()

_v3_resources = None


def _get_v3_resources():
    global _v3_resources
    if _v3_resources is None:
        if _v3_node_types is None:
            return None
        _v3_resources = (df, _v3_node_types, _v3_type_labels,
                         _v3_type_les, _v3_type_vocabs, _vectorizer, btree)
    return _v3_resources


def get_embedding_v3(query_lat, query_lon):
    """Return (64-d ndarray, attn_weights) using GeoSemantics v3, or (None, None)."""
    if _v3_model is None:
        return None, None
    res = _get_v3_resources()
    if res is None:
        return None, None
    from geosemantics_v3 import build_multiscale_graphs_v3
    graphs = build_multiscale_graphs_v3(query_lat, query_lon, external_resources=res)
    if graphs is None:
        return None, None
    with torch.no_grad():
        emb, attn = _v3_model(graphs)
    return emb.numpy(), attn.numpy()


# ---------------------------------------------------------------------------
# OSM Completeness / Confidence scoring
# ---------------------------------------------------------------------------

def get_confidence_score(query_lat, query_lon, radius=500):
    """
    Return a confidence score [0, 1] for the semantic character prediction.

    Factors:
      - Node count      (more nodes = more signal)
      - Type diversity  (V3: having multiple OSM types = richer context)
      - Tag richness    (nodes with many tags = more semantic content)
      - Graph density   (denser neighbourhood = more context)

    Returns dict with overall score and component breakdown.
    """
    q   = np.radians([[query_lat, query_lon]])
    idx = btree.query_radius(q, r=radius / 6_371_000.0)[0]
    n   = len(idx)
    if n == 0:
        return {'score': 0.0, 'label': 'No data', 'n_nodes': 0,
                'components': {}}

    # --- Node count score (log-scaled, saturates at ~200) ---
    count_score = min(1.0, math.log(n + 1) / math.log(201))

    # --- Tag richness: fraction of nodes with ≥1 other_tag ---
    sel_ot = df.iloc[idx]['other_tags'].fillna('')
    tagged = (sel_ot != '').sum()
    richness_score = tagged / max(n, 1)

    # --- Type diversity (V3 only) ---
    diversity_score = 0.5
    if _v3_node_types is not None:
        types_present = len(set(_v3_node_types[idx].tolist()))
        diversity_score = min(1.0, types_present / 5.0)

    # --- Graph density: how many KNN edges exist relative to theoretical max ---
    k      = min(8, n - 1)
    max_e  = n * k
    density_score = min(1.0, k / 8.0)  # k=8 → fully dense neighbourhood

    overall = (0.35 * count_score
             + 0.25 * richness_score
             + 0.25 * diversity_score
             + 0.15 * density_score)

    if overall >= 0.75:
        label = 'High confidence — dense and diverse OSM context'
    elif overall >= 0.50:
        label = 'Moderate confidence — reasonable OSM coverage'
    elif overall >= 0.25:
        label = 'Low confidence — sparse OSM features; character may be incomplete'
    else:
        label = 'Very low confidence — minimal OSM data in this area'

    return {
        'score': round(overall, 3),
        'label': label,
        'n_nodes': int(n),
        'components': {
            'node_count':     round(count_score, 3),
            'tag_richness':   round(richness_score, 3),
            'type_diversity': round(diversity_score, 3),
            'graph_density':  round(density_score, 3),
        },
    }


# ---------------------------------------------------------------------------
# Multi-scale character breakdown
# ---------------------------------------------------------------------------

_SCALE_RADII = [200, 700, 2000]
_SCALE_NAMES = ['micro', 'meso', 'macro']
_SCALE_LABELS = ['Micro (200 m)', 'Meso (700 m)', 'Macro (2 km)']


def get_multiscale_character(query_lat, query_lon):
    """
    Return character profiles at three spatial scales.

    micro  200 m  — street-level context
    meso   700 m  — neighbourhood / village scale
    macro 2000 m  — district / broader landscape

    Uses V3 character analysis when cache is available, else V2.
    Returns dict with 'scales' list and 'dominant_scale' name.
    """
    char_fn = get_location_character_v3 if _v3_node_types is not None else get_location_character

    scales = []
    for radius, name, label in zip(_SCALE_RADII, _SCALE_NAMES, _SCALE_LABELS):
        c = char_fn(query_lat, query_lon, radius)
        if c is None:
            scales.append({'name': name, 'label': label, 'radius': radius,
                           'char_dims': {}, 'label_str': '–', 'n_nodes': 0})
            continue
        dims  = c.get('char_dims', {})
        top_d = max(dims, key=dims.get) if dims else '–'
        scales.append({
            'name':      name,
            'label':     label,
            'radius':    radius,
            'char_dims': dims,
            'label_str': c.get('label', '–'),
            'dominant':  top_d,
            'n_nodes':   c.get('n_pois', 0) or c.get('n_labelled', 0),
            'source':    c.get('source', 'v2'),
        })

    # Dominant scale = scale where character is most pronounced (highest top dim score)
    def _top_score(s):
        d = s['char_dims']
        return max(d.values()) if d else 0.0

    dom_idx   = max(range(len(scales)), key=lambda i: _top_score(scales[i]))
    dom_scale = scales[dom_idx]['name']

    # Build natural-language multi-scale description
    nl = _multiscale_nl(scales, dom_scale)

    return {
        'scales':         scales,
        'dominant_scale': dom_scale,
        'nl':             nl,
    }


def _multiscale_nl(scales, dom_scale):
    descs = []
    for s in scales:
        d = s['char_dims']
        if not d:
            continue
        top = max(d, key=d.get)
        pct = round(d[top] * 100)
        descs.append(f'{s["label"]}: <b>{top}</b> ({pct}%)')

    if not descs:
        return 'Insufficient data for multi-scale analysis.'

    dom_labels = {
        'micro': 'street-level character — what is immediately within 200 m defines this place.',
        'meso':  'neighbourhood context shapes identity — the surrounding quarter matters most.',
        'macro': 'city/landscape scale defines identity — district structure overrides street details.',
    }
    dom_desc = dom_labels.get(dom_scale, 'multi-scale character.')
    joined   = ' · '.join(descs)
    return f'{joined}. Identity is <b>{dom_scale}</b>-dominant: {dom_desc}'


# ---------------------------------------------------------------------------
# Similar-place retrieval (requires precomputed morph data)
# ---------------------------------------------------------------------------

_morph_locations_cache = None


def _load_morph_locations():
    global _morph_locations_cache
    if _morph_locations_cache is not None:
        return _morph_locations_cache
    morph_path = os.path.join(BASE_DIR, '_poi_cache', 'morph_data.json')
    if not os.path.exists(morph_path):
        return None
    try:
        with open(morph_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        _morph_locations_cache = data.get('locations', [])
        return _morph_locations_cache
    except Exception:
        return None


def get_similar_places(query_lat, query_lon, top_k=5, mode='embedding', dim_filter=None):
    """
    Find the most similar precomputed locations to (query_lat, query_lon).

    mode:
      'embedding'  — cosine similarity of v2/v3 embedding vectors
      'character'  — Euclidean similarity on character dimension profile
      'dim'        — sort by a specific dimension score (dim_filter required)

    Returns list of dicts: {name, lat, lon, similarity, label, char_dims, dist_km}
    """
    locs = _load_morph_locations()
    if not locs:
        return []

    # Get query character/embedding
    char_q  = (get_location_character_v3(query_lat, query_lon)
               if _v3_node_types is not None
               else get_location_character(query_lat, query_lon))
    dims_q  = char_q['char_dims'] if char_q else {}
    emb_q, _ = get_embedding_v2(query_lat, query_lon)

    # Confidence of the query location — call scorer directly since char_q doesn't carry it.
    conf_q_obj = get_confidence_score(query_lat, query_lon)
    conf_q = conf_q_obj.get('score', None)
    # If query confidence is very low (sparse/noisy area), fall back to character mode
    # so that GNN embedding noise doesn't produce spurious similarity results.
    if mode == 'embedding' and conf_q is not None and conf_q < 0.30:
        mode = 'character'

    results = []
    for loc in locs:
        sim = None
        if mode == 'embedding' and emb_q is not None and loc.get('embedding'):
            ea  = np.array(loc['embedding'], dtype=float)
            if ea.shape != emb_q.shape:
                continue  # skip stale entries with wrong embedding dimension
            sim = float(cosine_sim(emb_q, ea))
            # Penalise pairs with a large confidence gap — a low-confidence
            # (sparse) area should not rank as highly similar to a dense city.
            conf_l = loc.get('confidence', 0.5)
            if conf_q is not None and conf_l:
                gap_penalty = min(conf_q, conf_l) / (max(conf_q, conf_l) + 1e-8)
                sim *= gap_penalty ** 0.5   # soft square-root dampening
            # Also apply character-based contrastive penalty on top of embedding sim
            if dims_q and loc.get('semantic_dims'):
                char_sim = character_dimension_similarity(dims_q, loc['semantic_dims'])
                # Blend: 70% embedding + 30% character to catch Nature↔Urban conflicts
                sim = 0.70 * sim + 0.30 * char_sim
        elif mode == 'character' and dims_q:
            dims_l = loc.get('semantic_dims', {})
            sim    = character_dimension_similarity(dims_q, dims_l)
        elif mode == 'dim' and dim_filter and loc.get('semantic_dims'):
            sim = loc['semantic_dims'].get(dim_filter, 0.0)

        if sim is None:
            continue

        dist_km = _hav(query_lon, query_lat, loc['lon'], loc['lat']) / 1000.0
        results.append({
            'name':       loc.get('name', '?'),
            'lat':        loc['lat'],
            'lon':        loc['lon'],
            'label':      loc.get('label', '–'),
            'similarity': round(float(sim), 4),
            'char_dims':  loc.get('semantic_dims', {}),
            'dist_km':    round(dist_km, 1),
            'id':         loc.get('id', -1),
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    # Exclude query location itself (within 500 m)
    results = [r for r in results if r['dist_km'] > 0.5]
    return results[:top_k]


def get_location_character_v3(query_lat, query_lon, radius=500):
    """
    V3-aware character analysis using ALL OSM node types.

    If a GNN classification head (_v3_gnn_clf) is trained, it bypasses the
    heuristic and predicts directly from the GNN embedding.
    """
    global _v3_node_types, _v3_gnn_clf
    
    _try_load_gnn_clf()
    if _v3_gnn_clf is not None:
        emb, _ = get_embedding_v3(query_lat, query_lon)
        if emb is not None:
            # Build 71-d combined feature: V3 GNN embedding (64) + saliency (7)
            sal_vec = _saliency_dims_vector(query_lat, query_lon, radius)
            feat = np.concatenate([emb, sal_vec]).astype(np.float64).reshape(1, -1)
            bundle = _v3_gnn_clf
            probs = bundle['clf'].predict_proba(feat)[0]
            classes = bundle['le'].classes_ if bundle['le'] is not None else bundle['clf'].classes_

            char_norm = {k: 0.0 for k in _CHAR_COLORS}
            char_dims = {k: 0.0 for k in _CHAR_COLORS}

            for i, cls in enumerate(classes):
                char_norm[cls] = round(float(probs[i]), 3)
                char_dims[cls] = round(float(probs[i]), 3)

            return {
                'char_dims': char_dims,
                'char_norm': char_norm
            }

    # Fallback to heuristic
    if _v3_node_types is None:
        return get_location_character(query_lat, query_lon, radius)

    q   = np.radians([[query_lat, query_lon]])
    
    # Dynamically expand radius in rural areas to find context
    current_radius = radius
    idx = btree.query_radius(q, r=current_radius / 6_371_000.0)[0]
    while len(idx) < 5 and current_radius < 5000:
        current_radius += 500
        idx = btree.query_radius(q, r=current_radius / 6_371_000.0)[0]
        
    if len(idx) < 3:
        return None

    sel_ntypes = _v3_node_types[idx]
    sel_tlbls  = _v3_type_labels[idx]

    sal, _, _ = get_saliency(query_lat, query_lon, radius)
    if sal is not None and len(sal) == len(idx):
        s_min    = sal.min()
        s_range  = (sal.max() - s_min) + 1e-8
        sal_norm = (sal - s_min) / s_range
    else:
        sal_norm = np.ones(len(idx))

    char_raw  = {k: 0.0 for k in _CHAR_COLORS}
    n_counted = 0
    n_urban_indicators = 0

    for i, nt in enumerate(sel_ntypes):
        weight  = float(sal_norm[i]) * 0.7 + 0.3
        nt_int  = int(nt)
        if nt_int == _V3_NODE_TYPE_POI:
            cat_str = _v3_type_les[0].classes_[int(sel_tlbls[i])]  # e.g. "amenity=cafe"
            poi_key = cat_str.split('=')[0]
            dim     = _V3_POI_CHAR_MAP.get(poi_key, 'Community')
            
            if poi_key in ['amenity', 'shop', 'office', 'healthcare', 'craft', 'leisure', 'tourism']:
                n_urban_indicators += 1

            if poi_key == 'tourism':
                weight *= 100.0  # Boost Tourism
            elif poi_key == 'historic':
                weight *= 100.0  # Boost Heritage
            else:
                weight *= 2.0  # Default POI voice
        else:
            dim = _V3_TYPE_CHAR_DIM.get(nt_int, 'Community')
            
            if nt_int == 2: # Nature
                weight *= 50.0  # High weight for Rural/Alpine areas
            elif nt_int == 1: # Transport
                lbl = _v3_type_les[1].classes_[int(sel_tlbls[i])]
                if 'aeroway' in lbl or 'station' in lbl:
                    dim = 'Transport'
                    weight *= 100.0
                else:
                    weight *= 0.5
            elif nt_int == 3: # Built / Infrastructure
                lbl = _v3_type_les[3].classes_[int(sel_tlbls[i])]
                if 'industrial' in lbl or 'commercial' in lbl or 'works' in lbl or 'power' in lbl:
                    dim = 'Infrastructure'
                    weight *= 50.0
                elif 'residential' in lbl or 'house' in lbl or 'apartments' in lbl:
                    dim = 'Community'
                    weight *= 10.0
                else:
                    dim = 'Infrastructure'
                    weight *= 0.1 # Prevent generic buildings from dominating
                n_urban_indicators += 1
            elif nt_int == 4: # Place
                place_lbl = _v3_type_les[4].classes_[int(sel_tlbls[i])]
                if 'city' in place_lbl or 'quarter' in place_lbl or 'neighbourhood' in place_lbl or 'square' in place_lbl:
                    dim = 'Urban'
                elif 'town' in place_lbl or 'village' in place_lbl or 'suburb' in place_lbl:
                    dim = 'Community'
                else:
                    weight *= 0.1 # Ignore random place tags like place=yes
                weight *= 10.0 # Massive weight for major place tags
        char_raw[dim] += weight
        n_counted += 1

    if n_counted == 0:
        return {
            'char_dims': {'Nature': 1.0},
            'char_norm': {'Nature': 1.0}
        }

    char_raw = _apply_contextual_suppression(char_raw, n_urban_indicators)

    total     = sum(char_raw.values()) + 1e-8
    char_norm = {k: round(v / total, 3) for k, v in char_raw.items()}
    return {
        'char_dims':   char_norm,
        'char_colors': _CHAR_COLORS,
        'label':       _location_label(char_norm, n_counted),
        'n_pois':      len(idx),
        'n_labelled':  n_counted,
        'source':      'v3',
    }
