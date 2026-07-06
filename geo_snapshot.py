"""
geo_snapshot.py — Run the trained GeoSemantics models on an arbitrary POI
snapshot that is NOT the cached, present-day Austria-wide dataset.

Two callers need exactly this:
  - temporal.py       (historical Overpass snapshots — different point in time)
  - morph_app.py       (counterfactual what-if — present data + synthetic edits)

All three reuse the GLOBAL trained saliency GNN, the GLOBAL trained V2 GATv2
embedding model, the GLOBAL category LabelEncoder and other-tags vectorizer
(all loaded once by inference.py), and the rule-based V3 node-typing logic
from geosemantics_v3.py.  Nothing here is retrained — every analysis is a
genuine forward pass of the existing weights on new input data.
"""
import math

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

import inference as inf
from geosemantics_v3 import _classify_node_types_vectorized, _get_type_category, NODE_TYPE_POI

_TYPE_CHAR_DIM = inf._V3_TYPE_CHAR_DIM
_POI_CHAR_MAP  = inf._V3_POI_CHAR_MAP
_CHAR_COLORS   = inf._CHAR_COLORS

_DEDICATED_COLS = ('highway', 'natural', 'place', 'man_made')


def records_to_df(records):
    """records: list of {'lat','lon','tags': {...}}.

    Produces a DataFrame with the exact column schema inference.py /
    geosemantics_v3.py expect (highway/natural/place/man_made dedicated
    columns + an 'other_tags' hstore-format string for everything else).
    """
    rows = []
    for r in records:
        tags = r['tags']
        ded = {k: tags.get(k, '') for k in _DEDICATED_COLS}
        other = {k: v for k, v in tags.items() if k not in _DEDICATED_COLS and v}
        other_tags_str = ','.join(f'"{k}"=>"{v}"' for k, v in other.items())
        rows.append({
            'lat': r['lat'], 'lon': r['lon'],
            'highway':  ded['highway'],  'natural':  ded['natural'],
            'place':    ded['place'],    'man_made': ded['man_made'],
            'other_tags': other_tags_str,
            'name': tags.get('name', ''),
        })
    return pd.DataFrame(rows)


def build_resources(rec_df):
    """Build a (df, cat_labels, le, vectorizer, btree) tuple — the exact
    shape geosemantics_v2.build_multiscale_graphs(external_resources=...)
    and inference.py's local-graph builder expect — reusing the GLOBAL
    trained label encoder + vectorizer so node features stay compatible
    with the trained model weights. Unseen category values fall back to
    '' (the catch-all bucket every trained encoder already has, since most
    real POIs carry their semantic tag in other_tags, not the 4 dedicated
    columns)."""
    le    = inf.le
    known = set(le.classes_)
    h = rec_df['highway'].values; n = rec_df['natural'].values
    p = rec_df['place'].values;   m = rec_df['man_made'].values
    cat_primary = np.where(h != '', h,
                  np.where(n != '', n,
                  np.where(p != '', p,
                  np.where(m != '', m, ''))))
    cat_primary = np.array([c if c in known else '' for c in cat_primary], dtype=object)
    cat_labels  = le.transform(cat_primary).astype(np.int32)

    coords_rad = np.radians(rec_df[['lat', 'lon']].values)
    btree = BallTree(coords_rad, metric='haversine')
    return rec_df, cat_labels, le, inf.vectorizer, btree


def local_graph(query_lat, query_lon, radius, res):
    """Mirrors inference.build_local_graph but over an externally supplied
    resource tuple instead of the global Austria-wide cache."""
    rec_df, cat_labels, le, vectorizer, btree = res
    q   = np.radians([[query_lat, query_lon]])
    idx = btree.query_radius(q, r=radius / 6_371_000.0)[0]
    if len(idx) < 3:
        return None, None

    sel_df     = rec_df.iloc[idx].reset_index(drop=True)
    coords     = sel_df[['lat', 'lon']].values
    sel_labels = cat_labels[idx]
    n_cat      = len(le.classes_)
    sel_cat    = np.zeros((len(idx), n_cat), dtype=np.float32)
    sel_cat[np.arange(len(idx)), sel_labels] = 1.0

    sub_keys  = inf._other_tags_keys(sel_df['other_tags'].tolist())
    sel_other = vectorizer.transform(sub_keys).toarray().astype(np.float32)

    coords_rel = StandardScaler().fit_transform(
        coords - np.array([query_lat, query_lon]))
    node_feats = np.hstack([coords_rel, sel_cat, sel_other]).astype(np.float32).copy()

    k = min(inf.K_NEIGHBORS, len(sel_df) - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbr.kneighbors(coords)

    edges, eattr = [], []
    for i, neighs in enumerate(indices):
        for nb in neighs[1:]:
            d = inf._hav(coords[i, 1], coords[i, 0], coords[nb, 1], coords[nb, 0])
            b = inf._brng(coords[i, 1], coords[i, 0], coords[nb, 1], coords[nb, 0])
            same = int(sel_labels[i] == sel_labels[nb])
            edges.append([i, nb])
            eattr.append([d, math.sin(b), math.cos(b), same])

    if not edges:
        return None, None

    g = Data(
        x=torch.tensor(node_feats),
        edge_index=torch.tensor(np.array(edges), dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(np.array(eattr, dtype=np.float32)),
    )
    return g, sel_df


def character_from_records(rec_df, query_lat, query_lon, radius=600):
    """Saliency-weighted character profile, computed by the SAME rule
    (V3 node-typing + trained saliency GNN weighting) as
    inference.get_location_character_v3 — just over an externally supplied
    snapshot instead of the cached Austria-wide dataset."""
    res = build_resources(rec_df)
    g, sel_df = local_graph(query_lat, query_lon, radius, res)
    if g is None:
        return None

    with torch.no_grad():
        sal = inf._saliency_model(g.x, g.edge_index).numpy()
    s_min = sal.min(); s_range = (sal.max() - s_min) + 1e-8
    sal_norm = (sal - s_min) / s_range

    node_types = _classify_node_types_vectorized(sel_df)
    hw  = sel_df['highway'].fillna('').tolist()
    nat = sel_df['natural'].fillna('').tolist()
    pl  = sel_df['place'].fillna('').tolist()
    mm  = sel_df['man_made'].fillna('').tolist()
    ot  = sel_df['other_tags'].fillna('').tolist()

    char_raw  = {k: 0.0 for k in _CHAR_COLORS}
    n_counted = 0
    for i, nt in enumerate(node_types):
        weight = float(sal_norm[i]) * 0.7 + 0.3
        nt_int = int(nt)
        if nt_int == NODE_TYPE_POI:
            cat_str = _get_type_category(nt_int, hw[i], nat[i], pl[i], mm[i], ot[i])
            dim = _POI_CHAR_MAP.get(cat_str.split('=')[0], 'Community')
        else:
            dim = _TYPE_CHAR_DIM.get(nt_int, 'Community')
        char_raw[dim] += weight
        n_counted += 1

    if n_counted == 0:
        return None
    total     = sum(char_raw.values()) + 1e-8
    char_norm = {k: round(v / total, 3) for k, v in char_raw.items()}
    return {
        'char_dims': char_norm,
        'label':     inf._location_label(char_norm, n_counted),
        'n_pois':    len(sel_df),
        'source':    'v3-rule+saliency',
    }


def embedding_from_records(rec_df, query_lat, query_lon):
    """Genuine forward pass of the trained V2 GATv2 contrastive embedding
    model on an externally supplied snapshot. Returns (64-d ndarray,
    scale-attention ndarray) or (None, None)."""
    if not inf.v2_available:
        return None, None
    res = build_resources(rec_df)
    from geosemantics_v2 import build_multiscale_graphs
    graphs = build_multiscale_graphs(query_lat, query_lon, external_resources=res)
    if graphs is None:
        return None, None
    with torch.no_grad():
        emb, attn = inf._v2_model(graphs)
    return emb.numpy(), attn.numpy()
