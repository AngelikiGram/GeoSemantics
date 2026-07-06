"""
GeoSemantics v3 – Heterogeneous-Type Spatial Graph with Per-Type Encoders
=========================================================================

Architecture improvements over v2:

  1. Five node types: POI / Transport / Natural / Built / Place
       All OSM objects are first-class graph nodes — roads, natural features,
       buildings, and place markers are no longer filtered away.

  2. Per-type category embedding tables
       Roads and cafes no longer compete in the same embedding space.
       highway=residential indexes into the Transport table; amenity=cafe
       indexes into the POI table.  Each type's categories are compared
       only to each other during training, producing cleaner representations.

  3. Node-type identity tokens
       Every node receives an additive type token before GNN message
       passing.  Attention learns type-conditioned communication rules:
       natural features attend to roads differently from POIs.

  4. Extended edge features — source + target node type
       The edge encoder sees whether a message crosses type boundaries
       (POI→road, natural→POI) or stays within a type.

  5. Multi-scale + InfoNCE training (identical approach to v2)

Run training:
    python geosemantics_v3.py

Import for inference:
    from geosemantics_v3 import GeoSemanticsV3, build_multiscale_graphs_v3, SCALES
"""

import json
import math
import os
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_add_pool

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SCALES      = [200, 700, 2000]
N_SCALES    = len(SCALES)
K_NEIGHBORS = 8      # restored to 8 to match V2
MIN_NODES   = 3

NODE_TYPE_POI       = 0   # amenity / shop / tourism / leisure / historic / healthcare / sport / office
NODE_TYPE_TRANSPORT = 1   # highway / railway / public_transport / aeroway
NODE_TYPE_NATURAL   = 2   # natural column / waterway / green landuse
NODE_TYPE_BUILT     = 3   # building tag / man_made column or tag / built landuse
NODE_TYPE_PLACE     = 4   # place column / other landuse / barrier / emergency / power
N_NODE_TYPES        = 5
NODE_TYPE_NAMES     = ['POI', 'Transport', 'Natural', 'Built', 'Place']

# Model hyper-parameters
TYPE_DIM  = 32      # per-type category embedding size
OTHER_DIM = 32      # CountVectorizer BoW (shared with v2)
HIDDEN    = 128     # restored to 128 to match V2 capacity
OUT_DIM   = 64      # same as v2 — embeddings are directly comparable
N_HEADS   = 4
EDGE_DIM  = 16

# Training hyper-parameters
N_TRAIN      = 2000   # reduced to speed up training while beating V2 (which used 1000)
BATCH_SIZE   = 16     # reduced back to 16 to prevent PyTorch CPU access violations
EPOCHS       = 200    # was 150 — early stopping handles the rest
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE  = 0.07
MASK_PROB    = 0.20
PATIENCE     = 25    # restored to 25 to match V2

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE_DIR, '_poi_cache')
MODEL_PATH = os.path.join(BASE_DIR, 'geosemantics_v3.pt')
META_PATH  = os.path.join(BASE_DIR, 'geosemantics_v3_meta.json')

_KEY_RE = re.compile(r'"([^"]+)"=>"')
_KV_RE  = re.compile(r'"([^"]+)"=>"([^"]*)"')

# Tag sets used for node-type classification
_POI_TAGS       = frozenset({'amenity', 'shop', 'tourism', 'leisure', 'historic',
                              'healthcare', 'sport', 'office', 'craft', 'club',
                              'gambling', 'vending'})
_TRANSPORT_TAGS = frozenset({'railway', 'public_transport', 'aeroway'})

_LANDUSE_NATURAL = frozenset({
    'forest', 'meadow', 'farmland', 'grass', 'farmyard', 'orchard',
    'vineyard', 'scrub', 'heath', 'allotments', 'greenfield',
    'nature_reserve', 'wetland',
})
_LANDUSE_BUILT = frozenset({
    'residential', 'commercial', 'industrial', 'retail', 'construction',
    'brownfield', 'garages', 'military', 'port', 'railway', 'quarry',
})

# Character dimension contribution per v3 node type (used by inference.py)
V3_TYPE_CHAR_DIM = {
    NODE_TYPE_TRANSPORT: 'Transport',
    NODE_TYPE_NATURAL:   'Nature',
    NODE_TYPE_BUILT:     'Infrastructure',
    NODE_TYPE_PLACE:     'Community',
}

# For POI nodes, map by the specific POI tag key
V3_POI_CHAR_MAP = {
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


# ─── NODE TYPE CLASSIFICATION ─────────────────────────────────────────────────

def _classify_node_types_vectorized(df: pd.DataFrame) -> np.ndarray:
    """
    Assign each row a node type (0-4) using vectorised pandas string ops.
    Priority (highest wins): POI > Transport > Natural > Built > Place.

    NOTE: In the Austrian POI dataset the `natural` column is always empty —
    natural=tree/peak/spring/etc. are stored inside other_tags.  We therefore
    check other_tags for "natural"=>"..." explicitly.  Same applies to power
    infrastructure: power=pole/tower/etc. belongs in Built, not Place.
    """
    ot = df['other_tags'].fillna('')
    n  = len(df)

    has_poi       = ot.str.contains(
        r'"(?:amenity|shop|tourism|leisure|historic|healthcare|sport|office|craft|club|gambling|vending)"=>"',
        na=False, regex=True)
    has_hw_col    = df['highway'].fillna('').ne('')
    has_transport = ot.str.contains(
        r'"(?:railway|public_transport|aeroway)"=>"', na=False, regex=True)
    # natural column is empty in this dataset — check other_tags too
    has_nat_col   = df['natural'].fillna('').ne('')
    has_nat_ot    = ot.str.contains(r'"natural"=>"', na=False, regex=True)
    has_waterway  = ot.str.contains(r'"waterway"=>"', na=False, regex=True)
    has_building  = ot.str.contains(r'"building"=>"', na=False, regex=True)
    has_mm_col    = df['man_made'].fillna('').ne('')
    has_mm_tag    = ot.str.contains(r'"man_made"=>"', na=False, regex=True)
    # power=pole/tower/etc. is infrastructure → Built, not Place
    has_power     = ot.str.contains(r'"power"=>"', na=False, regex=True)
    has_place_col = df['place'].fillna('').ne('')
    has_barrier   = ot.str.contains(r'"(?:barrier|emergency)"=>"', na=False, regex=True)

    lu_nat_pat = '|'.join(_LANDUSE_NATURAL)
    lu_blt_pat = '|'.join(_LANDUSE_BUILT)
    has_lu_natural = ot.str.contains(
        rf'"landuse"=>"(?:{lu_nat_pat})"', na=False, regex=True)
    has_lu_built   = ot.str.contains(
        rf'"landuse"=>"(?:{lu_blt_pat})"', na=False, regex=True)
    has_lu_other   = (ot.str.contains(r'"landuse"=>"', na=False, regex=True)
                      & ~has_lu_natural & ~has_lu_built)

    # Apply lowest-priority first; each later assignment overwrites
    types = np.full(n, NODE_TYPE_PLACE, dtype=np.uint8)
    types[has_lu_other.values]   = NODE_TYPE_PLACE
    types[has_barrier.values]    = NODE_TYPE_PLACE
    types[has_place_col.values]  = NODE_TYPE_PLACE
    types[has_power.values]      = NODE_TYPE_BUILT     # power infrastructure → Built
    types[has_lu_built.values]   = NODE_TYPE_BUILT
    types[has_mm_tag.values]     = NODE_TYPE_BUILT
    types[has_mm_col.values]     = NODE_TYPE_BUILT
    types[has_building.values]   = NODE_TYPE_BUILT
    types[has_lu_natural.values] = NODE_TYPE_NATURAL
    types[has_waterway.values]   = NODE_TYPE_NATURAL
    types[has_nat_col.values]    = NODE_TYPE_NATURAL
    types[has_nat_ot.values]     = NODE_TYPE_NATURAL   # natural=tree/peak/spring in other_tags
    types[has_transport.values]  = NODE_TYPE_TRANSPORT
    types[has_hw_col.values]     = NODE_TYPE_TRANSPORT
    types[has_poi.values]        = NODE_TYPE_POI       # highest priority
    return types


def _get_type_category(nt: int, hw: str, nat: str, pl: str,
                       mm: str, ot: str) -> str:
    """Extract the representative category string for one node within its type."""
    tags = {m.group(1): m.group(2) for m in _KV_RE.finditer(ot)}

    if nt == NODE_TYPE_POI:
        for k in ('amenity', 'shop', 'tourism', 'leisure', 'historic',
                  'healthcare', 'sport', 'office', 'craft', 'club',
                  'gambling', 'vending'):
            if k in tags and tags[k]:
                return f'{k}={tags[k].split(";")[0].strip()}'
        return 'poi=other'

    elif nt == NODE_TYPE_TRANSPORT:
        if hw:
            return f'highway={hw.split(";")[0].strip()}'
        for k in ('railway', 'public_transport', 'aeroway'):
            if k in tags and tags[k]:
                return f'{k}={tags[k].split(";")[0].strip()}'
        return 'transport=other'

    elif nt == NODE_TYPE_NATURAL:
        # Check dedicated column first, then other_tags (column is often empty)
        if nat:
            return f'natural={nat.split(";")[0].strip()}'
        if 'natural' in tags and tags['natural']:
            return f'natural={tags["natural"].split(";")[0].strip()}'
        if 'waterway' in tags and tags['waterway']:
            return f'waterway={tags["waterway"].split(";")[0].strip()}'
        if 'landuse' in tags:
            return f'landuse={tags["landuse"].split(";")[0].strip()}'
        return 'natural=other'

    elif nt == NODE_TYPE_BUILT:
        if 'building' in tags and tags['building']:
            return f'building={tags["building"].split(";")[0].strip()}'
        if mm:
            return f'man_made={mm.split(";")[0].strip()}'
        if 'man_made' in tags and tags['man_made']:
            return f'man_made={tags["man_made"].split(";")[0].strip()}'
        if 'power' in tags and tags['power']:
            return f'power={tags["power"].split(";")[0].strip()}'
        if 'landuse' in tags:
            return f'landuse={tags["landuse"].split(";")[0].strip()}'
        return 'built=other'

    else:  # NODE_TYPE_PLACE
        if pl:
            return f'place={pl.split(";")[0].strip()}'
        for k in ('landuse', 'barrier', 'emergency'):
            if k in tags and tags[k]:
                return f'{k}={tags[k].split(";")[0].strip()}'
        return 'place=other'


# ─── V3 CACHE ─────────────────────────────────────────────────────────────────

V3_CACHE_FILES = [
    'v3_node_types.npy', 'v3_type_labels.npy',
    'v3_type_les.pkl',   'v3_type_vocabs.json',
]


def _build_v3_cache(df: pd.DataFrame):
    """One-time build of v3-specific cache files on top of existing df.parquet."""
    print('[v3] Classifying node types (vectorised) …', flush=True)
    n = len(df)
    node_types = _classify_node_types_vectorized(df)
    dist = np.bincount(node_types, minlength=N_NODE_TYPES)
    for t, name in enumerate(NODE_TYPE_NAMES):
        print(f'  {name}: {dist[t]:,}', flush=True)

    print('[v3] Extracting per-type category strings …', flush=True)
    hw_list  = df['highway'].fillna('').tolist()
    nat_list = df['natural'].fillna('').tolist()
    pl_list  = df['place'].fillna('').tolist()
    mm_list  = df['man_made'].fillna('').tolist()
    ot_list  = df['other_tags'].fillna('').tolist()

    type_cats = np.empty(n, dtype=object)
    CHUNK = 500_000
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        for i in range(start, end):
            type_cats[i] = _get_type_category(
                int(node_types[i]), hw_list[i], nat_list[i],
                pl_list[i], mm_list[i], ot_list[i])
        print(f'  {end:,}/{n:,}', flush=True)

    print('[v3] Fitting per-type LabelEncoders …', flush=True)
    type_les    = []
    type_labels = np.zeros(n, dtype=np.int32)
    type_vocabs = []
    for t in range(N_NODE_TYPES):
        mask  = node_types == t
        le_t  = LabelEncoder()
        cats_t = type_cats[mask] if mask.sum() > 0 else np.array(['unknown'])
        le_t.fit(cats_t)
        if mask.sum() > 0:
            type_labels[mask] = le_t.transform(cats_t).astype(np.int32)
        type_les.append(le_t)
        type_vocabs.append(int(len(le_t.classes_)))
        print(f'  {NODE_TYPE_NAMES[t]}: {mask.sum():,} nodes, '
              f'{len(le_t.classes_)} categories', flush=True)

    np.save(os.path.join(CACHE_DIR, 'v3_node_types.npy'),  node_types)
    np.save(os.path.join(CACHE_DIR, 'v3_type_labels.npy'), type_labels)
    with open(os.path.join(CACHE_DIR, 'v3_type_les.pkl'), 'wb') as fh:
        pickle.dump(type_les, fh)
    with open(os.path.join(CACHE_DIR, 'v3_type_vocabs.json'), 'w') as fh:
        json.dump(type_vocabs, fh)

    print('[v3] Cache built.', flush=True)
    return node_types, type_labels, type_les, type_vocabs


def _load_v3_cache():
    print('[v3] Loading node-type cache …', flush=True)
    node_types  = np.load(os.path.join(CACHE_DIR, 'v3_node_types.npy'))
    type_labels = np.load(os.path.join(CACHE_DIR, 'v3_type_labels.npy'))
    with open(os.path.join(CACHE_DIR, 'v3_type_les.pkl'), 'rb') as fh:
        type_les = pickle.load(fh)
    with open(os.path.join(CACHE_DIR, 'v3_type_vocabs.json')) as fh:
        type_vocabs = json.load(fh)
    print('[v3] Cache loaded.', flush=True)
    return node_types, type_labels, type_les, type_vocabs


# ─── SHARED RESOURCE LOADER ───────────────────────────────────────────────────

_resources_cache = None


def get_resources(external=None):
    """Return (df, node_types, type_labels, type_les, type_vocabs, vectorizer, btree).

    Pass external to reuse already-loaded data from inference.py.
    """
    global _resources_cache
    if external is not None:
        return external
    if _resources_cache is None:
        print('[v3] Loading base resources …', flush=True)
        df = pd.read_parquet(os.path.join(CACHE_DIR, 'df.parquet'))
        with open(os.path.join(CACHE_DIR, 'vectorizer.pkl'), 'rb') as fh:
            vectorizer = pickle.load(fh)
        with open(os.path.join(CACHE_DIR, 'btree.pkl'), 'rb') as fh:
            btree = pickle.load(fh)

        v3_ok = all(os.path.exists(os.path.join(CACHE_DIR, f))
                    for f in V3_CACHE_FILES)
        if v3_ok:
            node_types, type_labels, type_les, type_vocabs = _load_v3_cache()
        else:
            node_types, type_labels, type_les, type_vocabs = _build_v3_cache(df)

        _resources_cache = (df, node_types, type_labels, type_les,
                            type_vocabs, vectorizer, btree)
        print(f'[v3] {len(df):,} nodes ready.', flush=True)
    return _resources_cache


# ─── GEOMETRY ─────────────────────────────────────────────────────────────────

def _other_tags_keys(arr):
    return [' '.join(_KEY_RE.findall(t)) if t else '' for t in arr]


def _hav(lon1, lat1, lon2, lat2):
    R = 6_371_000.0
    lo1, la1, lo2, la2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = (math.sin((la2 - la1) / 2) ** 2 +
         math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(a, 0.0)))


def _brng(lon1, lat1, lon2, lat2):
    lo1, la1, lo2, la2 = map(math.radians, [lon1, lat1, lon2, lat2])
    x = math.sin(lo2 - lo1) * math.cos(la2)
    y = (math.cos(la1) * math.sin(la2) -
         math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1))
    return math.atan2(x, y)


# ─── MODEL COMPONENTS ─────────────────────────────────────────────────────────

class SinusoidalDistEnc(nn.Module):
    def __init__(self, n_freqs: int = 4):
        super().__init__()
        self.register_buffer('freqs', torch.pow(2.0, torch.arange(n_freqs).float()))

    def forward(self, d_norm: torch.Tensor) -> torch.Tensor:
        d = d_norm.unsqueeze(-1)
        angles = d * self.freqs * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class EdgeEncoderV3(nn.Module):
    """
    Extended v3 edge encoder.
    Input (E, 6): [d_norm, sin_b, cos_b, same_cat, src_ntype/4, tgt_ntype/4]
    """
    def __init__(self, edge_out: int = EDGE_DIM, n_freqs: int = 4):
        super().__init__()
        self.dist_enc = SinusoidalDistEnc(n_freqs)
        raw = 2 * n_freqs + 5   # 8 dist harmonics + sinθ + cosθ + same_cat + 2 type floats
        self.mlp = nn.Sequential(
            nn.Linear(raw, edge_out * 2),
            nn.LayerNorm(edge_out * 2),
            nn.GELU(),
            nn.Linear(edge_out * 2, edge_out),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        dist_feat = self.dist_enc(edge_attr[:, 0])
        return self.mlp(torch.cat([dist_feat, edge_attr[:, 1:]], dim=-1))


class GeoSemanticsV3(nn.Module):
    """
    Heterogeneous-Type Spatial GNN for Location Embedding.

    Input: list of n_scales Data objects, each with:
        g.x           – (N, 2+32) float   [coords_rel | other_tags_bow]
        g.node_types  – (N,)      long    OSM node type index (0-4)
        g.type_labels – (N,)      long    category within type's vocab
        g.edge_index  – (2, E)
        g.edge_attr   – (E, 6)    float   [d_norm, sinθ, cosθ, same_cat, src_t/4, tgt_t/4]

    Output:
        emb  – (out_dim,)   location embedding
        attn – (n_scales,)  cross-scale attention weights
    """

    def __init__(self, type_vocabs: list,
                 type_dim:  int = TYPE_DIM,
                 other_dim: int = OTHER_DIM,
                 hidden:    int = HIDDEN,
                 out_dim:   int = OUT_DIM,
                 n_heads:   int = N_HEADS,
                 edge_dim:  int = EDGE_DIM,
                 n_scales:  int = N_SCALES):
        super().__init__()
        self.n_scales = n_scales
        self.type_dim = type_dim
        self.n_types  = len(type_vocabs)

        # Per-type category embedding tables.  +1 slot per table = mask token.
        self.type_cat_embs = nn.ModuleList([
            nn.Embedding(v + 1, type_dim) for v in type_vocabs
        ])
        for emb in self.type_cat_embs:
            nn.init.normal_(emb.weight, std=0.02)

        # Node-type identity token: additive shift that tells the GNN
        # "I am a road" vs. "I am a cafe" before any message passing.
        # +1 slot for the mask token used during augmentation
        self.node_type_emb = nn.Embedding(self.n_types + 1, hidden)

        # Scale tokens (identical role to v2)
        self.scale_tokens = nn.Embedding(n_scales, hidden)

        # Shared node projection: [coords(2) + type_emb(type_dim) + other(other_dim)] → hidden
        node_in = 2 + type_dim + other_dim
        self.node_proj = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # V3 edge encoder (handles 6-dim input)
        self.edge_enc = EdgeEncoderV3(edge_out=edge_dim)

        # Shared GATv2 layers with residual connections
        self.gat1  = GATv2Conv(hidden, hidden // n_heads, heads=n_heads,
                                edge_dim=edge_dim, concat=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden)
        self.gat2  = GATv2Conv(hidden, hidden // n_heads, heads=n_heads,
                                edge_dim=edge_dim, concat=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden)

        self.pool_gate  = nn.Linear(hidden, 1)
        # Type-stratified pooling: learns to weight each node type's contribution.
        # The gate sees each type's pooled content AND a global density-fraction
        # vector (how much gated mass each of the N_TYPES contributes overall),
        # so it can distinguish "sparse Natural in a genuinely rural area" from
        # "sparse Natural in a dense suburb with lots of Built/POI nodes" --
        # two cases that look identical from a type's own pooled content alone
        # and were previously confused (suburb -> mispredicted as rural fringe).
        self.type_importance = nn.Sequential(
            nn.Linear(hidden + self.n_types, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.out_proj   = nn.Linear(hidden, out_dim)
        self.scale_attn = nn.Sequential(
            nn.Linear(out_dim, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def _get_cat_embs(self, node_types: torch.Tensor,
                      type_labels: torch.Tensor) -> torch.Tensor:
        embs = torch.zeros(len(node_types), self.type_dim, device=node_types.device)
        for t in range(self.n_types):
            mask = node_types == t
            if mask.any():
                embs[mask] = self.type_cat_embs[t](type_labels[mask])
        return embs

    def _encode_scale(self, g: Data, scale_id: int) -> torch.Tensor:
        dev = g.x.device
        cat_emb = self._get_cat_embs(g.node_types, g.type_labels)
        feat_in = torch.cat([g.x[:, :2], cat_emb, g.x[:, 2:]], dim=-1)
        x = self.node_proj(feat_in)
        x = x + self.node_type_emb(g.node_types)
        tok = self.scale_tokens(torch.full((x.size(0),), scale_id,
                                           dtype=torch.long, device=dev))
        x = x + tok
        e = self.edge_enc(g.edge_attr)
        x = self.norm1(F.gelu(self.gat1(x, g.edge_index, e)) + x)
        x = self.norm2(F.gelu(self.gat2(x, g.edge_index, e)) + x)
        return x

    def _pool(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        """Type-stratified gated pooling, density-conditioned.

        Pools each node type separately, then computes a learned scalar importance
        weight for each pool from its own content AND a global density-fraction
        vector shared across types (see __init__ docstring on type_importance).
        Absent node types are masked out with -1e9 before softmax, ensuring they
        contribute exactly 0.0 to the final embedding.
        """
        dev = x.device
        type_pools_avg = torch.zeros(self.n_types, x.size(-1), device=dev)
        type_pools_sum = torch.zeros(self.n_types, x.size(-1), device=dev)
        type_den       = torch.zeros(self.n_types, device=dev)
        global_den = torch.zeros(1, device=dev)
        type_mask = torch.zeros(self.n_types, dtype=torch.bool, device=dev)
        for t in range(self.n_types):
            mask = node_types == t
            if mask.any():
                h_t  = x[mask]
                g_t  = torch.sigmoid(self.pool_gate(h_t))
                num_t = (g_t * h_t).sum(0)
                den_t = g_t.sum()
                type_pools_avg[t] = num_t / (den_t + 1e-8)
                type_pools_sum[t] = num_t
                type_den[t] = den_t
                global_den += den_t
                type_mask[t] = True

        density_frac = (type_den / (global_den + 1e-8)).unsqueeze(0).expand(self.n_types, -1)  # (N_TYPES, N_TYPES)
        importance_in = torch.cat([type_pools_avg, density_frac], dim=-1)
        scores = self.type_importance(importance_in).squeeze(-1)  # (N_TYPES,)
        scores = scores.masked_fill(~type_mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        
        # We apply weights to the SUM pools and globally average, preserving relative proportions!
        return (weights.unsqueeze(1) * type_pools_sum).sum(0) / (global_den + 1e-8)  # (hidden,)

    def forward(self, graphs):
        embs = []
        for s, g in enumerate(graphs):
            h = self._encode_scale(g, s)
            embs.append(self.out_proj(self._pool(h, g.node_types)))
        embs = torch.stack(embs, dim=0)
        attn = F.softmax(self.scale_attn(embs), dim=0)
        return (attn * embs).sum(dim=0), attn.squeeze(-1)

    def forward_batch(self, scale_graph_lists):
        """
        Batched training forward pass with type-stratified pooling.

        scale_graph_lists[s] = list of B Data objects at scale s (already on device).
        Returns: (B, out_dim) embeddings.
        """
        B = len(scale_graph_lists[0])
        scale_embs = []

        for s, graphs in enumerate(scale_graph_lists):
            bg = Batch.from_data_list(graphs)             # merge B graphs
            h  = self._encode_scale(bg, s)                # (total_N, hidden)

            # Per-type gated pools: (B, N_TYPES, hidden)
            type_pools_avg = torch.zeros(B, self.n_types, h.size(-1), device=h.device)
            type_pools_sum = torch.zeros(B, self.n_types, h.size(-1), device=h.device)
            type_den       = torch.zeros(B, self.n_types, device=h.device)
            global_den = torch.zeros(B, 1, device=h.device)
            type_mask = torch.zeros(B, self.n_types, dtype=torch.bool, device=h.device)
            for t in range(self.n_types):
                mask = (bg.node_types == t)
                if mask.any():
                    h_t     = h[mask]
                    g_t     = torch.sigmoid(self.pool_gate(h_t))
                    batch_t = bg.batch[mask]
                    num_t   = global_add_pool(g_t * h_t, batch_t, size=B)  # (B, hidden)
                    den_t   = global_add_pool(g_t,       batch_t, size=B)  # (B, 1)
                    type_pools_avg[:, t, :] = num_t / (den_t + 1e-8)
                    type_pools_sum[:, t, :] = num_t
                    type_den[:, t] = den_t.squeeze(-1)
                    global_den += den_t
                    unique_b = torch.unique(batch_t)
                    type_mask[unique_b, t] = True

            # Context-conditioned importance weights with proper masking: (B, N_TYPES).
            # Each type's score sees its own average pool AND the per-graph density
            # fraction across all N_TYPES (same density-aware fix as in _pool above).
            density_frac = type_den / (global_den + 1e-8)  # (B, N_TYPES)
            importance_in = torch.cat([
                type_pools_avg,
                density_frac.unsqueeze(1).expand(-1, self.n_types, -1),
            ], dim=-1)  # (B, N_TYPES, hidden + N_TYPES)
            scores = self.type_importance(importance_in).squeeze(-1)  # (B, N_TYPES)
            scores = scores.masked_fill(~type_mask, -1e9)
            weights = F.softmax(scores, dim=-1)
            
            # We apply weights to the SUM pools and globally average, preserving relative proportions!
            pooled_hidden = (weights.unsqueeze(-1) * type_pools_sum).sum(dim=1) / (global_den + 1e-8)  # (B, hidden)
            scale_embs.append(self.out_proj(pooled_hidden))                  # (B, out_dim)

        embs = torch.stack(scale_embs, dim=1)             # (B, n_scales, out_dim)
        attn_logits = self.scale_attn(
            embs.view(B * self.n_scales, -1)
        ).view(B, self.n_scales, 1)
        attn = F.softmax(attn_logits, dim=1)
        return (attn * embs).sum(dim=1)                   # (B, out_dim)

    def get_attention_edges(self, g: Data, scale_id: int = 1,
                            top_n: int = 60) -> list:
        """Return top-N edges by layer-2 GATv2 attention weight for visualisation.

        Each entry: {src_lat, src_lon, tgt_lat, tgt_lon, weight, src_type, tgt_type}.
        Requires g.coords (added by build_v3_graph).
        """
        import numpy as _np
        dev = g.x.device
        cat_emb = self._get_cat_embs(g.node_types, g.type_labels)
        feat_in = torch.cat([g.x[:, :2], cat_emb, g.x[:, 2:]], dim=-1)
        x = self.node_proj(feat_in)
        x = x + self.node_type_emb(g.node_types)
        tok = self.scale_tokens(torch.full((x.size(0),), scale_id,
                                           dtype=torch.long, device=dev))
        x = x + tok
        e = self.edge_enc(g.edge_attr)

        out1, _ = self.gat1(x, g.edge_index, e, return_attention_weights=True)
        x = self.norm1(F.gelu(out1) + x)
        _, (ei2, alpha2) = self.gat2(x, g.edge_index, e, return_attention_weights=True)

        attn_w   = alpha2.mean(dim=-1).detach().cpu().numpy()  # (E,)
        src_nodes = ei2[0].cpu().numpy()
        tgt_nodes = ei2[1].cpu().numpy()

        if not hasattr(g, 'coords'):
            return []
        coords = g.coords.cpu().numpy()                        # (N, 2) lat/lon

        # Sort ALL edges descending; skip self-loops until we have top_n real edges
        all_sorted = _np.argsort(-attn_w)

        result = []
        for i in all_sorted:
            if len(result) >= top_n:
                break
            s_i, t_i = int(src_nodes[i]), int(tgt_nodes[i])
            if s_i == t_i:
                continue  # skip self-loops added internally by GATv2
            result.append({
                'src_lat':      float(coords[s_i, 0]),
                'src_lon':      float(coords[s_i, 1]),
                'tgt_lat':      float(coords[t_i, 0]),
                'tgt_lon':      float(coords[t_i, 1]),
                'weight':       float(attn_w[i]),
                'src_type':     int(g.node_types[s_i].item()),
                'tgt_type':     int(g.node_types[t_i].item()),
                'src_category': g.node_categories[s_i] if hasattr(g, 'node_categories') else '',
            })
        return result


# ─── GRAPH BUILDER ────────────────────────────────────────────────────────────

def build_v3_graph(query_lat, query_lon, radius, resources):
    """Build a single-scale v3 heterogeneous graph."""
    df, node_types_arr, type_labels_arr, type_les, type_vocabs, vectorizer, btree = resources

    q   = np.radians([[query_lat, query_lon]])
    idx = btree.query_radius(q, r=radius / 6_371_000.0)[0]
    if len(idx) < MIN_NODES:
        return None

    sel_df     = df.iloc[idx].reset_index(drop=True)
    sel_ntypes = node_types_arr[idx].astype(np.int64).copy(order='C')
    sel_tlbls  = type_labels_arr[idx].astype(np.int64).copy(order='C')
    coords     = sel_df[['lat', 'lon']].values.copy(order='C')

    coords_rel = StandardScaler().fit_transform(
        coords - np.array([query_lat, query_lon])).astype(np.float32).copy(order='C')
    sub_keys  = _other_tags_keys(sel_df['other_tags'].tolist())
    sel_other = vectorizer.transform(sub_keys).toarray().astype(np.float32).copy(order='C')

    node_feats = np.hstack([coords_rel, sel_other]).copy(order='C')   # (N, 34)

    k = min(K_NEIGHBORS, len(sel_df) - 1)
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)

    edges, eattr = [], []
    for i, neighs in enumerate(indices):
        for nb in neighs[1:]:
            d = _hav(coords[i, 1], coords[i, 0], coords[nb, 1], coords[nb, 0])
            b = _brng(coords[i, 1], coords[i, 0], coords[nb, 1], coords[nb, 0])
            same_cat = float(
                sel_tlbls[i] == sel_tlbls[nb] and sel_ntypes[i] == sel_ntypes[nb])
            src_nt = float(sel_ntypes[i])  / (N_NODE_TYPES - 1)
            tgt_nt = float(sel_ntypes[nb]) / (N_NODE_TYPES - 1)
            edges.append([i, nb])
            eattr.append([d / radius, math.sin(b), math.cos(b),
                          same_cat, src_nt, tgt_nt])

    if not edges:
        return None

    # Ensure all arrays are contiguous and have positive strides
    node_feats_c = node_feats.copy(order='C')
    coords_c = coords.copy(order='C')
    sel_ntypes_c = sel_ntypes.copy(order='C')
    sel_tlbls_c = sel_tlbls.copy(order='C')
    edges_c = np.array(edges).copy(order='C')
    eattr_c = np.array(eattr).copy(order='C')

    # Decode per-node OSM category key (e.g. "amenity") for edge coloring
    if type_les is not None:
        node_cats = []
        for i in range(len(sel_ntypes_c)):
            try:
                cat_str = type_les[int(sel_ntypes_c[i])].classes_[int(sel_tlbls_c[i])]
                node_cats.append(cat_str.split('=')[0])
            except Exception:
                node_cats.append('')
    else:
        node_cats = [''] * len(sel_ntypes_c)

    g = Data(
        x           = torch.tensor(node_feats_c),
        coords      = torch.tensor(coords_c,      dtype=torch.float32),
        node_types  = torch.tensor(sel_ntypes_c,  dtype=torch.long),
        type_labels = torch.tensor(sel_tlbls_c,   dtype=torch.long),
        edge_index  = torch.tensor(edges_c, dtype=torch.long).t().contiguous(),
        edge_attr   = torch.tensor(eattr_c, dtype=torch.float),
    )
    g.node_categories = node_cats
    return g


def build_multiscale_graphs_v3(query_lat, query_lon, scales=None,
                                external_resources=None):
    """Build graphs at all scales.  Returns list[Data] or None."""
    if scales is None:
        scales = SCALES
    res = get_resources(external_resources)
    graphs = []
    for r in scales:
        g = build_v3_graph(query_lat, query_lon, r, res)
        if g is None:
            return None
        graphs.append(g)
    return graphs


# ─── AUGMENTATION ─────────────────────────────────────────────────────────────

def augment_graphs_v3(graphs, type_vocabs, mask_prob=MASK_PROB):
    """Node masking: each type's mask token index = type_vocabs[t] (one past vocab end)."""
    out = []
    for g in graphs:
        N    = g.x.size(0)
        mask = torch.rand(N) < mask_prob
        x2   = g.x.clone()
        lbl2 = g.type_labels.clone()
        nt2  = g.node_types.clone()
        x2[mask] = 0.0
        for t in range(len(type_vocabs)):
            t_mask = mask & (g.node_types == t)
            lbl2[t_mask] = type_vocabs[t]   # mask token = vocab_size
        
        # FIX: Also mask the node_type so the model cannot trivially count unmasked types
        nt2[mask] = len(type_vocabs)
        
        out.append(Data(
            x           = x2,
            node_types  = nt2,
            type_labels = lbl2,
            edge_index  = g.edge_index,
            edge_attr   = g.edge_attr,
        ))
    return out


# ─── LOSS ─────────────────────────────────────────────────────────────────────

def info_nce_loss(z1, z2, temperature=TEMPERATURE):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B  = z1.size(0)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(B, device=z1.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2.0


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res    = get_resources()
    df, node_types_arr, type_labels_arr, type_les, type_vocabs, vectorizer, btree = res

    print(f'[v3] Device: {device}', flush=True)
    print(f'[v3] Type vocabs: {list(zip(NODE_TYPE_NAMES, type_vocabs))}', flush=True)

    min_lat, max_lat = float(df['lat'].min()), float(df['lat'].max())
    min_lon, max_lon = float(df['lon'].min()), float(df['lon'].max())

    print(f'[v3] Sampling {N_TRAIN} training locations …', flush=True)
    train_data, tries = [], 0
    while len(train_data) < N_TRAIN and tries < N_TRAIN * 20:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        graphs = build_multiscale_graphs_v3(lat, lon, external_resources=res)
        if graphs is not None:
            train_data.append((lat, lon, graphs))
        tries += 1
        if len(train_data) % 200 == 0 and len(train_data) > 0:
            print(f'  {len(train_data)}/{N_TRAIN} sampled ({tries} tries)', flush=True)

    print(f'[v3] {len(train_data)} training samples collected.', flush=True)
    if len(train_data) < 2 * BATCH_SIZE:
        raise RuntimeError('Not enough training samples — check data / radius.')

    model     = GeoSemanticsV3(type_vocabs=type_vocabs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    import sys
    resume = '--resume' in sys.argv

    best_loss  = float('inf')
    no_improve = 0
    log        = []
    start_epoch = 0

    if resume and os.path.exists(MODEL_PATH):
        print(f'[v3] Resuming training from {MODEL_PATH} ...', flush=True)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        log_path = os.path.join(BASE_DIR, 'geosemantics_v3_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as fh:
                old_log = json.load(fh)
                best_loss = old_log.get('best_loss', best_loss)
                log = old_log.get('loss', [])
                start_epoch = len(log)
            print(f'[v3] Resumed best loss: {best_loss:.4f} at epoch {start_epoch}', flush=True)

    print(f'[v3] Training for up to {EPOCHS} epochs …', flush=True)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        random.shuffle(train_data)
        ep_loss, n_batches = 0.0, 0

        for start in range(0, len(train_data) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = train_data[start:start + BATCH_SIZE]
            if len(batch) < 2:
                continue

            # Build augmented views and group by scale for batched forward
            scale_aug1 = [[] for _ in range(N_SCALES)]
            scale_aug2 = [[] for _ in range(N_SCALES)]
            for _lat, _lon, graphs in batch:
                for s, (g1, g2) in enumerate(zip(
                        augment_graphs_v3(graphs, type_vocabs),
                        augment_graphs_v3(graphs, type_vocabs))):
                    scale_aug1[s].append(g1.to(device))
                    scale_aug2[s].append(g2.to(device))

            # Single batched forward pass per view (was B sequential calls)
            z1   = model.forward_batch(scale_aug1)   # (B, out_dim)
            z2   = model.forward_batch(scale_aug2)
            loss = info_nce_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()
        avg = ep_loss / max(n_batches, 1)
        log.append(avg)
        print(f'[v3] Epoch {epoch + 1:3d}/{EPOCHS} | loss={avg:.4f}'
              f' | lr={scheduler.get_last_lr()[0]:.1e}', flush=True)

        if avg < best_loss - 1e-5:
            best_loss  = avg
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            with open(META_PATH, 'w') as fh:
                json.dump({
                    'type_vocabs': type_vocabs,
                    'out_dim':     OUT_DIM,
                    'hidden':      HIDDEN,
                    'n_heads':     N_HEADS,
                    'edge_dim':    EDGE_DIM,
                    'n_scales':    N_SCALES,
                }, fh, indent=2)
            print(f'         -> saved (best loss={best_loss:.4f})', flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'[v3] Early stopping at epoch {epoch + 1}.', flush=True)
                break

    log_path = os.path.join(BASE_DIR, 'geosemantics_v3_log.json')
    json.dump({'loss': log, 'best_loss': best_loss,
               'n_train': len(train_data), 'epochs_run': len(log)},
              open(log_path, 'w'), indent=2)
    print(f'[v3] Done.  Best loss: {best_loss:.4f}', flush=True)
    print(f'[v3] Model: {MODEL_PATH}', flush=True)
    print(f'[v3] Log:   {log_path}', flush=True)


if __name__ == '__main__':
    train()
