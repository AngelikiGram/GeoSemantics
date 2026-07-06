"""
GeoSemantics v2 – Direction-Aware, Multi-Scale Contrastive Location Embedding
=============================================================================

Architecture (paper contributions):

  1. Learnable POI-type embeddings
       Replaces one-hot encoding.  Types are embedded into a shared space
       where spatial co-occurrence drives semantic similarity—café and
       restaurant end up close without any linguistic supervision.

  2. Sinusoidal distance encoding + compass bearing in edge features
       Each edge carries [sin/cos distance harmonics, sin(θ), cos(θ),
       same-type flag].  GATv2Conv uses these features in its attention
       coefficient computation, making message weights direction-aware.

  3. Multi-scale graphs  (200 m / 700 m / 2 000 m)
       Scale tokens injected into every node distinguish micro / meso /
       macro context.  A shared GATv2 stack processes all three; a learned
       cross-scale attention then fuses the three graph-level embeddings.

  4. InfoNCE contrastive training with node-masking augmentation
       Two augmented views of the same location (random POI masking) form
       positive pairs; all other locations in the batch are negatives.
       Temperature τ = 0.07 following SimCLR / MoCo best practice.

Run training:
    python geosemantics_v2.py

Import for inference:
    from geosemantics_v2 import GeoSemanticsV2, build_multiscale_graphs, SCALES

Reference:
    GATv2: Brody et al., ICLR 2022
    InfoNCE: Oord et al., arXiv 2018
    GraphCL: You et al., NeurIPS 2020
    SimCLR:  Chen et al., ICML 2020
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
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SCALES      = [200, 700, 2000]      # metres: micro / meso / macro
N_SCALES    = len(SCALES)
K_NEIGHBORS = 8

# Model hyper-parameters
POI_DIM   = 64      # learnable POI-type embedding size
OTHER_DIM = 32      # CountVectorizer other-tags features
HIDDEN    = 128     # GNN hidden dimension
OUT_DIM   = 64      # final embedding dimension
N_HEADS   = 4       # GAT attention heads
EDGE_DIM  = 16      # edge MLP output dimension

# Training hyper-parameters
N_TRAIN      = 1000   # training locations to sample
BATCH_SIZE   = 16
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE  = 0.07   # InfoNCE temperature
MASK_PROB    = 0.20   # fraction of nodes to mask per augmentation
PATIENCE     = 25     # early-stopping patience

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE_DIR, '_poi_cache')
MODEL_PATH = os.path.join(BASE_DIR, 'geosemantics_v2.pt')

_KEY_RE = re.compile(r'"([^"]+)"=>"')


# ─── SHARED RESOURCE LOADER ───────────────────────────────────────────────────

_resources_cache = None


def _load_resources():
    """Load pre-built cache files (built by inference.py on first run)."""
    df         = pd.read_parquet(os.path.join(CACHE_DIR, 'df.parquet'))
    cat_labels = np.load(os.path.join(CACHE_DIR, 'cat_labels.npy'))
    with open(os.path.join(CACHE_DIR, 'le.pkl'),         'rb') as f:
        le         = pickle.load(f)
    with open(os.path.join(CACHE_DIR, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(CACHE_DIR, 'btree.pkl'),      'rb') as f:
        btree      = pickle.load(f)
    return df, cat_labels, le, vectorizer, btree


def get_resources(external=None):
    """Return (df, cat_labels, le, vectorizer, btree).

    Pass `external` to reuse already-loaded resources from inference.py
    and avoid loading the 2.4 M POI dataset a second time.
    """
    global _resources_cache
    if external is not None:
        return external
    if _resources_cache is None:
        print('[v2] Loading POI cache …', flush=True)
        _resources_cache = _load_resources()
        print(f'[v2] {len(_resources_cache[0]):,} POIs loaded.', flush=True)
    return _resources_cache


# ─── GEOMETRY HELPERS ─────────────────────────────────────────────────────────

def _other_tags_keys(arr):
    return [' '.join(_KEY_RE.findall(t)) if t else '' for t in arr]


def _hav(lon1, lat1, lon2, lat2):
    R = 6_371_000.0
    lo1, la1, lo2, la2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = math.sin((la2 - la1) / 2) ** 2 + \
        math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return R * 2 * math.asin(math.sqrt(max(a, 0.0)))


def _brng(lon1, lat1, lon2, lat2):
    lo1, la1, lo2, la2 = map(math.radians, [lon1, lat1, lon2, lat2])
    x = math.sin(lo2 - lo1) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return math.atan2(x, y)


# ─── MODEL COMPONENTS ─────────────────────────────────────────────────────────

class SinusoidalDistEnc(nn.Module):
    """
    Encode normalised distance d ∈ [0, 1] with multi-frequency sinusoids.

    Distances are normalised by the scale radius so the same encoding
    is meaningful across all three spatial scales.
    """
    def __init__(self, n_freqs: int = 4):
        super().__init__()
        self.register_buffer('freqs', torch.pow(2.0, torch.arange(n_freqs).float()))

    def forward(self, d_norm: torch.Tensor) -> torch.Tensor:
        # d_norm: (E,) in [0, 1]
        d      = d_norm.unsqueeze(-1)                    # (E, 1)
        angles = d * self.freqs * math.pi                # (E, n_freqs)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (E, 2·n_freqs)


class EdgeEncoder(nn.Module):
    """
    Project raw edge attributes → fixed-size edge embedding used in GATv2Conv.

    Input (E, 4):  [d_norm, sin(bearing), cos(bearing), same_type]
    Output (E, edge_out):  learned spatial relationship encoding
    """
    def __init__(self, edge_out: int = EDGE_DIM, n_freqs: int = 4):
        super().__init__()
        self.dist_enc = SinusoidalDistEnc(n_freqs)
        raw = 2 * n_freqs + 3        # 8 dist harmonics + sinθ + cosθ + type_sim
        self.mlp = nn.Sequential(
            nn.Linear(raw, edge_out * 2),
            nn.LayerNorm(edge_out * 2),
            nn.GELU(),
            nn.Linear(edge_out * 2, edge_out),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        dist_feat = self.dist_enc(edge_attr[:, 0])
        return self.mlp(torch.cat([dist_feat, edge_attr[:, 1:]], dim=-1))


class GeoSemanticsV2(nn.Module):
    """
    Direction-Aware, Multi-Scale GNN for Semantic Location Embedding.

    Input: list of n_scales Data objects, each with:
        g.x          – (N, 2+32) float   [coords_rel | other_tags_bow]
        g.poi_labels – (N,)      long    POI category indices
        g.edge_index – (2, E)            KNN connectivity
        g.edge_attr  – (E, 4)    float   [d_norm, sinθ, cosθ, same_type]

    Output:
        emb      – (out_dim,)   location embedding
        attn     – (n_scales,)  per-scale attention weights (interpretable)
    """

    def __init__(self, n_poi_types: int = 222,
                 poi_dim:   int = POI_DIM,
                 other_dim: int = OTHER_DIM,
                 hidden:    int = HIDDEN,
                 out_dim:   int = OUT_DIM,
                 n_heads:   int = N_HEADS,
                 edge_dim:  int = EDGE_DIM,
                 n_scales:  int = N_SCALES):
        super().__init__()
        self.n_scales = n_scales

        # Contribution 1: jointly-learned POI type embeddings
        # +1 = mask token used in contrastive augmentation
        self.poi_emb = nn.Embedding(n_poi_types + 1, poi_dim)
        nn.init.normal_(self.poi_emb.weight, std=0.02)

        # Contribution 3: scale tokens differentiate micro / meso / macro context
        self.scale_tokens = nn.Embedding(n_scales, hidden)

        # Node input projection: [coords(2) + poi_emb(poi_dim) + other(other_dim)] → hidden
        node_in = 2 + poi_dim + other_dim
        self.node_proj = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # Contribution 2: direction-aware edge encoder (shared across scales)
        self.edge_enc = EdgeEncoder(edge_out=edge_dim)

        # Shared GATv2 layers with residual connections
        self.gat1  = GATv2Conv(hidden, hidden // n_heads, heads=n_heads,
                                edge_dim=edge_dim, concat=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden)

        self.gat2  = GATv2Conv(hidden, hidden // n_heads, heads=n_heads,
                                edge_dim=edge_dim, concat=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden)

        # Attention pooling gate (graph-level readout)
        self.pool_gate = nn.Linear(hidden, 1)

        # Contribution 3: cross-scale attention aggregation
        self.out_proj   = nn.Linear(hidden, out_dim)
        self.scale_attn = nn.Sequential(
            nn.Linear(out_dim, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def _encode_scale(self, g: Data, scale_id: int) -> torch.Tensor:
        dev = g.x.device

        # Compose node features
        poi_feat = self.poi_emb(g.poi_labels)
        x = self.node_proj(torch.cat([g.x[:, :2], poi_feat, g.x[:, 2:]], dim=-1))

        # Inject scale context into every node
        tok = self.scale_tokens(torch.full((x.size(0),), scale_id,
                                           dtype=torch.long, device=dev))
        x = x + tok

        e  = self.edge_enc(g.edge_attr)

        # Two GATv2 layers with residual
        x = self.norm1(F.gelu(self.gat1(x, g.edge_index, e)) + x)
        x = self.norm2(F.gelu(self.gat2(x, g.edge_index, e)) + x)
        return x

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.pool_gate(x))        # (N, 1)
        return (gates * x).sum(0) / (gates.sum() + 1e-8)

    def get_attention_edges(self, g: Data, scale_id: int = 1, top_n: int = 60) -> list:
        """Return top-N GATv2 layer-2 attention edges for visualisation."""
        import numpy as _np
        dev = g.x.device
        poi_feat = self.poi_emb(g.poi_labels)
        x = self.node_proj(torch.cat([g.x[:, :2], poi_feat, g.x[:, 2:]], dim=-1))
        tok = self.scale_tokens(torch.full((x.size(0),), scale_id, dtype=torch.long, device=dev))
        x = x + tok
        e = self.edge_enc(g.edge_attr)
        out1, _ = self.gat1(x, g.edge_index, e, return_attention_weights=True)
        x = self.norm1(F.gelu(out1) + x)
        _, (ei2, alpha2) = self.gat2(x, g.edge_index, e, return_attention_weights=True)

        attn_w    = alpha2.mean(dim=-1).detach().cpu().numpy()
        src_nodes = ei2[0].cpu().numpy()
        tgt_nodes = ei2[1].cpu().numpy()

        if not hasattr(g, 'coords'):
            return []
        coords = g.coords.cpu().numpy()

        # Sort ALL edges descending; skip self-loops until we have top_n real edges
        all_sorted = _np.argsort(-attn_w)

        result = []
        for i in all_sorted:
            if len(result) >= top_n:
                break
            s_i, t_i = int(src_nodes[i]), int(tgt_nodes[i])
            if s_i == t_i:
                continue  # skip self-loops added internally by GATv2
            src_cat = ''
            if hasattr(g, 'label_classes'):
                try:
                    src_cat = str(g.label_classes[g.poi_labels[s_i].item()]).split('=')[0]
                except Exception:
                    pass
            result.append({
                'src_lat':      float(coords[s_i, 0]),
                'src_lon':      float(coords[s_i, 1]),
                'tgt_lat':      float(coords[t_i, 0]),
                'tgt_lon':      float(coords[t_i, 1]),
                'weight':       float(attn_w[i]),
                'src_type':     0,
                'tgt_type':     0,
                'src_category': src_cat,
            })
        return result

    def forward(self, graphs):
        embs = []
        for s, g in enumerate(graphs):
            h   = self._encode_scale(g, s)
            emb = self.out_proj(self._pool(h))
            embs.append(emb)

        embs = torch.stack(embs, dim=0)                           # (n_scales, out_dim)
        attn = F.softmax(self.scale_attn(embs), dim=0)            # (n_scales, 1)
        final = (attn * embs).sum(dim=0)                          # (out_dim,)
        return final, attn.squeeze(-1)


# ─── GRAPH BUILDERS ───────────────────────────────────────────────────────────

def build_v2_graph(query_lat, query_lon, radius, resources):
    """Build a single-scale v2 graph."""
    df, cat_labels, le, vectorizer, btree = resources

    q   = np.radians([[query_lat, query_lon]])
    idx = btree.query_radius(q, r=radius / 6_371_000.0)[0]
    if len(idx) < 3:
        return None

    sel_df     = df.iloc[idx].reset_index(drop=True)
    sel_labels = cat_labels[idx]
    coords     = sel_df[['lat', 'lon']].values

    coords_rel = StandardScaler().fit_transform(
        coords - np.array([query_lat, query_lon])).astype(np.float32)
    sub_keys  = _other_tags_keys(sel_df['other_tags'].tolist())
    sel_other = vectorizer.transform(sub_keys).toarray().astype(np.float32)

    # x = [coords_rel (2) | other_tags (32)]  – poi embedding handled in model
    node_feats = np.hstack([coords_rel, sel_other]).copy()

    k = min(K_NEIGHBORS, len(sel_df) - 1)
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)

    edges, eattr = [], []
    for i, neighs in enumerate(indices):
        for n in neighs[1:]:
            d = _hav(coords[i, 1], coords[i, 0], coords[n, 1], coords[n, 0])
            b = _brng(coords[i, 1], coords[i, 0], coords[n, 1], coords[n, 0])
            d_norm = float(d / radius)
            edges.append([i, n])
            eattr.append([d_norm, math.sin(b), math.cos(b),
                          float(sel_labels[i] == sel_labels[n])])

    if not edges:
        return None

    g = Data(
        x          = torch.tensor(node_feats),
        poi_labels = torch.tensor(sel_labels.astype(np.int64).copy(), dtype=torch.long),
        edge_index = torch.tensor(np.array(edges), dtype=torch.long).t().contiguous(),
        edge_attr  = torch.tensor(np.array(eattr, dtype=np.float32)),
        coords     = torch.tensor(coords.copy(), dtype=torch.float32),
    )
    g.label_classes = le.classes_
    return g


def build_multiscale_graphs(query_lat, query_lon, scales=None, external_resources=None):
    """
    Build graphs at all spatial scales.

    Pass `external_resources` = (df, cat_labels, le, vectorizer, btree)
    to reuse already-loaded data from inference.py.

    Returns list[Data] or None if any scale has fewer than 3 POIs.
    """
    if scales is None:
        scales = SCALES
    res = get_resources(external_resources)

    graphs = []
    for r in scales:
        g = build_v2_graph(query_lat, query_lon, r, res)
        if g is None:
            return None
        graphs.append(g)
    return graphs


# ─── AUGMENTATION ─────────────────────────────────────────────────────────────

def augment_graphs(graphs, n_poi_types, mask_prob=MASK_PROB):
    """
    Node-masking augmentation for contrastive learning.

    A random fraction of nodes have their coordinate features zeroed and
    their POI label replaced with a mask token (index n_poi_types).
    This forces the model to infer a node's role from its neighbourhood
    rather than its own identity—analogous to BERT-style masked modelling.
    """
    out = []
    for g in graphs:
        N    = g.x.size(0)
        mask = torch.rand(N) < mask_prob

        x2   = g.x.clone()
        lbl2 = g.poi_labels.clone()
        x2[mask]   = 0.0
        lbl2[mask] = n_poi_types    # mask token index

        out.append(Data(
            x          = x2,
            poi_labels = lbl2,
            edge_index = g.edge_index,
            edge_attr  = g.edge_attr,
        ))
    return out


# ─── CONTRASTIVE LOSS ─────────────────────────────────────────────────────────

def info_nce_loss(z1, z2, temperature=TEMPERATURE):
    """
    Symmetric InfoNCE / NT-Xent loss.

    z1, z2 : (B, D) – embeddings of two augmented views of B locations.
    Positive pairs are on the diagonal; all B²-B off-diagonal entries are
    treated as negatives (hard in-batch negatives).
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B  = z1.size(0)

    logits = torch.matmul(z1, z2.T) / temperature    # (B, B)
    labels = torch.arange(B, device=z1.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2.0


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res    = get_resources()
    df, cat_labels, le, vectorizer, btree = res
    n_poi_types = len(le.classes_)

    print(f'[v2] Device: {device} | n_poi_types: {n_poi_types}', flush=True)

    # ── Sample training locations ─────────────────────────────────────────────
    min_lat, max_lat = float(df['lat'].min()), float(df['lat'].max())
    min_lon, max_lon = float(df['lon'].min()), float(df['lon'].max())

    print(f'[v2] Sampling {N_TRAIN} training locations …', flush=True)
    train_data, tries = [], 0
    while len(train_data) < N_TRAIN and tries < N_TRAIN * 5:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        graphs = build_multiscale_graphs(lat, lon, external_resources=res)
        if graphs is not None:
            train_data.append((lat, lon, graphs))
        tries += 1
        if len(train_data) % 200 == 0 and len(train_data) > 0:
            print(f'  {len(train_data)}/{N_TRAIN} sampled', flush=True)

    print(f'[v2] {len(train_data)} training samples collected.', flush=True)
    if len(train_data) < 2 * BATCH_SIZE:
        raise RuntimeError('Not enough training samples – check RADIUS / N_TRAIN.')

    # ── Model, optimiser, scheduler ──────────────────────────────────────────
    model     = GeoSemanticsV2(n_poi_types=n_poi_types).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_loss  = float('inf')
    no_improve = 0
    log        = []

    print(f'[v2] Training for up to {EPOCHS} epochs …', flush=True)

    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(train_data)
        ep_loss, n_batches = 0.0, 0

        for start in range(0, len(train_data) - BATCH_SIZE + 1, BATCH_SIZE):
            batch     = train_data[start:start + BATCH_SIZE]
            z1_list, z2_list = [], []

            for _lat, _lon, graphs in batch:
                # Two independently-masked views of the same location
                aug1 = [g.to(device) for g in augment_graphs(graphs, n_poi_types)]
                aug2 = [g.to(device) for g in augment_graphs(graphs, n_poi_types)]

                e1, _ = model(aug1)
                e2, _ = model(aug2)
                z1_list.append(e1)
                z2_list.append(e2)

            if len(z1_list) < 2:
                continue

            z1   = torch.stack(z1_list)   # (B, out_dim)
            z2   = torch.stack(z2_list)
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

        print(f'[v2] Epoch {epoch + 1:3d}/{EPOCHS} | loss={avg:.4f}'
              f' | lr={scheduler.get_last_lr()[0]:.1e}', flush=True)

        if avg < best_loss - 1e-5:
            best_loss  = avg
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'         → saved (best loss={best_loss:.4f})', flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'[v2] Early stopping at epoch {epoch + 1}.', flush=True)
                break

    # Save training log for analysis / paper plots
    log_path = os.path.join(BASE_DIR, 'geosemantics_v2_log.json')
    json.dump({'loss': log, 'best_loss': best_loss,
               'n_train': len(train_data), 'epochs_run': len(log)},
              open(log_path, 'w'), indent=2)
    print(f'[v2] Done.  Best loss: {best_loss:.4f}', flush=True)
    print(f'[v2] Model: {MODEL_PATH}', flush=True)
    print(f'[v2] Log:   {log_path}', flush=True)


if __name__ == '__main__':
    train()
