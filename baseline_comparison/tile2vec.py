"""
tile2vec.py — a Tile2Vec-style baseline (Jean et al. 2019, AAAI): a CNN
encoder trained purely from satellite/aerial imagery via a spatial
triplet-contrastive objective (nearby tiles pulled together, distant tiles
pushed apart) — no OSM tags, no graph structure at all.

Imagery: real Austrian orthophoto tiles from basemap.at (see tile_utils.py).
This is a faithful match to Tile2Vec's original design (it was designed for
satellite tiles), unlike the Urban2Vec baseline in this folder.

CNN is intentionally small (4 conv layers, ~64-d output) to stay trainable
on CPU within minutes, consistent with this project's CPU-only design point;
this is a lighter encoder than the original paper's, traded for tractability.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tile_utils import fetch_tiles_batch, TILE_SIZE  # noqa: E402

EMB_DIM = 64
LAT_RANGE = (46.30, 49.05)
LON_RANGE = (9.40, 17.20)
N_TRAIN_ANCHORS = 260
NEIGHBOR_RADIUS_DEG = 0.003   # ~300m, "near" positive
DISTANT_MIN_DEG = 0.3         # ~25km+, "far" negative
EPOCHS = 12
BATCH_SIZE = 16
MARGIN = 0.3
SEED = 0


class TileCNN(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        z = self.fc(h)
        return F.normalize(z, dim=-1)


def _to_chw(img_hwc):
    return torch.from_numpy(img_hwc).permute(2, 0, 1).float()


def _sample_training_triplets(rng, n, cache):
    """Anchors and distant points are sampled from real POI coordinates (so
    they are guaranteed to fall inside Austria's actual basemap.at coverage,
    not just inside the rectangular lat/lon bounding box, which also covers
    parts of neighbouring countries — random points there returned empty
    tiles and inflated the failure rate in an earlier run of this script).
    Only the small neighbour offset is synthetic, since ~300m from a real
    POI is overwhelmingly still inside Austria."""
    n_pois = len(cache.df)
    anchor_idx = rng.choice(n_pois, size=n, replace=False)
    distant_idx = rng.choice(n_pois, size=n, replace=False)
    anchor_coords = cache.df.iloc[anchor_idx][['lat', 'lon']].values
    distant_coords = cache.df.iloc[distant_idx][['lat', 'lon']].values

    anchors, neighbors, distants = [], [], []
    for (lat, lon), (flat, flon) in zip(anchor_coords, distant_coords):
        dlat, dlon = rng.uniform(-NEIGHBOR_RADIUS_DEG, NEIGHBOR_RADIUS_DEG, size=2)
        if abs(flat - lat) + abs(flon - lon) < DISTANT_MIN_DEG:
            flat, flon = flat + 1.0, flon + 1.0  # rare same-area collision; push out of range
        anchors.append((lat, lon))
        neighbors.append((lat + dlat, lon + dlon))
        distants.append((flat, flon))
    return anchors, neighbors, distants


def train_tile2vec_encoder(cache=None):
    if cache is None:
        import poi_cache
        cache = poi_cache.load()
    rng = np.random.RandomState(SEED)
    anchors, neighbors, distants = _sample_training_triplets(rng, N_TRAIN_ANCHORS, cache)

    print(f'[tile2vec] fetching {3 * N_TRAIN_ANCHORS} training tiles...', flush=True)
    a_imgs = fetch_tiles_batch(anchors, desc='anchor')
    n_imgs = fetch_tiles_batch(neighbors, desc='neighbor')
    d_imgs = fetch_tiles_batch(distants, desc='distant')

    triplets = [(a, n, d) for a, n, d in zip(a_imgs, n_imgs, d_imgs)
                if a is not None and n is not None and d is not None]
    print(f'[tile2vec] {len(triplets)}/{N_TRAIN_ANCHORS} valid triplets after tile fetch', flush=True)
    if len(triplets) < 20:
        raise RuntimeError('Too few valid tiles fetched to train Tile2Vec — check network access.')

    model = TileCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    triplet_loss = nn.TripletMarginLoss(margin=MARGIN)

    A = torch.stack([_to_chw(t[0]) for t in triplets])
    N = torch.stack([_to_chw(t[1]) for t in triplets])
    D = torch.stack([_to_chw(t[2]) for t in triplets])

    n_samples = A.shape[0]
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            za, zn, zd = model(A[b]), model(N[b]), model(D[b])
            loss = triplet_loss(za, zn, zd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(b)
        print(f'[tile2vec] epoch {epoch + 1}/{EPOCHS}  loss={total_loss / n_samples:.4f}', flush=True)
    print(f'[tile2vec] training done in {time.time() - t0:.1f}s', flush=True)
    model.eval()
    return model


def build_tile2vec(cache=None, model=None):
    """Returns get_emb_fn(lat, lon) -> np.ndarray or None."""
    if model is None:
        model = train_tile2vec_encoder(cache)

    def get_emb_fn(lat, lon):
        from tile_utils import fetch_tile
        img = fetch_tile(lat, lon)
        if img is None:
            return None
        with torch.no_grad():
            z = model(_to_chw(img).unsqueeze(0))
        return z.squeeze(0).numpy()

    return get_emb_fn, model


if __name__ == '__main__':
    fn, _ = build_tile2vec()
    e = fn(47.5622, 13.6493)
    print('Hallstatt embedding shape:', None if e is None else e.shape)
