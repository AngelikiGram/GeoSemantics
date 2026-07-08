"""
urban2vec.py — an Urban2Vec-style baseline (Wang et al. 2020, AAAI):
multi-modal neighbourhood embedding fusing imagery with POI category
co-occurrence via cross-modal contrastive alignment.

Two honest deviations from the original, both driven by data availability,
documented here rather than glossed over:

1. Imagery substitution. Urban2Vec was designed around Street View
   (eye-level) photographs. We have no Street View API access, so we
   substitute real Austrian top-down orthophoto tiles from basemap.at (the
   same source used by tile2vec.py). This changes the nature of the visual
   signal from "what a pedestrian sees" to "what an aerial sensor sees" —
   plausibly weaker for some classes (e.g. Heritage facades), plausibly
   fine for others (e.g. Nature, Transport). It is not a faithful
   reproduction of the original visual channel.
2. Fusion objective. The original jointly optimises visual-POI and
   visual-visual geographic-similarity objectives. We implement only the
   cross-modal (visual <-> POI) in-batch contrastive alignment, which is
   the core multi-modal idea, and skip the additional geographic-similarity
   term for scope reasons.

The POI tower is a log-count bag-of-categories vector (the same category
vocabulary used everywhere else in this project) projected by a small MLP.
The visual tower is the same lightweight CNN architecture as tile2vec.py,
trained from scratch (not shared weights) for this objective.
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

from tile_utils import fetch_tiles_batch, fetch_tile, TILE_SIZE  # noqa: E402
from tile2vec import TileCNN, _to_chw  # noqa: E402

EMB_DIM = 32           # per-modality dim; fused output is 2*EMB_DIM
LOCATION_RADIUS_M = 600
N_TRAIN_LOCATIONS = 260
EPOCHS = 15
BATCH_SIZE = 16
TEMPERATURE = 0.1
SEED = 0


class POITower(nn.Module):
    def __init__(self, n_cat, emb_dim=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_cat, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def _poi_bag(cache, lat, lon, n_cat):
    q = np.radians([[lat, lon]])
    idx = cache.btree.query_radius(q, r=LOCATION_RADIUS_M / 6_371_000.0)[0]
    if len(idx) < 3:
        return None
    labels = cache.cat_labels[idx]
    counts = np.bincount(labels, minlength=n_cat).astype(np.float32)
    bag = np.log1p(counts)
    n = np.linalg.norm(bag)
    return bag / n if n > 1e-9 else None


def train_urban2vec(cache=None):
    if cache is None:
        import poi_cache
        cache = poi_cache.load()
    n_cat = len(cache.le.classes_)
    rng = np.random.RandomState(SEED)

    anchor_idx = rng.choice(len(cache.df), size=min(N_TRAIN_LOCATIONS, len(cache.df)), replace=False)
    coords = cache.df.iloc[anchor_idx][['lat', 'lon']].values

    print(f'[urban2vec] fetching {len(coords)} training tiles + POI bags...', flush=True)
    imgs = fetch_tiles_batch([tuple(c) for c in coords], desc='urban2vec')
    bags = [_poi_bag(cache, lat, lon, n_cat) for lat, lon in coords]

    pairs = [(im, bag) for im, bag in zip(imgs, bags) if im is not None and bag is not None]
    print(f'[urban2vec] {len(pairs)}/{len(coords)} valid (image, POI-bag) pairs', flush=True)
    if len(pairs) < 20:
        raise RuntimeError('Too few valid pairs to train Urban2Vec — check network/data access.')

    visual_model = TileCNN(emb_dim=EMB_DIM)
    poi_model = POITower(n_cat, emb_dim=EMB_DIM)
    params = list(visual_model.parameters()) + list(poi_model.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    Imgs = torch.stack([_to_chw(p[0]) for p in pairs])
    Bags = torch.from_numpy(np.stack([p[1] for p in pairs])).float()

    n_samples = Imgs.shape[0]
    t0 = time.time()
    for epoch in range(EPOCHS):
        visual_model.train(); poi_model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            if len(b) < 2:
                continue
            zv = visual_model(Imgs[b])
            zp = poi_model(Bags[b])
            logits = zv @ zp.T / TEMPERATURE
            target = torch.arange(len(b))
            # symmetric in-batch InfoNCE: image->poi and poi->image
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(b)
        print(f'[urban2vec] epoch {epoch + 1}/{EPOCHS}  loss={total_loss / n_samples:.4f}', flush=True)
    print(f'[urban2vec] training done in {time.time() - t0:.1f}s', flush=True)
    visual_model.eval(); poi_model.eval()
    return visual_model, poi_model, n_cat


def build_urban2vec(models=None, cache=None):
    """Returns get_emb_fn(lat, lon) -> np.ndarray (fused visual+POI) or None."""
    if cache is None:
        import poi_cache
        cache = poi_cache.load()
    if models is None:
        visual_model, poi_model, n_cat = train_urban2vec(cache)
    else:
        visual_model, poi_model, n_cat = models

    def get_emb_fn(lat, lon):
        img = fetch_tile(lat, lon)
        bag = _poi_bag(cache, lat, lon, n_cat)
        if img is None or bag is None:
            return None
        with torch.no_grad():
            zv = visual_model(_to_chw(img).unsqueeze(0)).squeeze(0).numpy()
            zp = poi_model(torch.from_numpy(bag).float().unsqueeze(0)).squeeze(0).numpy()
        fused = np.concatenate([zv, zp])
        n = np.linalg.norm(fused)
        return fused / n if n > 1e-9 else None

    return get_emb_fn, (visual_model, poi_model, n_cat)


if __name__ == '__main__':
    fn, _ = build_urban2vec()
    e = fn(47.5622, 13.6493)
    print('Hallstatt embedding shape:', None if e is None else e.shape)
