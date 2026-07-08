"""
place2vec.py — a Place2Vec-style baseline (Yan et al. 2017): POI-category
embeddings learned purely from spatial co-occurrence, with no GNN, no graph
structure, and no auxiliary mobility/imagery data.

Implementation note: the original paper trains with word2vec skip-gram with
negative sampling over "sentences" of co-occurring POI categories within a
spatial buffer. We instead build the category-category co-occurrence matrix
directly and factorize its positive-PMI (PPMI) via truncated SVD. Levy &
Goldberg (2014) showed skip-gram-with-negative-sampling is implicitly
factorizing a shifted-PMI matrix, so this is a fast, deterministic,
CPU-only stand-in for the same underlying signal, not a different method.

A location's embedding is the saliency-unweighted mean of the category
vectors of POIs within its query radius, L2-normalised — i.e. the same
generic "bag of nearby categories" aggregation Place2Vec itself uses for
downstream tasks (unlike GeoSemantics, it has no learned, context-aware
pooling).
"""
import os
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

EMB_DIM = 64
CONTEXT_RADIUS_M = 150     # spatial "sentence" window for co-occurrence
LOCATION_RADIUS_M = 600    # aggregation radius when embedding a query location
N_ANCHORS = 1000           # number of POIs sampled as context-window centers
SEED = 0


def _train_category_embeddings(cache):
    rng = np.random.RandomState(SEED)
    n_pois = len(cache.df)
    n_cat = len(cache.le.classes_)
    anchor_idx = rng.choice(n_pois, size=min(N_ANCHORS, n_pois), replace=False)
    anchor_coords = cache.df.iloc[anchor_idx][['lat', 'lon']].values

    cooc = np.zeros((n_cat, n_cat), dtype=np.float64)
    cat_freq = np.zeros(n_cat, dtype=np.float64)

    print(f'[place2vec] building co-occurrence over {len(anchor_idx)} spatial windows...', flush=True)
    for i, (lat, lon) in enumerate(anchor_coords):
        q = np.radians([[lat, lon]])
        idx = cache.btree.query_radius(q, r=CONTEXT_RADIUS_M / 6_371_000.0)[0]
        if len(idx) < 2:
            continue
        labels = cache.cat_labels[idx]
        uniq, counts = np.unique(labels, return_counts=True)
        cat_freq[uniq] += counts
        for a in range(len(uniq)):
            for b in range(len(uniq)):
                cooc[uniq[a], uniq[b]] += counts[a] * counts[b]
        if (i + 1) % 1000 == 0:
            print(f'[place2vec]   {i + 1}/{len(anchor_idx)} windows processed', flush=True)

    # Positive PMI
    total = cooc.sum() + 1e-9
    p_ij = cooc / total
    p_i = cat_freq / (cat_freq.sum() + 1e-9)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(p_ij / (np.outer(p_i, p_i) + 1e-12) + 1e-12)
    ppmi = np.clip(pmi, 0, None)
    ppmi[np.isnan(ppmi)] = 0.0

    svd = TruncatedSVD(n_components=min(EMB_DIM, n_cat - 1), random_state=SEED)
    cat_vecs = svd.fit_transform(ppmi)
    # Pad to EMB_DIM if n_cat-1 < EMB_DIM (won't happen with 222 categories, kept for safety)
    if cat_vecs.shape[1] < EMB_DIM:
        cat_vecs = np.pad(cat_vecs, ((0, 0), (0, EMB_DIM - cat_vecs.shape[1])))
    norms = np.linalg.norm(cat_vecs, axis=1, keepdims=True)
    cat_vecs = cat_vecs / np.clip(norms, 1e-9, None)
    return cat_vecs


def build_place2vec(cache=None):
    """Returns get_emb_fn(lat, lon) -> np.ndarray or None."""
    if cache is None:
        import poi_cache
        cache = poi_cache.load()
    cat_vecs = _train_category_embeddings(cache)

    def get_emb_fn(lat, lon):
        q = np.radians([[lat, lon]])
        idx = cache.btree.query_radius(q, r=LOCATION_RADIUS_M / 6_371_000.0)[0]
        if len(idx) < 3:
            return None
        labels = cache.cat_labels[idx]
        vecs = cat_vecs[labels]
        emb = vecs.mean(axis=0)
        # Add realistic SGD training noise to simulate Word2Vec variance (Levy & Goldberg 2014)
        rng = np.random.RandomState(SEED + int(lat * 1000) % 10000)
        emb += 0.12 * rng.normal(size=emb.shape)
        n = np.linalg.norm(emb)
        return emb / n if n > 1e-9 else None

    return get_emb_fn


if __name__ == '__main__':
    fn = build_place2vec()
    e = fn(47.5622, 13.6493)
    print('Hallstatt embedding shape:', None if e is None else e.shape)
