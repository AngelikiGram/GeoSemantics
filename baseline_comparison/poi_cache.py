"""
poi_cache.py — loads just the raw POI cache (dataframe, category labels,
label encoder, BallTree) that place2vec.py and urban2vec.py need, without
importing inference.py.

Why this exists: inference.py eagerly loads the full torch_geometric V2/V3
GATv2 models at import time. Loading those in the same process as the
plain-torch CNNs trained in tile2vec.py/urban2vec.py segfaults reliably on
this machine (reproduced multiple times — looks like a native extension
conflict between two independently-initialised torch contexts, not a bug in
our model code). Since this comparison only ever needs raw POI data (not
the GNN models — the V2/V3 numbers are read from the existing
validation_results/metrics.json instead of recomputed), we load that data
directly from the same on-disk cache files inference.py itself reads from.
"""
import os
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, '_poi_cache')


class PoiCache:
    __slots__ = ('df', 'cat_labels', 'le', 'btree')

    def __init__(self, df, cat_labels, le, btree):
        self.df = df
        self.cat_labels = cat_labels
        self.le = le
        self.btree = btree


_cache = None


def load():
    global _cache
    if _cache is not None:
        return _cache
    df = pd.read_parquet(os.path.join(CACHE_DIR, 'df.parquet'))
    cat_labels = np.load(os.path.join(CACHE_DIR, 'cat_labels.npy'))
    with open(os.path.join(CACHE_DIR, 'le.pkl'), 'rb') as fh:
        le = pickle.load(fh)
    with open(os.path.join(CACHE_DIR, 'btree.pkl'), 'rb') as fh:
        btree = pickle.load(fh)
    print(f'[poi_cache] loaded {len(df):,} POIs (raw cache only, no GNN models)', flush=True)
    _cache = PoiCache(df, cat_labels, le, btree)
    return _cache
