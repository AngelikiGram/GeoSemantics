"""
fix_v3_cache.py
───────────────
Regenerates the two missing V3 cache files (v3_type_les.pkl, v3_type_vocabs.json)
using the existing v3_node_types.npy and the loaded POI dataframe.
Run once, from the project root:
    python scripts/fix_v3_cache.py
"""
import os, json, pickle, sys

import numpy as np
from sklearn.preprocessing import LabelEncoder

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT_DIR)

CACHE_DIR = os.path.join(_ROOT_DIR, '_poi_cache')

print('[fix] Loading existing npy files …', flush=True)
node_types  = np.load(os.path.join(CACHE_DIR, 'v3_node_types.npy'))
type_labels = np.load(os.path.join(CACHE_DIR, 'v3_type_labels.npy'))

print('[fix] Loading POI dataframe (reuses inference cache) …', flush=True)
import inference as inf
df = inf.df

from geosemantics_v3 import (
    N_NODE_TYPES, NODE_TYPE_NAMES, _get_type_category
)

print('[fix] Extracting per-type category strings …', flush=True)
n        = len(df)
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

print('[fix] Fitting per-type LabelEncoders …', flush=True)
type_les    = []
type_vocabs = []
for t in range(N_NODE_TYPES):
    mask   = node_types == t
    le_t   = LabelEncoder()
    cats_t = type_cats[mask] if mask.sum() > 0 else np.array(['unknown'])
    le_t.fit(cats_t)
    type_les.append(le_t)
    type_vocabs.append(int(len(le_t.classes_)))
    print(f'  {NODE_TYPE_NAMES[t]}: {mask.sum():,} nodes, {len(le_t.classes_)} categories', flush=True)

print('[fix] Saving pkl and json …', flush=True)
with open(os.path.join(CACHE_DIR, 'v3_type_les.pkl'), 'wb') as fh:
    pickle.dump(type_les, fh)
with open(os.path.join(CACHE_DIR, 'v3_type_vocabs.json'), 'w') as fh:
    json.dump(type_vocabs, fh)

print('[fix] Done. Restart morph_app.py to pick up the new cache.', flush=True)
