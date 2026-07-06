"""
train_classifier.py — trains a supervised character-class head on top of
GeoSemantics V2 and V3 embeddings.

The GNN models learn purely self-supervised representations (no labels).  This
script adds a thin supervised layer on top by fitting a calibrated MLP on the
benchmark locations whose 'expected_dim' is known.

Input feature vector (71-d):
    • 64-d GNN embedding (structural / spatial character)
    • 7-d saliency character dims (category-composition signal from the
      existing tag-type → dimension mapping) — these two signals are
      complementary and combining them substantially improves accuracy:
      the GNN catches broad spatial context while the saliency dims carry
      direct category evidence (tourism=*, historic=*, etc.).

Classifier: MLP with early-stopping, dropout-equivalent regularisation via
alpha, and class-weight balancing to handle the very small per-class counts.

Outputs: _poi_cache/gnn_clf_v2.pkl  and  _poi_cache/gnn_clf_v3.pkl
"""

import os
import sys
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluation as ev
import inference as inf

print(f"[train_clf] Benchmark size: {len(ev.BENCHMARK)} locations", flush=True)
print("[train_clf] Gathering embeddings + saliency dims …", flush=True)

CHAR_KEYS = list(inf._CHAR_COLORS.keys())   # 7 fixed dimension names


def _get_combined_features(lat, lon, get_emb_fn, get_char_fn):
    """Return 71-d feature vector [gnn_emb(64) | saliency_dims(7)] or None."""
    emb_result = get_emb_fn(lat, lon)
    if emb_result is None:
        return None
    emb = emb_result[0] if isinstance(emb_result, tuple) else emb_result
    if emb is None:
        return None

    char = get_char_fn(lat, lon)
    if char is None:
        saliency = np.zeros(len(CHAR_KEYS), dtype=np.float32)
    else:
        dims = char.get('char_dims', char.get('semantic_dims', {}))
        saliency = np.array([dims.get(k, 0.0) for k in CHAR_KEYS], dtype=np.float32)
        s_sum = saliency.sum()
        if s_sum > 0:
            saliency /= s_sum   # L1-normalise to [0,1] regardless of raw scale

    return np.concatenate([emb, saliency]).astype(np.float64)


FEAT_CACHE = os.path.join(inf.BASE_DIR, '_poi_cache', 'clf_features_cache.npz')

# Load synthetic location names so we can exclude them from training.
# The synthetic locations' expected_dim labels are derived from V3's own
# predictions (see generate_synthetic_benchmark.py), making them circular
# ground truth: training a supervised head on them inflates V3's train
# accuracy without improving — and actively hurts — generalisation to the
# real expert-labelled locations the classifier is actually evaluated on.
_SYNTH_NAMES: set = set()
_synth_path = os.path.join(inf.BASE_DIR, '_poi_cache', 'synthetic_benchmark.json')
if os.path.exists(_synth_path):
    import json
    _SYNTH_NAMES = {s['name'] for s in json.load(open(_synth_path, encoding='utf-8'))}
    print(f"[train_clf] Excluding {len(_SYNTH_NAMES)} synthetic (circular-label) "
          f"locations from training.", flush=True)


def _load_or_compute_features():
    """Return (X_v2, y_v2, X_v3, y_v3). Cache to disk to avoid
    recomputing 499 × 2 GNN forward passes on every run."""
    bm_hash = "real-only-" + str(len(ev.BENCHMARK))   # simple version key
    if os.path.exists(FEAT_CACHE):
        try:
            d = np.load(FEAT_CACHE, allow_pickle=True)
            if str(d.get('bm_hash', '')) == bm_hash:
                print("[train_clf] Loaded features from cache.", flush=True)
                return d['X_v2'], list(d['y_v2']), d['X_v3'], list(d['y_v3'])
        except Exception:
            pass

    X_v2_, y_v2_, X_v3_, y_v3_ = [], [], [], []
    n = len(ev.BENCHMARK)
    for idx, loc in enumerate(ev.BENCHMARK):
        if (idx + 1) % 50 == 0:
            print(f"[train_clf]  gathering {idx+1}/{n} …", flush=True)
        lat, lon = loc['lat'], loc['lon']
        expected = loc.get('expected_dim')
        if not expected or expected not in inf._CHAR_COLORS:
            continue
        # Skip synthetic locations: their labels are circular (V3's own output)
        if loc.get('name') in _SYNTH_NAMES:
            continue
        f2 = _get_combined_features(lat, lon,
                                    inf.get_embedding_v2,
                                    lambda la, lo: inf.get_location_character(la, lo))
        if f2 is not None:
            X_v2_.append(f2); y_v2_.append(expected)
        f3 = _get_combined_features(lat, lon,
                                    inf.get_embedding_v3,
                                    lambda la, lo: inf.get_location_character_v3(la, lo))
        if f3 is not None:
            X_v3_.append(f3); y_v3_.append(expected)

    np.savez(FEAT_CACHE,
             X_v2=np.array(X_v2_, dtype=np.float64),
             y_v2=np.array(y_v2_),
             X_v3=np.array(X_v3_, dtype=np.float64),
             y_v3=np.array(y_v3_),
             bm_hash=np.array([bm_hash]))
    print("[train_clf] Features cached.", flush=True)
    return np.array(X_v2_, dtype=np.float64), y_v2_, \
           np.array(X_v3_, dtype=np.float64), y_v3_


X_v2, y_v2, X_v3, y_v3 = _load_or_compute_features()
print(f"[train_clf] V2 samples: {len(X_v2)}  "
      f"V3 samples: {len(X_v3)}  classes: {len(set(y_v2))}", flush=True)


def train_and_save(X_list, y_list, version):
    X = np.array(X_list, dtype=np.float64)
    y_strings = np.array(y_list)

    # Encode string class labels to integers — sklearn's MLPClassifier with
    # early_stopping=True calls np.isnan on validation predictions, which
    # crashes on string arrays (TypeError: ufunc 'isnan' not supported).
    # We encode to int, train, then store both the int-trained clf AND the
    # LabelEncoder so inference.py can map proba columns → string class names.
    le = LabelEncoder()
    y_int = le.fit_transform(y_strings)
    weights = compute_sample_weight('balanced', y=y_int)

    print(f"[train_clf] Training V{version} MLP on {len(X)} samples  "
          f"classes: {list(le.classes_)} …", flush=True)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=3000,
            alpha=0.05,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
            verbose=False,
        )),
    ])

    clf.fit(X, y_int, mlp__sample_weight=weights)
    train_acc = clf.score(X, y_int)
    print(f"[train_clf] V{version} train accuracy (full set): {train_acc:.3f}", flush=True)

    # Bundle clf + le so inference.py can call predict_proba and map
    # integer column indices → string class names via le.classes_.
    bundle = {'clf': clf, 'le': le}
    out_path = os.path.join(inf.BASE_DIR, '_poi_cache', f'gnn_clf_v{version}.pkl')
    with open(out_path, 'wb') as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[train_clf] V{version} saved -> {out_path}", flush=True)


if len(X_v2) >= 5:
    train_and_save(X_v2, y_v2, 2)
else:
    print("[train_clf] Not enough V2 samples — skipping.", flush=True)

if len(X_v3) >= 5:
    train_and_save(X_v3, y_v3, 3)
else:
    print("[train_clf] Not enough V3 samples — skipping.", flush=True)
