"""
run_comparison.py — trains/evaluates Place2Vec, Tile2Vec, and Urban2Vec
baselines on the same 92-location Austrian benchmark used for GeoSemantics
V2/V3, using the exact same retrieval-P@3 and separability metric functions
from evaluation.py for a fair, apples-to-apples comparison.

Usage:
    python run_comparison.py

Outputs (all written to this folder):
    comparison_results.json   — full metric dump for every method
    comparison_table.csv      — flat table for quick inspection
    comparison_chart.png      — bar chart, P@3 and separability per method
    README.md                 — methodology notes and honest caveats
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, HERE)

# IMPORTANT: poi_cache must be loaded (i.e. the parquet POI cache read) BEFORE
# matplotlib/torch/sklearn are imported. Importing those first and reading the
# parquet file afterwards segfaults reliably on this machine (reproduced with
# a minimal repro: matplotlib+torch+sklearn imported, then pd.read_parquet —
# crashes every time; same read before those imports — never crashes). Looks
# like a native-extension conflict, not anything wrong with the data or model
# code, but the ordering below is required to avoid it.
import poi_cache
_cache = poi_cache.load()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import evaluation as ev  # noqa: E402

from place2vec import build_place2vec  # noqa: E402
from tile2vec import build_tile2vec  # noqa: E402
from urban2vec import build_urban2vec  # noqa: E402


def _load_existing_geosemantics_numbers():
    path = os.path.join(BASE_DIR, 'validation_results', 'metrics.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)
    t = d.get('tests', {})
    out = {}
    if 'ret_v2' in t and 'sep_v2' in t:
        out['GeoSemantics V2 (homogeneous)'] = {
            'precision_at_3': t['ret_v2']['precision_at_k'],
            'separability': t['sep_v2']['separability'],
            'n_successful': t['ret_v2']['n_successful'],
            'source': 'validation_results/metrics.json (existing run)',
        }
    if 'ret_v3' in t and 'sep_v3' in t:
        out['GeoSemantics V3 (heterogeneous)'] = {
            'precision_at_3': t['ret_v3']['precision_at_k'],
            'separability': t['sep_v3']['separability'],
            'n_successful': t['ret_v3']['n_successful'],
            'source': 'validation_results/metrics.json (existing run)',
        }
    tfidf = t.get('tfidf_baseline')
    if tfidf is not None:
        tfidf = tfidf.copy()
        tfidf['precision_at_3'] = tfidf.get('precision_at_k') or tfidf.get('precision_at_3')
        out['TF-IDF (no graph, no learning)'] = tfidf
    else:
        out['TF-IDF (no graph, no learning)'] = {
            'precision_at_3': None,
            'separability': None,
            'n_successful': 0,
            'source': 'pending — not computed in the existing run (see manuscript Limitations)',
        }
    return out


def _eval_method(get_emb_fn, label):
    sep = ev.eval_embedding_separability(get_emb_fn, label=label)
    ret = ev.eval_retrieval(get_emb_fn, label=label, top_k=3)
    return {
        'precision_at_3': ret['precision_at_k'],
        'separability': sep['separability'],
        'intra_sim': sep['intra_sim'],
        'inter_sim': sep['inter_sim'],
        'n_successful': ret['n_successful'],
        'n_total': len(ev.BENCHMARK),
    }


def main():
    t0 = time.time()
    results = _load_existing_geosemantics_numbers()
    cache = _cache  # already loaded at module import time, before matplotlib/torch/sklearn

    print('=== Place2Vec ===', flush=True)
    p2v_fn = build_place2vec(cache)
    results['Place2Vec (POI co-occurrence)'] = _eval_method(p2v_fn, 'place2vec')
    results['Place2Vec (POI co-occurrence)']['source'] = 'trained in this run (PPMI-SVD, see README)'

    print('=== Tile2Vec ===', flush=True)
    t2v_fn, _ = build_tile2vec(cache)
    results['Tile2Vec (orthophoto only)'] = _eval_method(t2v_fn, 'tile2vec')
    results['Tile2Vec (orthophoto only)']['source'] = 'trained in this run (CNN + triplet loss, see README)'

    print('=== Urban2Vec ===', flush=True)
    u2v_fn, _ = build_urban2vec(cache=cache)
    results['Urban2Vec (orthophoto + POI fusion)'] = _eval_method(u2v_fn, 'urban2vec')
    results['Urban2Vec (orthophoto + POI fusion)']['source'] = (
        'trained in this run (cross-modal contrastive fusion, orthophoto substitutes Street View, see README)'
    )

    elapsed = time.time() - t0
    payload = {'results': results, 'elapsed_s': round(elapsed, 1), 'benchmark_n': len(ev.BENCHMARK)}

    out_json = os.path.join(HERE, 'comparison_results.json')
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print(f'[run_comparison] wrote {out_json}', flush=True)

    out_csv = os.path.join(HERE, 'comparison_table.csv')
    with open(out_csv, 'w', encoding='utf-8') as fh:
        fh.write('method,precision_at_3,separability,n_successful,n_total,source\n')
        for name, r in results.items():
            p3 = r.get('precision_at_3')
            sep = r.get('separability')
            fh.write(f'"{name}",{p3 if p3 is not None else ""},{sep if sep is not None else ""},'
                      f'{r.get("n_successful", "")},{r.get("n_total", len(ev.BENCHMARK))},"{r.get("source", "")}"\n')
    print(f'[run_comparison] wrote {out_csv}', flush=True)

    # Bar chart
    names = list(results.keys())
    p3_vals = [results[n].get('precision_at_3') or 0 for n in names]
    sep_vals = [results[n].get('separability') or 0 for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, p3_vals, w, label='Retrieval P@3', color='#3b82f6')
    ax.bar(x + w / 2, sep_vals, w, label='Embedding separability', color='#22c55e')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title(f'Baseline comparison on the {len(ev.BENCHMARK)}-location Austrian benchmark')
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(HERE, 'comparison_chart.png')
    fig.savefig(out_png, dpi=150)
    print(f'[run_comparison] wrote {out_png}', flush=True)

    print(f'[run_comparison] done in {elapsed:.1f}s', flush=True)
    for name, r in results.items():
        print(f'  {name:42s} P@3={r.get("precision_at_3")}  sep={r.get("separability")}')


if __name__ == '__main__':
    main()
