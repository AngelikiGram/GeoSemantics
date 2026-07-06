"""
precompute_semantic_grid.py — Dense Austria-wide character grid for the
"Character Layer" map overlay (Google-Maps-style colored regions instead
of roads).

For each cell of a regular lat/lon grid, computes the same saliency-
weighted character formula used everywhere else in the app
(inference.get_location_character_v3) — not a separate heuristic — and
records the dominant dimension + full breakdown. Cells with fewer than 3
nearby POIs (open countryside gaps, foreign territory clipped by the
bounding box, lakes, etc.) are skipped, mirroring the existing saliency
heatmap precompute's behaviour.

This is pure local inference — no Overpass calls, no rate limits — so the
only cost is wall-clock CPU time. A 100x100 grid takes roughly 1-2 hours
depending on hardware; it is fully resumable (checkpoints every 200 cells)
so it is safe to stop and restart.

Output: _poi_cache/semantic_grid.json
        {grid: N, radius: R, cells: [{lat, lon, dominant_dim, semantic_dims, n_pois}, ...]}

Usage:
    python precompute_semantic_grid.py                  # 100x100, 700m radius
    python precompute_semantic_grid.py --grid 60         # coarser/faster preview
    python precompute_semantic_grid.py --radius 1000
"""
import argparse
import json
import os
import time

import numpy as np

import inference as inf
import _old.render_character_layer as rcl

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_poi_cache')
OUT_PATH  = os.path.join(CACHE_DIR, 'semantic_grid.json')

# Matches the real data extent (inference.df lat/lon range) with a small buffer.
LAT_RANGE = (46.30, 49.05)
LON_RANGE = (9.40, 17.20)


def build_grid(n):
    lats = np.linspace(LAT_RANGE[0], LAT_RANGE[1], n)
    lons = np.linspace(LON_RANGE[0], LON_RANGE[1], n)
    return [(float(la), float(lo)) for la in lats for lo in lons]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=100, help='N x N grid resolution (default 100)')
    parser.add_argument('--radius', type=int, default=700, help='Character radius in meters (default 700, the "Meso" scale)')
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        rcl.ensure_boundary_mask()
        print('[grid] Austria boundary mask ready (clips render to the real border).', flush=True)
    except Exception as e:
        print(f'[grid] WARNING: could not fetch/build Austria boundary mask ({e}) — '
              f'the Character Layer render may bleed color past the border until this '
              f'is retried with network access.', flush=True)

    points = build_grid(args.grid)
    print(f'[grid] {args.grid}x{args.grid} = {len(points)} candidate cells, radius={args.radius}m', flush=True)

    cells = []
    done_keys = set()
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, encoding='utf-8') as fh:
                prev = json.load(fh)
            if prev.get('grid') == args.grid and prev.get('radius') == args.radius:
                cells = prev.get('cells', [])
                done_keys = {(round(c['lat'], 6), round(c['lon'], 6)) for c in cells}
                print(f'[grid] Resuming — {len(cells)} cells already done.', flush=True)
        except Exception:
            pass

    t_start = time.time()
    n_skip = 0
    for i, (lat, lon) in enumerate(points):
        key = (round(lat, 6), round(lon, 6))
        if key in done_keys:
            continue
        char = inf.get_location_character_v3(lat, lon, args.radius)
        if char is not None:
            dims = char['char_dims']
            cells.append({
                'lat': round(lat, 5), 'lon': round(lon, 5),
                'dominant_dim': max(dims, key=dims.get),
                'semantic_dims': {k: round(v, 3) for k, v in dims.items()},
                'label': char.get('label', '-'),
                'n_pois': char.get('n_pois', 0),
            })
        else:
            n_skip += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(points):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_s = (len(points) - i - 1) / rate if rate > 0 else 0
            print(f'  [{i+1}/{len(points)}] {len(cells)} valid, {n_skip} skipped — '
                  f'{elapsed:.0f}s elapsed, ETA {eta_s/60:.0f} min', flush=True)
            with open(OUT_PATH, 'w', encoding='utf-8') as fh:
                json.dump({'grid': args.grid, 'radius': args.radius, 'cells': cells}, fh)

    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        json.dump({'grid': args.grid, 'radius': args.radius, 'cells': cells}, fh)
    print(f'\n[grid] Done — {len(cells)} valid cells saved to {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
