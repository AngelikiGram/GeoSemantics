"""
precompute_temporal.py — Precompute real historical OSM character snapshots
for a curated set of representative Austrian locations.

For each location, fetches the actual historical OSM state at several past
years via Overpass `[date:]` attic queries (verified genuine — not a
current-data fallback, see temporal.py) and runs every snapshot through the
SAME trained saliency GNN + V2 embedding model used for live inference.

Output: _poi_cache/temporal_<year>.json (one file per year, matching the
schema morph_app.py's /api/morph/temporal route already reads), plus
_poi_cache/temporal_series.json (one file holding the full per-location
year-by-year series + embedding-drift trajectory, used by the new
/api/morph/temporal_series route and the "History" UI tab).

Usage:
    python precompute_temporal.py
    python precompute_temporal.py --years 2010 2014 2018 2022 now
"""
import argparse
import json
import os

import temporal as tmp

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_poi_cache')

# Same 7 archetypes evaluation.py uses to define the benchmark classes —
# a single curated, diverse set reused everywhere in the project.
STATIC_LOCATIONS = [
    {'id': 0, 'name': 'Vienna Stephansplatz',  'lat': 48.2082, 'lon': 16.3738, 'archetype': 'Urban core'},
    {'id': 1, 'name': 'Hallstatt',              'lat': 47.5625, 'lon': 13.6493, 'archetype': 'Tourism heritage hotspot'},
    {'id': 2, 'name': 'Grossglockner',          'lat': 47.0740, 'lon': 12.6950, 'archetype': 'Alpine nature area'},
    {'id': 3, 'name': 'Wien Hauptbahnhof',      'lat': 48.1968, 'lon': 16.3695, 'archetype': 'Transport hub'},
    {'id': 4, 'name': 'Melk',                   'lat': 48.1400, 'lon': 15.5600, 'archetype': 'Village community center'},
    {'id': 5, 'name': 'Vienna Donaustadt',      'lat': 48.2400, 'lon': 16.4100, 'archetype': 'Industrial/infrastructure'},
    {'id': 6, 'name': 'Vienna Floridsdorf',     'lat': 48.2450, 'lon': 16.4200, 'archetype': 'Residential suburb'},
    {'id': 7, 'name': 'Innsbruck Altstadt',     'lat': 47.2682, 'lon': 11.3927, 'archetype': 'Urban core'},
    {'id': 8, 'name': 'Salzburg Altstadt',      'lat': 47.7982, 'lon': 13.0450, 'archetype': 'Tourism heritage hotspot'},
    {'id': 9, 'name': 'Graz Hauptplatz',        'lat': 47.0707, 'lon': 15.4395, 'archetype': 'Urban core'},
    {'id': 10, 'name': 'Bregenz Seebuhne',      'lat': 47.5031, 'lon': 9.7471,  'archetype': 'Nature tourism'},
    {'id': 11, 'name': 'Neusiedl am See',       'lat': 47.9468, 'lon': 16.8419, 'archetype': 'Alpine nature area'},
    {'id': 12, 'name': 'Semmering',             'lat': 47.6333, 'lon': 15.8333, 'archetype': 'Alpine nature area'},
    {'id': 13, 'name': 'Durnstein (Wachau)',    'lat': 48.3914, 'lon': 15.5183, 'archetype': 'Tourism heritage hotspot'},
    {'id': 14, 'name': 'Klagenfurt center',     'lat': 46.6228, 'lon': 14.3051, 'archetype': 'Urban core'},
]

def load_locations(all_austrian=False):
    if not all_austrian:
        return STATIC_LOCATIONS
    df_path = os.path.join(CACHE_DIR, 'df.parquet')
    btree_path = os.path.join(CACHE_DIR, 'btree.pkl')
    if not (os.path.exists(df_path) and os.path.exists(btree_path)):
        print('Using static fallback locations.', flush=True)
        return STATIC_LOCATIONS
    try:
        import pandas as pd
        import pickle
        import numpy as np
        df_all = pd.read_parquet(df_path)
        with open(btree_path, 'rb') as fh:
            tree = pickle.load(fh)
        
        cities_df = df_all[df_all['place'] == 'city'].copy()
        towns_df = df_all[df_all['place'] == 'town'].copy()
        villages_df = df_all[df_all['place'] == 'village'].copy()
        suburbs_df = df_all[df_all['place'] == 'suburb'].copy()
        nh_df = df_all[df_all['place'] == 'neighbourhood'].copy()
        
        r_rad = 1000.0 / 6371000.0

        if len(villages_df) > 0:
            coords = np.radians(villages_df[['lat', 'lon']].values)
            counts = [len(idx) for idx in tree.query_radius(coords, r=r_rad)]
            villages_df['poi_density'] = counts
            top_villages_df = villages_df.sort_values(by='poi_density', ascending=False).head(1500)
        else:
            top_villages_df = pd.DataFrame()

        if len(suburbs_df) > 0:
            coords = np.radians(suburbs_df[['lat', 'lon']].values)
            counts = [len(idx) for idx in tree.query_radius(coords, r=r_rad)]
            suburbs_df['poi_density'] = counts
            top_suburbs_df = suburbs_df.sort_values(by='poi_density', ascending=False).head(300)
        else:
            top_suburbs_df = pd.DataFrame()

        if len(nh_df) > 0:
            coords = np.radians(nh_df[['lat', 'lon']].values)
            counts = [len(idx) for idx in tree.query_radius(coords, r=r_rad)]
            nh_df['poi_density'] = counts
            top_nh_df = nh_df.sort_values(by='poi_density', ascending=False).head(400)
        else:
            top_nh_df = pd.DataFrame()
            
        locations = []
        loc_id = 0
        for _, row in cities_df.iterrows():
            name = row['name'] if pd.notna(row['name']) else f"City at {row['lat']:.4f},{row['lon']:.4f}"
            locations.append({'id': loc_id, 'name': name, 'lat': float(row['lat']), 'lon': float(row['lon']), 'archetype': 'City'})
            loc_id += 1
        for _, row in towns_df.iterrows():
            name = row['name'] if pd.notna(row['name']) else f"Town at {row['lat']:.4f},{row['lon']:.4f}"
            locations.append({'id': loc_id, 'name': name, 'lat': float(row['lat']), 'lon': float(row['lon']), 'archetype': 'Town'})
            loc_id += 1
        for _, row in top_villages_df.iterrows():
            name = row['name'] if pd.notna(row['name']) else f"Village at {row['lat']:.4f},{row['lon']:.4f}"
            locations.append({'id': loc_id, 'name': name, 'lat': float(row['lat']), 'lon': float(row['lon']), 'archetype': 'Village'})
            loc_id += 1
        for _, row in top_suburbs_df.iterrows():
            name = row['name'] if pd.notna(row['name']) else f"Suburb at {row['lat']:.4f},{row['lon']:.4f}"
            locations.append({'id': loc_id, 'name': name, 'lat': float(row['lat']), 'lon': float(row['lon']), 'archetype': 'Suburb'})
            loc_id += 1
        for _, row in top_nh_df.iterrows():
            name = row['name'] if pd.notna(row['name']) else f"Neighbourhood at {row['lat']:.4f},{row['lon']:.4f}"
            locations.append({'id': loc_id, 'name': name, 'lat': float(row['lat']), 'lon': float(row['lon']), 'archetype': 'Neighbourhood'})
            loc_id += 1
            
        print(f'Loaded {len(locations)} locations dynamically from df.parquet ({len(cities_df)} cities, {len(towns_df)} towns, {len(top_villages_df)} top villages, {len(top_suburbs_df)} top suburbs, {len(top_nh_df)} top neighbourhoods).', flush=True)
        return locations
    except Exception as e:
        print(f'Error loading dynamic locations: {e}. Falling back to static locations.', flush=True)
        return STATIC_LOCATIONS


def _parse_years(raw):
    years = []
    for y in raw:
        years.append(None if y == 'now' else int(y))
    return years


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', nargs='+', default=['2010', '2014', '2018', '2020', '2022', '2024', 'now'])
    parser.add_argument('--fetch-radius', type=int, default=700)
    parser.add_argument('--char-radius', type=int, default=600)
    parser.add_argument('--delay', type=float, default=1.5)
    parser.add_argument('--all-austrian', action='store_true', help='Precompute all cities + high-density villages')
    args = parser.parse_args()
    years = _parse_years(args.years)

    locations = load_locations(all_austrian=args.all_austrian)

    os.makedirs(CACHE_DIR, exist_ok=True)
    series_path = os.path.join(CACHE_DIR, 'temporal_series.json')

    series = {}
    if os.path.exists(series_path):
        try:
            with open(series_path, encoding='utf-8') as fh:
                series = json.load(fh)
            cleaned_series = {}
            for k, loc in series.items():
                snaps = loc.get('snapshots', [])
                if snaps and all(s.get('insufficient') for s in snaps):
                    print(f'Discarding fully insufficient cache entry for {loc["name"]} to force retry.', flush=True)
                else:
                    cleaned_series[k] = loc
            series = cleaned_series
            print(f'Resuming — {len(series)} locations already done.', flush=True)
        except Exception:
            series = {}

    for loc in locations:
        key = str(loc['id'])
        if key in series:
            print(f'[skip] {loc["name"]} already done.', flush=True)
            continue
        print(f'\n=== {loc["name"]} ({loc["archetype"]}) ===', flush=True)
        snaps = tmp.analyze_series(loc['lat'], loc['lon'], years=years,
                                    fetch_radius=args.fetch_radius,
                                    char_radius=args.char_radius,
                                    delay=args.delay, verbose=True)
        series[key] = {**loc, 'snapshots': snaps}
        with open(series_path, 'w', encoding='utf-8') as fh:
            json.dump(series, fh, indent=None)
        print(f'  Saved. ({len(series)}/{len(locations)} locations done)', flush=True)

    # Also emit the per-year files the existing /api/morph/temporal route
    # reads, for backward compatibility with that endpoint.
    by_year = {}
    for loc_data in series.values():
        for snap in loc_data['snapshots']:
            if snap.get('insufficient'):
                continue
            y = snap['year']
            by_year.setdefault(y, []).append({
                'id': loc_data['id'], 'name': loc_data['name'],
                'lat': loc_data['lat'], 'lon': loc_data['lon'],
                'year': y, 'semantic_dims': snap['semantic_dims'],
                'label': snap['label'], 'n_pois': snap['n_pois'],
                'source': snap['source'],
            })
    for y, entries in by_year.items():
        fname = f'temporal_{y}.json'
        with open(os.path.join(CACHE_DIR, fname), 'w', encoding='utf-8') as fh:
            json.dump({'year': y, 'n_locations': len(entries), 'locations': entries}, fh)
        print(f'Wrote {fname} ({len(entries)} locations)', flush=True)

    print('\nDone. Restart morph_app.py to serve the new temporal data.', flush=True)


if __name__ == '__main__':
    main()
