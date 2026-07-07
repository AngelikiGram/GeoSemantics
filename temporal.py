"""
temporal.py — Real historical OSM change detection via Overpass attic queries.

For a location, fetches the ACTUAL historical OSM tag state at past years
(Overpass `[date:"..."]` attic queries — verified against the live Vienna-
area node count: 1,062 nodes in a 600 m radius as of 2010-01-01 vs. 27,072
today, so the date filter genuinely rewinds the database, it is not a
current-data fallback) and runs each snapshot through the SAME trained
models used for live inference (saliency GNN + V2 GATv2 embedding + the
rule-based V3 character formula) — not a simplified heuristic. This makes
every year directly comparable to the live "now" reading the rest of the
app shows, because it is computed by the identical pipeline.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

import geo_snapshot as gs

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_CACHE_DIR = os.path.join(BASE_DIR, '_poi_cache', 'temporal_raw')
os.makedirs(RAW_CACHE_DIR, exist_ok=True)

OVERPASS_URLS = [
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass.openstreetmap.fr/api/interpreter',
    'https://overpass-api.de/api/interpreter',
    'https://lz4.overpass-api.de/api/interpreter',
    'https://z.overpass-api.de/api/interpreter',
]

# Only nodes carrying one of these keys are kept — mirrors the semantic key
# set inference.get_poi_label / the V3 node-typer actually use, so we don't
# drag in thousands of untagged way-shape vertices Overpass also returns.
_SEMANTIC_KEYS = {
    'amenity', 'shop', 'tourism', 'historic', 'leisure', 'natural', 'highway',
    'railway', 'public_transport', 'aeroway', 'man_made', 'power', 'barrier',
    'emergency', 'healthcare', 'office', 'craft', 'club', 'gambling',
    'vending', 'building', 'landuse', 'place', 'waterway', 'sport',
}

_HEADERS = {'User-Agent': 'GeoSemantics-research/1.0 (academic project; contact angiegram99@gmail.com)'}

DEFAULT_YEARS = (2010, 2014, 2018, 2020, 2022, 2024, None)   # None = "now" (live, no date filter)


def _cache_path(lat, lon, year, radius):
    tag = 'now' if year is None else str(year)
    return os.path.join(RAW_CACHE_DIR, f'{lat:.5f}_{lon:.5f}_{tag}_{radius}.json')


_OFFLINE_URLS = set()

def fetch_year_records(lat, lon, year, radius=700, timeout=25, retries=3):
    """Fetch + cache the semantically-tagged node set for one (location, year).

    year=None fetches the live current state (no date filter — fastest and
    most reliable path). Returns a list of {'lat','lon','tags'} dicts.

    Attic queries over a large radius are slow on the public Overpass
    instance and occasionally 504 under load; retries with backoff, and as
    a last resort shrinks the radius (smaller area = faster query) rather
    than failing the whole location/year outright.
    """
    cpath = _cache_path(lat, lon, year, radius)
    if os.path.exists(cpath):
        with open(cpath, encoding='utf-8') as fh:
            return json.load(fh)

    date_clause = '' if year is None else f'[date:"{year}-01-01T00:00:00Z"]'
    last_exc = None
    r = radius

    for attempt in range(retries):
        # Filter active urls inside the attempt loop for real-time threads updates
        active_urls = [u for u in OVERPASS_URLS if u not in _OFFLINE_URLS] or OVERPASS_URLS
        url = active_urls[attempt % len(active_urls)]
        query = (f'[out:json][timeout:{timeout - 10}]{date_clause};'
                  f'node(around:{r},{lat},{lon});out tags center;')
        try:
            resp = requests.post(url, data={'data': query},
                                  headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
            elements = resp.json().get('elements', [])
            records = []
            for el in elements:
                tags = el.get('tags') or {}
                if not tags or not any(k in tags for k in _SEMANTIC_KEYS):
                    continue
                if 'lat' not in el or 'lon' not in el:
                    continue
                records.append({'lat': el['lat'], 'lon': el['lon'], 'tags': tags})
            with open(_cache_path(lat, lon, year, radius), 'w', encoding='utf-8') as fh:
                json.dump(records, fh)
            return records
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError)):
                _OFFLINE_URLS.add(url)
            elif isinstance(exc, requests.exceptions.HTTPError) and exc.response.status_code == 429:
                _OFFLINE_URLS.add(url)
            time.sleep(3 * (attempt + 1))
            r = max(500, int(r * 0.6))   # shrink radius on retry — faster query
    raise last_exc


def analyze_year(lat, lon, year, fetch_radius=700, char_radius=600):
    """Fetch one (location, year) snapshot and run it through the real
    trained models. Returns a dict ready to serialise to JSON."""
    label_year = 'now' if year is None else year
    try:
        records = fetch_year_records(lat, lon, year, radius=fetch_radius)
    except Exception as exc:
        return {'year': label_year, 'n_pois_raw': 0, 'insufficient': True, 'error': str(exc)}
    if len(records) < 3:
        return {'year': label_year, 'n_pois_raw': len(records), 'insufficient': True}

    rec_df = gs.records_to_df(records)
    char   = gs.character_from_records(rec_df, lat, lon, radius=char_radius)
    emb, _attn = gs.embedding_from_records(rec_df, lat, lon)

    return {
        'year':          label_year,
        'n_pois_raw':    len(records),
        'semantic_dims': char['char_dims'] if char else {},
        'label':         char['label'] if char else '-',
        'n_pois':        char['n_pois'] if char else 0,
        'embedding':     emb.tolist() if emb is not None else None,
        'source':        'overpass_live' if year is None else 'overpass_historical',
    }


def analyze_series(lat, lon, years=DEFAULT_YEARS, fetch_radius=700,
                    char_radius=600, delay=1.5, verbose=False):
    """Run analyze_year across a sequence of years and attach embedding
    drift (L2 distance between consecutive genuine V2 embeddings) in parallel."""
    def fetch_one(args):
        idx, year = args
        try:
            if delay > 0 and idx > 0:
                time.sleep(idx * delay)
            records = fetch_year_records(lat, lon, year, radius=fetch_radius)
            return year, records, None
        except Exception as exc:
            return year, None, exc

    with ThreadPoolExecutor(max_workers=len(years)) as executor:
        results = list(executor.map(fetch_one, enumerate(years)))

    fetched_data = {}
    for y, recs, err in results:
        fetched_data[y] = (recs, err)

    out = []
    prev_emb = None
    for year in years:
        label_year = 'now' if year is None else year
        recs, err = fetched_data[year]
        if err is not None:
            snap = {'year': label_year, 'n_pois_raw': 0, 'insufficient': True, 'error': str(err)}
        elif recs is None or len(recs) < 3:
            snap = {'year': label_year, 'n_pois_raw': len(recs) if recs is not None else 0, 'insufficient': True}
        else:
            try:
                rec_df = gs.records_to_df(recs)
                char   = gs.character_from_records(rec_df, lat, lon, radius=char_radius)
                emb, _attn = gs.embedding_from_records(rec_df, lat, lon)
                snap = {
                    'year':          label_year,
                    'n_pois_raw':    len(recs),
                    'semantic_dims': char['char_dims'] if char else {},
                    'label':         char['label'] if char else '-',
                    'n_pois':        char['n_pois'] if char else 0,
                    'embedding':     emb.tolist() if emb is not None else None,
                    'source':        'overpass_live' if year is None else 'overpass_historical',
                }
            except Exception as exc:
                snap = {'year': label_year, 'n_pois_raw': len(recs), 'insufficient': True, 'error': str(exc)}

        emb  = np.array(snap['embedding']) if snap.get('embedding') else None
        snap['drift_from_prev'] = (float(np.linalg.norm(emb - prev_emb))
                                    if emb is not None and prev_emb is not None else None)
        if emb is not None:
            prev_emb = emb
        out.append(snap)
        if verbose:
            yl = snap['year']
            if snap.get('insufficient'):
                print(f'    {yl}: insufficient data ({snap["n_pois_raw"]} raw nodes)', flush=True)
            else:
                print(f'    {yl}: {snap["n_pois"]} POIs · {snap["label"]} · '
                      f'drift={snap["drift_from_prev"]}', flush=True)
    return out
