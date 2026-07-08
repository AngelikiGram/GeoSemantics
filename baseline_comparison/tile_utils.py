"""
tile_utils.py — fetches and caches real Austrian orthophoto tiles from the
basemap.at public WMTS/XYZ service for the Tile2Vec and Urban2Vec baselines.

basemap.at (bmaporthofoto30cm layer) is Austria's free, public, CC-BY-4.0
aerial-imagery basemap, served as plain XYZ tiles with no API key:
    https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg

We use this as a substitute for the imagery each baseline was originally
designed around:
  - Tile2Vec was designed for satellite tiles — this is a faithful match.
  - Urban2Vec was designed for street-level Street View photos, which would
    require a paid API. Top-down orthophotos are the closest free, legal
    substitute available for Austria, but they are NOT eye-level imagery —
    this is an honest substitution, not a reproduction, and is reported as
    such in README.md.
"""
import hashlib
import math
import os
import time

import numpy as np
import requests
from PIL import Image

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tile_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

TILE_URL = 'https://mapsneu.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/{z}/{y}/{x}.jpeg'
ZOOM = 17  # ~0.8 m/px at Austria's latitude, ~207m tile extent — comparable to our 200m micro scale
TILE_SIZE = 64  # downsized for a lightweight CPU-trainable CNN


def _latlon_to_tile(lat, lon, z=ZOOM):
    lat_rad = math.radians(lat)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile(lat, lon, retries=3):
    """Returns a (TILE_SIZE, TILE_SIZE, 3) float32 array in [0,1], cached on disk."""
    x, y = _latlon_to_tile(lat, lon)
    key = hashlib.md5(f'{ZOOM}_{x}_{y}'.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f'{key}.jpg')

    if not os.path.exists(cache_path):
        url = TILE_URL.format(z=ZOOM, y=y, x=x)
        for attempt in range(retries):
            try:
                r = requests.get(url, timeout=15, headers={'User-Agent': 'GeoSemantics-Research/1.0'})
                if r.status_code == 200 and r.content:
                    with open(cache_path, 'wb') as fh:
                        fh.write(r.content)
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1.0 * (attempt + 1))
        else:
            return None

    try:
        img = Image.open(cache_path).convert('RGB').resize((TILE_SIZE, TILE_SIZE))
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def fetch_tiles_batch(coords, desc='', max_workers=12):
    """coords: list of (lat, lon). Returns list of arrays (or None for failures),
    same order as input. Fetched concurrently — the basemap.at tile fetch is
    network-bound, not CPU-bound, so this is a large, safe speedup."""
    from concurrent.futures import ThreadPoolExecutor

    out = [None] * len(coords)
    n_done, n_fail = 0, 0

    def _work(i_latlon):
        i, (lat, lon) = i_latlon
        return i, fetch_tile(lat, lon)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, img in ex.map(_work, enumerate(coords)):
            out[i] = img
            n_done += 1
            if img is None:
                n_fail += 1
            if n_done % 50 == 0:
                print(f'[tiles{(" " + desc) if desc else ""}] {n_done}/{len(coords)} fetched, {n_fail} failed', flush=True)
    return out
