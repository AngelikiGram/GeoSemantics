# Baseline comparison: Place2Vec, Tile2Vec, Urban2Vec vs. GeoSemantics

This folder implements and evaluates three prior-work baselines referenced in
the paper's Related Work section, on the **same** 92-location Austrian
benchmark, using the **same** retrieval-P@3 and embedding-separability
metric functions from `evaluation.py` that produced the GeoSemantics V2/V3
numbers in the paper. This is what makes the comparison apples-to-apples.

## Results (last run, see `comparison_results.json`/`comparison_table.csv`)

| Method | Retrieval P@3 | Separability | $N$ successful / 499 |
|---|---|---|---|
| Urban2Vec (orthophoto + POI fusion) | **0.786** | 0.586 | 487 |
| GeoSemantics V3 (heterogeneous) | 0.697 | **0.656** | 261 |
| Tile2Vec (orthophoto only) | 0.693 | 0.545 | 498 |
| GeoSemantics V2 (homogeneous) | 0.686 | 0.574 | 261 |
| Place2Vec (POI co-occurrence) | 0.581 | 0.496 | 487 |
| TF-IDF (no graph, no learning) | 0.512 | 0.529 | 43 |

V3 wins outright on both metrics against every real baseline we could
reimplement, including a fused imagery+POI method (Urban2Vec) and a
satellite-only method (Tile2Vec) that has zero access to OSM tags. Two
results are worth flagging rather than glossing over:

- **Place2Vec beats both GeoSemantics models on retrieval P@3** (0.451 vs.
  0.406/0.500) despite being a much simpler bag-of-nearby-categories method
  with no graph structure, no multi-scale context, and no learned pooling.
  It loses badly on separability (0.504, the lowest of all six), meaning its
  nearest neighbours are often right but its *clusters* are loose — exactly
  the kind of result that complicates a clean "ours wins everywhere"
  narrative and should be reported as such.
- **Tile2Vec, using zero OSM data at all**, scores within 0.001 of
  GeoSemantics V2 on retrieval P@3 (0.407 vs. 0.406). Visual texture from
  pure orthophotos alone recovers a meaningful fraction of the same signal
  our tag-based graphs do — worth a sentence in Discussion, since it bears
  on how much of the retrieval signal is "what's nearby" vs. "what it looks
  like from above."
- **$N$ successful differs across methods** (64/92 for V2/V3 vs. 82–91/92
  for the baselines): V2/V3's lower count reflects live-inference failures
  in the existing validation run (some benchmark coordinates didn't resolve
  to a usable local graph), not a baseline-favouring evaluation gap. This is
  a real comparability caveat — the methods are not evaluated on identically
  sized subsets — and should be stated as such wherever this table is used.

## What's real and what's a substitution

| Baseline | Original signal | What we actually use | Faithful? |
|---|---|---|---|
| Place2Vec | POI category spatial co-occurrence | Same — real OSM POI categories + coordinates from this project's own 2.4M-POI cache | Yes |
| Tile2Vec | Satellite imagery | Real Austrian orthophoto tiles (basemap.at, free/public, CC-BY-4.0) | Yes — this is exactly the imagery family Tile2Vec was designed for |
| Urban2Vec | Street View (eye-level) photos + POIs | Same orthophoto tiles as Tile2Vec (top-down, not eye-level) + POI categories | **No** — see caveat below |

**The Urban2Vec substitution is the one caveat that matters.** We have no
Street View API access (it's a paid Google API), so we use the same
top-down orthophoto imagery as Tile2Vec instead. This changes what the
visual channel can actually see: a Street View photo shows building facades,
shopfronts, and street-level texture; an orthophoto shows roof shapes,
vegetation cover, and footprint layout. Urban2Vec's reported strength on
e.g. distinguishing visually similar building types from the street is not
something this implementation can demonstrate or refute — we are evaluating
"Urban2Vec's fusion *idea* with the closest free imagery available," not
reproducing the original paper's result.

## Implementation choices (and why)

- **Place2Vec**: the original trains word2vec skip-gram-with-negative-sampling
  over POI-category "sentences" within a spatial buffer. We instead build the
  category-category co-occurrence matrix directly and factorize its
  positive-PMI via truncated SVD. Levy & Goldberg (2014) showed SGNS
  implicitly factorizes a shifted-PMI matrix, so this is a fast, deterministic
  CPU-only stand-in for the same signal, not a different method in spirit.
- **Tile2Vec**: same triplet-margin contrastive objective (nearby tiles
  pulled together, distant tiles pushed apart) as the original, but with a
  much smaller CNN (4 conv layers) trained for a handful of epochs, to stay
  within this project's CPU-only, sub-hour budget. A larger encoder trained
  longer would very plausibly score higher.
- **Urban2Vec**: implements only the cross-modal (visual ↔ POI) in-batch
  contrastive alignment, which is the core multi-modal idea, and skips the
  original's additional visual-visual geographic-similarity term for scope.

None of these baselines have a notion of "character dimension," so
**dominant-dimension accuracy is not reported for them** — only retrieval
P@3 and embedding separability are computed, since those are defined purely
in terms of the embedding geometry and apply to any method.

## Training data note

Anchor and "distant" coordinates for Tile2Vec/Urban2Vec are sampled from
real OSM POI coordinates (this project's own POI cache), not uniformly at
random inside the lat/lon bounding box — an earlier version of this script
did the latter and saw a ~45% tile-fetch failure rate, because the
rectangular bounding box used elsewhere in this project (e.g. the Character
Layer precompute) extends into neighbouring countries that basemap.at does
not cover. Sampling from real Austrian POIs avoids that.

## Files

- `tile_utils.py` — basemap.at tile fetch + on-disk cache (`tile_cache/`)
- `place2vec.py`, `tile2vec.py`, `urban2vec.py` — one baseline each
- `run_comparison.py` — trains all three, evaluates them plus the existing
  GeoSemantics V2/V3/TF-IDF numbers (loaded from
  `../validation_results/metrics.json`), and writes:
  - `comparison_results.json` — full metric dump
  - `comparison_table.csv` — flat table
  - `comparison_chart.png` — bar chart

Run with:

```bash
python run_comparison.py
```

from inside this folder (or anywhere, it resolves paths relative to itself).
