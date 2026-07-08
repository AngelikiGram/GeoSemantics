# GeoSemantics
**An Interactive Engine for Explainable, Multi-Scale Place Character from Heterogeneous OpenStreetMap Graphs**

Flask web application that learns spatial character fingerprints from OpenStreetMap POI graphs using a heterogeneous GATv2 GNN (V3) with self-supervised contrastive training. Produces a 7-dimensional character readout (Urban, Tourism, Heritage, Nature, Transport, Infrastructure, Community) alongside an interactive UMAP embedding landscape, temporal analysis, and morphological comparison tools.

---

## Requirements

- Python 3.10+
- ~4 GB disk for precomputed cache
- `austrian-pois.geojson` source file (948 MB, not in repo — see Data below)

Install dependencies:

```bash
pip install flask torch torch_geometric pandas pyarrow scikit-learn umap-learn numpy scipy
pip install rtree shapely requests pillow tqdm
```

---

## Data

The raw POI source (`austrian-pois.geojson`) is not committed. Either:

- **Download from Releases** — grab `_poi_cache.zip` from the [GitHub Releases](../../releases) page, unzip into the project root, and skip the Preprocess step.
- **Build from source** — place `austrian-pois.geojson` in the project root and run preprocess (below).

---

## Setup: Preprocess (run once)

These build the `_poi_cache/` folder. Run in order:

```bash
# 1. Core POI index + UMAP embedding
python precompute_morph.py

# 2. All-Austria UMAP landscape (slow, ~30 min)
python precompute_all_places.py

# 3. Semantic grid overlay
python precompute_semantic_grid.py

# 4. Temporal snapshots (2010–now)
python precompute_temporal.py

# 5. Train character classifiers (V2 + V3)
python train_classifier.py
```

---

## Model Training (optional — pretrained weights included)

Pretrained `.pt` files (including `geosemantics_v3.pt` and `geosemantics_v2.pt`) are provided in the `models/` directory. To retrain from scratch:

```bash
# V2 — homogeneous GATv2
python geosemantics_v2.py

# V3 — heterogeneous 5-node-type GATv2 (recommended, ~2–4 h CPU)
python geosemantics_v3.py
```

---

## Run the App

```bash
python morph_app.py
# Open http://localhost:5000
```

---

## Reproduction Instructions

To reproduce the study findings (Tables and Figures from the paper):

1. **Evaluate the models & ablations**
```bash
# Runs full evaluation on the Austrian benchmark (V2 vs V3, ablation, plots)
python validation.py --plots --ablation
```
*Outputs will be saved in `validation_results/`, containing the separability metrics, character accuracy, and generated plots.*

2. **Run the baseline comparison**
```bash
# Compares Typed-GAT/Flat-GAT against Place2Vec, Tile2Vec, and Urban2Vec
cd baseline_comparison
python run_comparison.py
```
*Outputs are saved to `baseline_comparison/comparison_results.json` and `baseline_comparison/comparison_table.csv`.*

---

## Project Structure

```
morph_app.py                  ← Flask server
inference.py                  ← all inference + saliency logic
geosemantics_v2.py / v3.py    ← model definitions
temporal.py                   ← temporal change analysis
evaluation.py                 ← benchmark metrics
train_classifier.py           ← MLP character classifier
precompute_*.py               ← one-time precompute scripts
static/morph.html             ← frontend
saliency/                     ← saliency GCN
models/                       ← pretrained model weights (.pt files)
baseline_comparison/          ← Place2Vec, Tile2Vec, Urban2Vec baselines
validation_results/           ← output of validation.py (gitignored)
_poi_cache/                   ← precomputed data (gitignored)
```

---

## Files Not Needed at Runtime

These are development/research artefacts and can be ignored:

- `spatial_semantic_gnn.pt` — old V1 model, superseded
- `austrian-pois.geojson` — raw source, only needed for preprocess
- `results.tex`, `ref.bib` — paper manuscript
- `scripts/` — utility scripts (LaTeX helpers, one-off patches)
- `paper_figures/` — screenshot exports for paper
