# ============================================================
# SPATIAL–SEMANTIC GNN FOR SALIENCY REGRESSION (WORKING VERSION)
# ============================================================

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import math
import random
import os
import pickle

# ---------------------------- CONFIG -----------------------------

RADIUS_MIN = 10     # meters
RADIUS_MAX = 5000   # meters
K_NEIGHBORS = 8
N_TRAIN_SAMPLES = 2000
BATCH_SIZE = 8
EPOCHS = 500

# ---------------------------- HELPERS -----------------------------

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon)*np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return np.arctan2(x, y)

def parse_other_tags(tags):
    if not tags or not isinstance(tags, str):
        return {}
    tags = tags.strip()
    if tags.startswith('{'):
        try:
            return json.loads(tags)
        except:
            pass
    d = {}
    for segment in tags.split(","):
        if '=>"' in segment:
            k,v = segment.split('=>"',1)
            d[k.replace('"','').strip()] = v.replace('"','').strip()
    return d

# ---------------------------- LOAD DATA -----------------------------

with open('../austrian-pois.geojson','r',encoding='utf-8') as f:
    geo = json.load(f)

features = geo["features"]

rows=[]
for feat in features:
    lon,lat = feat["geometry"]["coordinates"]
    props = feat["properties"]
    rows.append({
        "lon":lon, "lat":lat,
        "highway":props.get("highway",""),
        "natural":props.get("natural",""),
        "place":props.get("place",""),
        "man_made":props.get("man_made",""),
        "other_tags":props.get("other_tags","")
    })

df = pd.DataFrame(rows)

# ---------------------------- CATEGORY ENCODING -----------------------------

cat_cols = ["highway","natural","place","man_made"]

cat_data = df[cat_cols].fillna("").agg(
    lambda row: next((v for v in row if v), ""), axis=1
).values.reshape(-1,1)

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_features = ohe.fit_transform(cat_data)

# ---------------------------- OTHER TAGS ENCODING -----------------------------

other_tags_list = [" ".join(parse_other_tags(t).keys()) for t in df["other_tags"]]
vectorizer = CountVectorizer(max_features=32)
other_tags_features = vectorizer.fit_transform(other_tags_list).toarray()

# ============================================================
# SALIENCY LABEL COMPUTATION
# ============================================================

def compute_saliency_labels(graph, cat_features_sel):
    edge_index = graph.edge_index.numpy()
    n = graph.x.shape[0]

    deg = np.bincount(edge_index[0], minlength=n)

    dists = graph.edge_attr[:,0].numpy()

    centrality = np.zeros(n)
    for i in range(n):
        mask = edge_index[0] == i
        if mask.any():
            centrality[i] = 1/(dists[mask].mean() + 1e-6)

    uniqueness = np.zeros(n)
    for i in range(n):
        mask = edge_index[0] == i
        neigh = edge_index[1][mask]
        if len(neigh)>0:
            own = np.argmax(cat_features_sel[i])
            neigh_cats = [np.argmax(cat_features_sel[j]) for j in neigh]
            freq = neigh_cats.count(own)/len(neigh)
            uniqueness[i] = 1 - freq
        else:
            uniqueness[i] = 1

    sal = 0.5*centrality + 0.3*uniqueness + 0.2*(deg/(deg.max()+1e-6))
    return sal.astype(np.float32)

# ============================================================
# UNIFIED GRAPH BUILDER (WITH SALIENCY)
# ============================================================

def build_local_graph(query_lat, query_lon, radius=None):
    if radius is None:
        radius = random.uniform(RADIUS_MIN, RADIUS_MAX)

    d = haversine(df["lon"].values, df["lat"].values, query_lon, query_lat)
    mask = d <= radius
    sel_df = df[mask].reset_index(drop=True)

    if len(sel_df) < 3:
        return None

    sel_cat = cat_features[mask]
    sel_other = other_tags_features[mask]

    coords = sel_df[["lat","lon"]].values
    coords_rel = coords - np.array([query_lat, query_lon])
    coords_rel = StandardScaler().fit_transform(coords_rel)

    node_features = np.hstack([coords_rel, sel_cat, sel_other])

    # ---- NEIGHBORS ----
    k = min(K_NEIGHBORS, len(sel_df)-1)
    nbr = NearestNeighbors(n_neighbors=k+1).fit(coords)
    dist_idx, idxs = nbr.kneighbors(coords)

    edge_idx=[]
    edge_attr=[]
    for i, neighs in enumerate(idxs):
        for j,n in enumerate(neighs[1:]):
            dist = haversine(coords[i,1],coords[i,0],coords[n,1],coords[n,0])
            brng = bearing(coords[i,1],coords[i,0],coords[n,1],coords[n,0])
            edge_idx.append([i,n])
            edge_attr.append([dist, math.sin(brng), math.cos(brng),
                              int(np.argmax(sel_cat[i]) == np.argmax(sel_cat[n]))])

    edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # ---- SALIENCY ----
    sal = compute_saliency_labels(g, sel_cat)
    g.y = torch.tensor(sal, dtype=torch.float)

    return g

# ============================================================
# SAMPLE TRAINING GRAPHS
# ============================================================

min_lat, max_lat = df["lat"].min(), df["lat"].max()
min_lon, max_lon = df["lon"].min(), df["lon"].max()

train_graphs=[]
tries=0

while len(train_graphs) < N_TRAIN_SAMPLES and tries < 3*N_TRAIN_SAMPLES:
    q_lat = random.uniform(min_lat, max_lat)
    q_lon = random.uniform(min_lon, max_lon)
    g = build_local_graph(q_lat, q_lon)
    if g is not None:
        train_graphs.append(g)
    tries += 1

print(f"Sampled {len(train_graphs)} graphs total.")

# Remove graphs missing labels (should be none now)
train_graphs = [g for g in train_graphs if hasattr(g,"y")]
print(f"Valid training graphs: {len(train_graphs)}")

if len(train_graphs)==0:
    print("ERROR: No graphs with labels were created. Check radius!")
    raise SystemExit

pickle.dump(train_graphs, open("sampled_train_graphs.pkl","wb"))
print("Saved training graphs.")

# ============================================================
# MODEL
# ============================================================

in_channels = train_graphs[0].x.shape[1]

class SaliencyGNN(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, 1)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index).squeeze(-1)

model = SaliencyGNN(in_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ============================================================
# TRAINING LOOP
# ============================================================

loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)

best_loss = float("inf")
epochs_no_improve = 0
patience = 20

for epoch in range(EPOCHS):
    model.train()
    total=0

    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total += loss.item()

    avg = total/len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss = {avg:.4f}")

    if avg < best_loss - 1e-6:
        best_loss = avg
        epochs_no_improve = 0
        torch.save(model.state_dict(), "saliency_gnn.pt")
        print("Saved best model.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict_saliency(query_lat, query_lon, model_path="saliency_gnn.pt"):
    model = SaliencyGNN(in_channels)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    g = build_local_graph(query_lat, query_lon, radius=1000)
    if g is None:
        return None, None

    with torch.no_grad():
        pred = model(g.x, g.edge_index).numpy()

    return pred, g

