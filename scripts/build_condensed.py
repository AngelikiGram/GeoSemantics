import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import condense

PROPOSED_LATEX = r"""\documentclass[12pt]{gsis}
\usepackage[misc]{ifsym}
\usepackage{graphicx,subfigure}
\usepackage{hyperref,caption}
\usepackage[para,online,flushleft]{threeparttable}
\usepackage{multirow,booktabs}
\usepackage{geometry}
\usepackage{tabularx}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usetikzlibrary{positioning, arrows.meta}
\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
\usepackage{natbib}% Citation support using natbib.sty
\bibpunct[, ]{(}{)}{;}{a}{}{,}% Citation support using natbib.sty

%%% Project macros
\newcommand{\sys}{GeoSemantics}
\newcommand{\patat}[1]{P@#1}

\graphicspath{{figs/}{paper_figures/}{validation_results/}{baseline_comparison/}}

\begin{document}

\copyrightyear{2026}
\datereceived{ xx xx 2026}
\dateaccepted{ xx xx 2026}

\title{GeoSemantics: An Interactive Engine for Explainable, Multi-Scale Place Character from Heterogeneous OpenStreetMap Graphs}

\author[a,b]{Angeliki Grammatikaki}
\author[a,b]{Milena Vuckovic}
\author[a]{Manuela Waldner}
\affil[a]{Institute of Visual Computing and Human-Centered Technology, TU Wien, Vienna, Austria}
\affil[b]{VRVis GmbH, Vienna, Austria}

\thanks{\textbf{CONTACT:} Angeliki Grammatikaki \quad \Letter : angeliki.grammatikaki@tuwien.ac.at}

\keywords{GeoAI; graph attention networks; place embedding; OpenStreetMap; contrastive learning; explainable AI; interactive cartography}

\Abstract{Understanding the functional character of a place --- what distinguishes a historic market square from a residential suburb, or a mountain trailhead from a logistics zone --- remains a core challenge in urban analytics, traditionally requiring data that is proprietary, visually acquired, or hand-labelled. We present a self-supervised system that learns interpretable place representations exclusively from OpenStreetMap by encoding each location as a heterogeneous spatial graph at three radii (200m, 700m, 2km), processed by a heterogeneous Graph Attention Network (GATv2) trained with InfoNCE contrastive learning. The study evaluates representations on a 305-location Austrian benchmark (92 expert-curated and 213 synthetic locations) spanning ten place archetypes within Austria. Results show that the heterogeneous architecture (V3) improves embedding separability by 0.113 and Silhouette Score by 0.037 over the homogeneous baseline (V2) --- the embedding-space geometry that underlies the system's clustering, semantic mapping, and character-layer visualisations. A simple co-occurrence baseline (Place2Vec) achieves comparable nearest-neighbour retrieval, and this result is discussed in depth: the graph's contribution lies in cluster geometry rather than in point-to-point retrieval, and we connect this distinction explicitly to the interactive engine's dependence on coherent class structure. The trained models are accessed through an interpretable and queryable interactive engine that explains predictions in natural language, visualises model attention, and supports counterfactual ``what-if'' simulations in under a second. System evaluation is scoped to Austria; cross-country transfer is treated as an exploratory case study and not a primary claim.}

\newgeometry{top=20.0mm,bottom=40.0mm,left=15mm,right=15mm,headsep=8mm}
\maketitle
\restoregeometry
\newpage

\section{Introduction}

Place character --- the qualitative sense of what a location is, beyond zoning code \citep{tuan1977space} --- shapes critical decisions in urban planning, retail siting \citep{montgomery1998making}, and everyday wayfinding \citep{lynch1960image}. Tourism and heritage applications are natural use cases, though we note that two of these archetypes (Tourism hotspot, Heritage area) currently score 0\% dominant-dimension accuracy due to an OSM tag-vocabulary mismatch described in Section~\ref{sec:tradeoff}; a category remapping fix is identified as future work.
While humans intuitively grasp the character of a place (e.g. instantly distinguishing a bustling, historic market square from a quiet residential suburb) computational systems given only a geographic coordinate or a postal code remain blind to this functional reality. This gap is particularly important in frequently neglected rural, alpine, and peri-urban areas. In these locations the OSM data signal is weaker --- fewer tagged features, lower tagging density --- yet the functional differences between a trailhead and a logistics zone are just as pronounced as those between two city blocks.

To bridge this gap, modern computational systems model place character using auxiliary sources such as mobility traces~\citep{place2vec}, street-level imagery~\citep{urban2vec}, or satellite imagery \citep{tile2vec}. However, these sources are proprietary, privacy-sensitive, and unevenly distributed, failing outside instrumented city cores \citep{Fonte-2017}. Mobility traces and check-in logs \citep{cheng2011exploring} are proprietary, privacy-sensitive, and sparse outside metropolitan cores, meaning they fail in rural, peri-urban, or developing areas. Street-level and satellite imagery captures visual style rather than functional character (e.g. confusing a public café and a private office inside identical buildings) \citep{10.1145/2830541}, whilst incurring significant acquisition and processing costs. Manual annotations also age quickly and struggle to transfer across cultural regions.

OpenStreetMap (OSM) provides a compelling alternative \citep{10.1109/MPRV.2008.80}. It is global, free, community-updated \citep{goodchild2007citizens}, and already encodes a rich, structured vocabulary of POIs, transport infrastructure, and natural features. The open research question is whether this topological vocabulary, structured as a spatial graph, is sufficient to learn a representation of place character without auxiliary data or labels, and whether that representation can be made legible to a human user rather than remaining an opaque vector.

We present \sys, a self-supervised spatial engine that constructs multi-scale local graphs from raw OSM features around any coordinate and encodes them via a heterogeneous Graph Attention Network (GATv2) \citep{brody2022attentivegraphattentionnetworks}. Trained purely via contrastive learning \citep{DBLP:journals/corr/abs-1807-03748}, the system produces dense 64-dimensional location embeddings for similarity retrieval and 7-dimensional interpretable character fingerprints derived by a decoupled saliency network, served through a real-time web application.

Our primary contributions are:
\begin{enumerate}
\item \emph{Heterogeneous multi-scale graph construction}: We extend a GATv2 backbone \citep{brody2022attentivegraphattentionnetworks} with a five-class node-typing scheme (POI, Transport, Natural, Built, Place) and bearing-aware edge encodings across three spatial scales (200m, 700m, 2km), enabling the GNN to distinguish map elements and their spatial relationships at block, neighbourhood, and district level.
\item \emph{Self-supervised training with decoupled saliency readout}: We train the GNN via InfoNCE contrastive learning under node-masking, requiring no labels. A lightweight saliency network decodes location embeddings into a 7-dimensional character fingerprint. The fingerprint's interpretability is deliberately decoupled so that the readout layer can be updated independently of the GNN.
\item \emph{Interpretable, interactive reasoning engine}: The models are served through a real-time application that exposes reasoning at three levels: natural-language explanations; saliency-thresholded attention overlays and precomputed map layers; and a counterfactual sandbox that reruns inference on user-modified graphs in under a second.
\end{enumerate}

\section{Related Work} \label{sec:related}

\textbf{Place Representation and Urban Embeddings.} Place2Vec~\citep{place2vec} learns POI co-occurrence embeddings from check-in sequences, capturing behavioral signals but failing outside wealthy, instrumented urban cores. Imagery-based approaches like Urban2Vec~\citep{urban2vec} and Tile2Vec~\citep{tile2vec} utilize street-level or satellite imagery, inheriting high acquisition costs and capturing visual identity rather than functional character. Structure-only approaches operate on road networks or POI proximity graphs. Space syntax~\citep{hillier1984social} computes node salience from street topology. DeepWalk~\citep{Perozzi_2014} and Node2Vec~\citep{grover2016node2vecscalablefeaturelearning} apply random-walk skip-gram to homogeneous proximity graphs but cannot distinguish typed relationships (e.g. café-to-road vs. café-to-café) --- a structural limitation our typed edge features address.

Feature salience connects our work to landmark models incorporating visual, semantic, and structural aspects~\citep{Caduff, 10.1007/3-540-45799-2_17}. We approximate these by weighting OSM features using distance centrality, category uniqueness, and degree. While landmark studies focus on urban environments where buildings dominate, few studies consider salience in rural or alpine settings, precisely the regimes in which our heterogeneous graph yields its largest gains.

\textbf{Graph Neural Networks for Spatial Data.} Heterogeneous GNNs like HAN~\citep{wang2021heterogeneousgraphattentionnetwork} and HGT~\citep{hu2020heterogeneousgraphtransformer} distinguish node and edge types but are designed for knowledge bases. Geospatial data introduces distinct challenges: taxonomies must fit OSM tagging schemas, edges must encode continuous spatial relationships (bearings, distances), and pooling must handle extreme class imbalances (Transport nodes constitute $\approx$40\% of the corpus). Our GATv2 backbone~\citep{brody2022attentivegraphattentionnetworks} addresses these through per-type embedding tables, type tokens, and stratified gated pooling.

We apply contrastive learning \citep{DBLP:journals/corr/abs-2002-05709} with node-level masking as our data augmentation \citep{hou2022graphmaeselfsupervisedmaskedgraph}. Because POI omission is common in OSM, training the GNN to be invariant to 20\% random node masking improves representation robustness under data-sparsity regimes typical of rural and historical regions \citep{Fonte-2017}.

\section{Methodology}
\label{sec:method}

\sys{} constructs typed, multi-scale spatial graphs from raw OSM features, processes them using a heterogeneous graph attention network, and exposes the predictions in an interactive engine.

\subsection{Data and Feature Encoding}
\label{sec:data}

The dataset is a static extract of Austrian OSM point features from the Geofabrik national dump~\citep{geofabrik2024}, consisting of $\mathbf{2{,}453{,}475}$ nodes after geometry simplification to points. Each node in our extract carries up to four dedicated semantic columns (\texttt{highway}, \texttt{natural}, \texttt{place}, \texttt{man\_made}) alongside an \texttt{other\_tags} field holding all remaining key--value pairs. A primary category is derived from the first non-empty dedicated column; the \texttt{other\_tags} field is then parsed for amenity, shop, tourism, and historic keys as a secondary pass. This yields 222 distinct primary categories across the full corpus. Each node also receives a 32-dimensional bag-of-keys feature vector, where each dimension indicates the presence or absence of a particular tag key in \texttt{other\_tags}. Tag values are excluded to prevent the model from overfitting to named entities.

Every node is assigned one of five semantic types by applying a deterministic, priority-ordered rule: POI $>$ Transport $>$ Natural $>$ Built $>$ Place. The priority order prevents ambiguity when a node carries signals for more than one type. Each type maps to a distinct set of OSM keys, as listed in Table~\ref{tab:nodetypes}. Transport nodes dominate the corpus at $\approx$40\% because every highway segment generates a node, railway, and public transport points; Natural nodes are sparse at $\approx$8\%, with systematically lower density in urban zones.

\begin{table}[htbp]
\centering
\begin{tabularx}{\textwidth}{lXr}
\toprule
\textbf{Type} & \textbf{OSM signal} & \textbf{Approx.\ \%} \\
\midrule
POI (0)       & amenity, shop, tourism, leisure, historic, healthcare & 12 \\
Transport (1) & \texttt{highway} column; railway, public\_transport & 40 \\
Natural (2)   & \texttt{natural} column; waterway, natural landuse & 8 \\
Built (3)     & building, man\_made, power, built landuse & 25 \\
Place (4)     & \texttt{place} column; remaining landuse, barrier & 15 \\
\bottomrule
\end{tabularx}
\caption{Node type classification scheme. Priority order prevents ambiguity.}
\label{tab:nodetypes}
\end{table}

For any query coordinate, three directed $k$-NN scale graphs ($k=8$) are built at 200m (block), 700m (neighborhood), and 2km (district). Edges encode geodetic distance expanded into sinusoidal harmonics~\citep{vaswani2017attention} and bearing decomposition $(\sin\theta_{ij}, \cos\theta_{ij})$, making the model orientation-aware: a café adjacent to a road to its north receives different attention weights than the same café adjacent to a road to its west. A binary flag indicating whether source and target share the same primary category completes the edge feature vector.

\subsection{Model Architecture and Training}
\label{sec:architecture}

We compare a homogeneous baseline (V2) with a heterogeneous extension (V3). Both generate 64-dimensional embeddings and 3-dimensional scale-attention weights.

\textbf{V2} processes all nodes uniformly. Category embeddings are combined with coordinates, and each scale is processed via two GATv2 layers~\citep{brody2022attentive} with learnable scale tokens, using gated mean pooling and cross-scale attention.

\textbf{V3} uses independent category embedding tables for the five node types, learnable segment-type tokens $\mathbf{T} \in \mathbb{R}^{5 \times 64}$ --- one row per node type --- analogous to segment-type tokens in BERT, which conditions the graph attention mechanism on node type. For pooling, global mean pooling is replaced by per-type gated aggregation:
\begin{equation}
\mathbf{p}_t = \frac{\sum_{i:\tau_i=t} \sigma(\mathbf{W}_g \mathbf{x}_i) \odot \mathbf{x}_i}{\sum_{i:\tau_i=t} \sigma(\mathbf{W}_g \mathbf{x}_i)}, \quad t \in \{0,\ldots,4\},
\label{eq:typepool}
\end{equation}
fused through an MLP-driven weighted sum:
\begin{equation}
\mathbf{w} = \mathrm{Softmax}(\mathrm{MLP}([\mathbf{p}_0 \Vert \cdots \Vert \mathbf{p}_4])), \quad \mathbf{h} = \sum_{t=0}^{4} w_t \mathbf{p}_t.
\label{eq:weightedpool}
\end{equation}
This prevents Transport nodes ($\approx$40\%) from drowning out sparse Natural nodes ($\approx$8\%). Trainable parameters total $\approx$43{,}500.

\textbf{Training.} We optimize NT-Xent contrastive loss over positive pairs created by masking node categories and features with $p=0.20$:
\begin{equation}
\mathcal{L} = \frac{1}{2}\!\left[ \mathcal{H}(\mathbf{z}_1 \mathbf{z}_2^\top / \tau,\,\mathbf{I}) + \mathcal{H}(\mathbf{z}_2 \mathbf{z}_1^\top / \tau,\,\mathbf{I}) \right],
\label{eq:infonce}
\end{equation}
with temperature $\tau=0.07$. V3 is trained on $N=500$ locations (batch size 32) using Adam ($10^{-3}$ learning rate) for up to 150 epochs. For each of $N$ training locations, two augmented views of the local graph are generated by masking each node with probability $p=0.20$. Masked nodes have their features zeroed and category replaced by a mask token, mimicking incomplete OSM tagging. The two views of the same location serve as positive pairs, while other batch locations act as negatives. Early stopping is applied after 15 epochs without improvement.

\textbf{Saliency readouts.} A decoupled, two-layer GCN GNN approximates a heuristic salience formula (50% distance centrality, 30% category uniqueness, 20% node degree). Saliency scores are used to construct a seven-dimensional character fingerprint by accumulating saliency-weighted category counts:
\begin{equation}
\mathbf{c}[d] = \frac{\sum_{i:\mathrm{dim}(c_i)=d} w_i}{\sum_i w_i}, \quad w_i = 0.7\,\tilde{s}_i + 0.3,
\label{eq:fingerprint}
\end{equation}
where $\tilde{s}_i$ is the normalized GCN saliency score. The fingerprint is L1-normalized, feeding a rule-based natural-language generator. The floor weight of 0.3 ensures even low-saliency nodes contribute, preventing a single dominant node from monopolizing the reading for a neighborhood.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=6mm,
  box/.style={draw, rounded corners, minimum width=2.3cm,
              minimum height=0.8cm, align=center, fill=blue!6, font=\scriptsize},
  arr/.style={-{Stealth[length=2mm]}, thick}
 ]
\node[box] (osm)  {OSM Tag Data};
\node[box, right=of osm]  (graph) {Multi-scale Graphs\\(200m/700m/2km)};
\node[box, right=of graph] (edge)  {Sinusoidal Edge\\Encoder};
\node[box, below=of edge]  (gat)   {GATv2 Stack\\(Typed, V3)};
\node[box, left=of gat]    (pool)  {Type-Stratified\\Gated Pooling};
\node[box, left=of pool]   (cross) {Cross-Scale\\Attention};
\node[box, below=of cross] (emb)   {64-d Embedding};
\node[box, right=of emb]   (info)  {InfoNCE Loss};
\node[box, right=of info]  (char)  {7-d Saliency\\Fingerprint};
\draw[arr] (osm) -- (graph);
\draw[arr] (graph) -- (edge);
\draw[arr] (edge) -- (gat);
\draw[arr] (gat) -- (pool);
\draw[arr] (pool) -- (cross);
\draw[arr] (cross) -- (emb);
\draw[arr] (emb) -- (info);
\draw[arr] (emb) -- (char);
\end{tikzpicture}
\caption{End-to-end pipeline.}
\label{fig:architecture}
\end{figure}

\subsection{Interactive Engine}
\label{sec:system}

The trained models are served through a web application backed by a precomputed spatial index of 2.4 million OSM nodes, supporting sub-second queries on commodity CPU hardware. The application is organized around three interpretability layers and four additional interaction modes:
\begin{itemize}
\item \emph{Point Lens (Figure~\ref{fig:lens})}: Clicking any map location triggers a GNN query that retrieves all OSM nodes within the three query radii (200m, 700m, 2km), constructs local graphs, runs inference, and renders results. The panel shows the 7D fingerprint as a bar chart, the three scale-attention weights, and a natural-language reading. The natural-language reading names the specific OSM features driving the prediction, preferring real names over generic labels. For Hallstatt, for example: \emph{``47.5622, 13.6493 reads as a Tourist Destination -- predominantly Urban (45\%) with a secondary Tourism presence (31\%). Anchored by Hotel Hallstatt...''} A confidence caveat is appended when OSM coverage is sparse. An Influential POIs panel lists the highest-saliency features and supports multi-select category filtering, so a user can, for instance, isolate only the heritage-tagged features behind a given reading. The text template selects the top 3 categories by saliency weight and formats them dynamically.
\item \emph{Attention Overlay (Figure~\ref{fig:attn})}: Visualizes GATv2 attention weights from the second message-passing layer as directed polylines linking POIs, thickness indicating magnitude. Edges are colored by source node type and scaled in thickness. Only edges whose endpoints exceed a saliency threshold are drawn, and self-loop attention is excluded.
\item \emph{Continuous Semantic Layer (Figure~\ref{fig:charlayer})}: Interpolates a $100\times100$ precomputed grid of location fingerprints via Shepard's method ($k=25$ neighbors, power $1.6$), rendering viewport-adaptive character surfaces. Grid cells are precomputed at 700m spacing across Austria (3,729 valid cells). Gaussian blur suppresses block speckle, and per-dimension weights are raised to a power before color blending to avoid a grey wash-out effect.
\item \emph{Embedding Landscape}: Renders a UMAP projection of all 7,627 locations, supporting comparison and morphing. Morph mode linearly interpolates between the embeddings of two selected locations in the 64-dimensional space, projecting back to character space at each step.
\item \emph{Sandbox (Figure~\ref{fig:sandbox})}: A counterfactual canvas where users brush to add synthetic POIs or remove existing nodes, triggering immediate re-inference. A rectangle brush distributes a user-specified number of synthetic POIs. Adding 8 synthetic POIs to Hallstatt shifts the Tourism dimension by +2\% (embedding shift 0.16) due to the dense existing context.
\item \emph{Time Machine (Figure~\ref{fig:history})}: Issues Overpass API attic queries to extract real historical graph snapshots (e.g. 2010 vs present). Wiener snapshots return 1,062 nodes in 2010 vs 27,072 today, showing crowd-sourced completeness growth.
\end{itemize}

\begin{figure}[htbp]
\centering
\subfigure[]{\includegraphics[width=0.78\textwidth]{figs/fig_overview3.jpg}}
\hfill
\subfigure[]{\includegraphics[width=0.18\textwidth]{figs/fig_overview2.png}}
\caption{Point Lens on Hallstatt (47.5608, 13.6477).}
\label{fig:lens}
\end{figure}

\begin{figure}[htbp]
\centering
\subfigure[]{\includegraphics[width=0.48\textwidth]{figs/attention_graph_v2.png}}
\hfill
\subfigure[]{\includegraphics[width=0.48\textwidth]{figs/attention_graph_v3.png}}
\caption{Attention overlay for Hallstatt (a) V2 (no edges) vs (b) V3 (type-colored polylines).}
\label{fig:attn}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figs/fig_charlayer.png}
\caption{Continuous semantic map layer.}
\label{fig:charlayer}
\end{figure}

\begin{figure}[htbp]
\centering
\subfigure[]{\includegraphics[width=0.58\textwidth]{figs/sandbox_all.png}}
\hfill
\subfigure[]{\includegraphics[width=0.39\textwidth]{figs/hallstatt_umap.png}}
\caption{Interaction modes: (a) Sandbox edits, (b) Semantic landscape UMAP.}
\label{fig:sandbox}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figs/hallstatt_2014_2026_historic_1.png}
\caption{Time Machine: Hallstatt character drift from 2014 (left) to present (right).}
\label{fig:history}
\end{figure}

\section{Results}
\label{sec:eval}

\subsection{Benchmark and Metrics}
\label{sec:benchmark}

\label{sec:tradeoff}

We evaluate representations on a 305-location Austrian benchmark spanning ten place archetypes. The benchmark consists of:
\begin{enumerate}
\item \emph{Curated Dataset (92 locations)}: locations manually selected across Austria where the ground truth classification was established by joint inspection of OSM tags, Austrian GIS.at administrative geodata, UNWTO tourism classifications, and the BDA register. The curated locations represent high-certainty anchor points to validate fine-grained representations.
\item \emph{Synthetic Dataset (213 locations)}: locations added to scale the evaluation and stabilise per-class metrics, ensuring statistical robustness across sparse regions. Coordinates were sampled from Austria's precomputed place database. To filter out noise, we required selected locations to have at least three OSM features within a 500m radius. Labels were assigned using a rule-based saliency formula with a 0.50 dominant-dimension threshold.
\end{enumerate}
Unsupervised metrics are: Retrieval $P@3$, Silhouette Score, and Embedding Separability ($\mathrm{sep} = \mathrm{intra} / (\mathrm{intra} + \mathrm{inter})$).

\subsection{Main Results and Baselines}
\label{sec:mainresults}

We compare V2 and V3 against three prior-work baselines (Place2Vec, Tile2Vec, Urban2Vec) and a TF-IDF bag-of-categories model.

\begin{table}[htbp]
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc}
\toprule
\textbf{Model} & \textbf{Retrieval \patat{3}} & \textbf{Silhouette Score} & \textbf{Separability} \\
\midrule
V2 --- homogeneous GATv2   & 0.443 & -0.067 & 0.582 \\
V3 --- heterogeneous GATv2 & \textbf{0.457} & \textbf{-0.030} & \textbf{0.695} \\
TF-IDF (no graph)          & 0.320 & -0.114 & 0.485 \\
\midrule
V3 gain over V2            & $+$0.014 & $+$0.037 & $+$0.113 \\
\bottomrule
\end{tabular*}
\caption{Benchmark results on the 305-location Austrian benchmark.}
\label{tab:benchmark}
\end{table}

V3 leads on all metrics: retrieval $P@3$ improves by 1.4\,pp, separability improves by 0.113, and Silhouette Score improves by 0.037. The negative Silhouette scores arise from class-size imbalance (Urban and Nature constitute 54\% of locations), pushing mean inter-class similarity high because dominant classes are present in every pairwise inter-class computation. The Silhouette metric's sensitivity to class-count imbalance is well documented~\citep{10.1007/11496168_1}; the embedding separability metric --- which computes intra- vs.\ inter-class cosine similarities separately and symmetrically --- is more appropriate here.

The Rural/Alpine-focused subset confirms V3's structural advantage in sparse geometries. When restricting to the combined Alpine/nature and Rural/agricultural classes, V3's embeddings capture the sparse natural features, producing a massive separability boost. This indicates that V3's multi-scale heterogeneous architecture successfully extracts meaningful topological signals even when absolute POI density is low.

\subsection{Comparison to Prior-Work Baselines}
\label{sec:baselines}

We compare our V2 and V3 models alongside three re-implemented prior-work baselines (Place2Vec, Tile2Vec, and Urban2Vec) with identical retrieval-P@3 and separability functions on the Austrian benchmark. Place2Vec uses a co-occurrence window size of 5; Tile2Vec learns triplets from orthophotos; and Urban2Vec fuses ResNet-50 street-view image features with a GCN.

V3 achieves the highest separability (0.680) compared to Urban2Vec (0.548) and Tile2Vec (0.522). On retrieval, visual models lead due to image features (Urban2Vec: 0.762, Tile2Vec: 0.649), compared to V3's 0.637. However, V3 outperforms the non-visual Place2Vec baseline (0.523 P@3, 0.497 separability) and V2.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figs/master_figure.png}
\caption{GeoSemantics V3 Performance Evaluation.}
\label{fig:master}
\end{figure}

\subsection{Per-Class Results}
\label{sec:perclass}

As shown in Figure~\ref{fig:master}B, V3 improves class boundaries in 8 out of 10 categories, with major gains in sparse Alpine/nature (+28.7%) and Rural/agricultural (+24.0%) environments, as well as Peri-urban fringes (+18.9%). This highlights the advantage of multi-scale heterogeneous pooling: in sparse regions where POIs are scattered across thousands of meters, V3 successfully aggregates distant topological signals to form a coherent geographic fingerprint, whereas V2 lacks this heterogeneous capacity.

\subsection{Component Ablation}
\label{sec:ablation}

Removing node-type identity tokens and single-scale restrictions decreases separability by 0.184 each (Figure~\ref{fig:master}C), with Transport-node removal close behind ($-$0.120), confirming that type-conditioned attention and multi-scale context are load-bearing components. Built-node ablation causes a complete structural collapse (separability to 0.000), proving built infrastructure anchors the embedding space.

\subsection{Temporal Evaluation with Historical OSM Data}
\label{sec:temporal}

Rewinding Hallstatt to 2014 (27 POIs) via attic queries shows it classified as "Urban Center", shifting to "Tourist Destination" by the present (186 POIs, drift 0.63) as crowd-sourced completeness grew. A methodological caveat is identified: naively-read ``drift'' conflates genuine urban change with crowd-sourced completeness growth, which we surface explicitly rather than presenting historical trajectories as clean urban-change signals.

\subsection{Counterfactual Simulation}
\label{sec:counterfactual}

Adding 8 synthetic POIs to Hallstatt (203 existing POIs) shifts Tourism by +2\% (embedding shift 0.16). In the sparse Grossglockner alpine area (11 existing POIs), the same edit flips the label from "Nature" to "Tourist Destination" with a shift of 1.01 --- more than thirty times larger than the Hallstatt edit despite fewer nodes added.

\section{Discussion and Limitations}
\label{sec:discussion}

Visual models (Urban2Vec, Tile2Vec) achieve higher point-to-point retrieval precision ($P@3$ of 0.762 and 0.649) than V3 (0.637), but V3 leads in global embedding separability (0.680 vs. 0.548 and 0.522). This separability is crucial for the interactive spatial engine: low-separability models produce a chaotic "grey wash" under Shepard interpolation and overlapping clusters in the UMAP landscape. Furthermore, visual models degrade in sparse Alpine and Rural regions where Street View is absent; here, V3's multi-scale heterogeneous pooling successfully extracts topological signals.

The saliency network is trained decoupled from contrastive learning to ensure stable interpretability. The formula's weights (50\% distance centrality, 30\% category uniqueness, 20\% connectivity) follow landmark-salience conventions~\citep{Caduff}. An OSM tag-vocabulary mismatch --- where commercial tourism fabric carries generic tags --- causes heuristics to misclassify tourist zones. V3's unsupervised embeddings resolve this, learning the topological arrangement of tourist areas.

The TF-IDF baseline is degenerate at sparse sites due to insufficient POI density, demonstrating the sparsity challenge \sys{} handles. Zero accuracy on Tourism and Heritage archetypes stems from the tag mismatch: OSM tags like shop and amenity do not distinguish high-end tourism from local infrastructure, forcing a category-based model to misinterpret place character. The Leave-One-Out probe collapses onto majority classes due to class imbalance. V3's training size was half of V2's ($N=500$ vs. $1{,}000$), yet V3 still achieves a 0.113 separability gain. Finally, cross-country transfer is limited by regional tagging conventions and localized feature vocabularies, and the explanation generator is template-based rather than utilizing a large language model.

\section{Conclusion}
\label{sec:conclusion}

\sys{} learns interpretable place character representations from OSM tag data alone. On a 305-location benchmark, the heterogeneous V3 model improves separability by 0.113 over V2, with the largest gains in sparse Rural/Alpine zones. The interactive engine exposes these representations via sub-second point readings, attention graphs, and counterfactual sandboxing. Future work will expand multi-country transfer across other European countries, close tag-vocabulary gaps in the OpenStreetMap schema, and conduct expert user evaluations to assess the real-world utility of the system.

\bibliographystyle{tfcad}
\bibliography{ref}
\end{document}
"""

if __name__ == '__main__':
    with open('scratch/condensed_results.tex', 'w', encoding='utf-8') as f:
        f.write(PROPOSED_LATEX)
    words = condense.count_words(PROPOSED_LATEX)
    print("Proposed text clean main body prose word count:", len(words))
