################################################################
#
# MODULES
#
################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

import os
import sys
import glob
import random
#import time
from timeit import default_timer as timer
import argparse
import dill as pickle

import torch
import torch.utils.data as data
from torch import nn, optim
from torch.nn import functional as F

from reduction_leaky import reduction
from utils import debuginfoStr, build_data_generic, build_graph_generic, relation_name
from sub_sample import sub_sample_generic
from pyHGT.model import GNN, GNN_from_raw
from omics_manifest import resolve_manifest, ManifestError
import itertools

from warnings import filterwarnings
filterwarnings("ignore")

import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix, roc_curve,
                              precision_recall_fscore_support, silhouette_score)



################################################################
#
# PUBLICATION STYLE (shared across every figure in this script)
#
################################################################
# One place to control the look of every plot, so the whole output bundle reads as
# a single, coherent figure set rather than 8 differently-styled ad-hoc charts.
# Colorblind-safe choices throughout (Okabe-Ito qualitative palette; viridis for
# sequential/continuous scales; RdBu_r for diverging/signed scales).

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # keep text as editable glyphs (Illustrator/Inkscape), not outlines
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Okabe-Ito: the standard colorblind-safe qualitative palette recommended for
# scientific figures. Used for CONDITION_COLORS (built once metadata is loaded)
# and anywhere else a small categorical legend is needed.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

SEQ_CMAP = "viridis"         # every continuous/sequential scale (attention, activity, distance)
DIV_CMAP = "RdBu_r"          # every diverging/signed scale (blue=low/negative, red=high/positive)

# Matplotlib marker cycle for an arbitrary number of omics (PRAMIGO's original script
# hardcoded exactly 2: circle for genes, diamond for metabolites). Cycles if there are
# more omics than markers - readability of a many-omic figure is a separate, real
# concern (see design notes) that a bigger marker alphabet alone doesn't solve.
OMIC_MARKERS = ['o', 'D', 's', '^', 'v', 'P', 'X', '*']
# Plotly symbol cycle for the interactive network plot (network_interactive.html).
# 'circle' is reserved for sample nodes so it's deliberately excluded here.
OMIC_PLOTLY_SHAPES = ['triangle-up', 'square', 'diamond', 'star', 'pentagon', 'hexagon2', 'cross', 'x']


def get_omic_palette(omic_names):
    """Deterministic omic-name -> color mapping, generalizing the original script's
    fixed OMIC1_COLOR/OMIC2_COLOR constants to an arbitrary number of omics. Cycles
    through OKABE_ITO minus black (reserved for text/outlines) - the same palette
    CONDITION_COLORS draws from, which is fine since omic identity and condition are
    never colored in the same legend at once."""
    palette = OKABE_ITO[:-1]
    return {name: palette[i % len(palette)] for i, name in enumerate(omic_names)}


def get_omic_markers(omic_names):
    return {name: OMIC_MARKERS[i % len(OMIC_MARKERS)] for i, name in enumerate(omic_names)}


def get_omic_plotly_shapes(omic_names):
    return {name: OMIC_PLOTLY_SHAPES[i % len(OMIC_PLOTLY_SHAPES)] for i, name in enumerate(omic_names)}


def hex_to_rgb_string(hex_color):
    """'#0072B2' -> 'rgb(0, 114, 178)', for Plotly marker/line color specs."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgb({r}, {g}, {b})'


def get_condition_palette(condition_values):
    """Deterministic condition -> color mapping shared by every plot that colors
    samples by CONDITION, so the same condition is always the same color across
    the whole figure bundle."""
    uniq = sorted(pd.unique(pd.Series(condition_values).astype(str)))
    return {cond: OKABE_ITO[i % len(OKABE_ITO)] for i, cond in enumerate(uniq)}


def robust_layout(graph, seed=42, scale=1.0, gap=2.5):
    """Kamada-Kawai layout that stays well-behaved on graphs with isolated nodes
    or several disconnected components - both are routine here (a correlation/
    attention-filtered feature graph regularly leaves nodes with no same-type
    partner). Calling nx.kamada_kawai_layout on the *whole* graph at once makes
    every node pair - including pairs sitting in different components - part of
    the same distance matrix; components (or lone isolated nodes) end up placed
    an enormous, disproportionate distance from everything else, so the actual
    network collapses into a speck in one corner of an otherwise blank canvas.
    Laying out each connected component on its own (where Kamada-Kawai only
    ever sees reachable, finite-distance pairs) and then tiling the components
    on a simple grid keeps every component's true internal geometry while
    making it impossible for cross-component "distance" to warp the layout.

    Also corrects a second, more subtle issue: everywhere else in this script,
    a bigger 'weight' means "more related" (more attention, stronger link). But
    nx.kamada_kawai_layout's `weight` parameter is a target *distance* - bigger
    weight pushes a pair *further apart*, not closer. Passed through unchanged,
    the single strongest edge in the graph ends up stretched the *furthest*
    apart of all, which then forces every properly-weaker-and-closer pair to be
    rescaled down into an indistinguishable clump next to it. So: normalize the
    raw weights once across the whole graph, then invert them into a bounded
    'kk_distance' edge attribute (high attention -> short distance) before
    laying anything out.
    """
    if graph.number_of_nodes() == 0:
        return {}

    raw_weights = np.array([d.get('weight', 1.0) for _, _, d in graph.edges(data=True)])
    if len(raw_weights):
        wmin, wmax = raw_weights.min(), raw_weights.max()
        for _, _, d in graph.edges(data=True):
            attraction = 0.1 + 0.9 * ((d.get('weight', 1.0) - wmin) / (wmax - wmin) if wmax > wmin else 0.5)
            d['attraction'] = attraction        # spring_layout: bigger -> closer (already correct semantics)
            d['kk_distance'] = 1.1 - attraction  # kamada_kawai: bigger -> further apart, so invert; bounded [0.1, 1.0]

    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    components.sort(key=lambda c: c.number_of_nodes(), reverse=True)

    def _layout_component(comp):
        if comp.number_of_nodes() == 1:
            return {next(iter(comp.nodes())): np.array([0.0, 0.0])}
        try:
            p = nx.kamada_kawai_layout(comp, weight='kk_distance')
        except Exception:
            p = nx.spring_layout(comp, seed=seed, k=None, iterations=300, weight='attraction')
        xs = np.array([v[0] for v in p.values()])
        ys = np.array([v[1] for v in p.values()])
        cx, cy = xs.mean(), ys.mean()
        spread = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
        return {n: np.array([(x - cx) / spread * scale, (y - cy) / spread * scale])
                for n, (x, y) in p.items()}

    n_comp = len(components)
    grid_cols = max(1, int(np.ceil(np.sqrt(n_comp))))
    pos = {}
    for idx, comp in enumerate(components):
        comp_pos = _layout_component(comp)
        row, col = divmod(idx, grid_cols)
        offset = np.array([col * gap * scale, -row * gap * scale])
        for node, xy in comp_pos.items():
            pos[node] = xy + offset
    return pos


def _frame_axes_to_pos(ax, pos, pad_fraction=0.08):
    """Explicitly set axis limits from the actual node coordinates (plus a
    small padding margin), instead of relying on matplotlib's autoscale after
    nx.draw_networkx_* calls. This guarantees the saved figure always frames
    exactly the plotted network - no leftover blank canvas and no content
    pushed into a corner - regardless of the layout's absolute coordinate
    scale."""
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    if not xs:
        return
    x_span = max(max(xs) - min(xs), 1e-6)
    y_span = max(max(ys) - min(ys), 1e-6)
    x_pad = x_span * pad_fraction
    y_pad = y_span * pad_fraction
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.set_aspect("equal")


################################################################
#
# DEEPMAPS PREPARATION FOR HGT PERFORMANCE
#
################################################################

# Protection stocastic procedures
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('cuda version: ',torch.version.cuda) # Checking for cuda version

# Arguments
parser = argparse.ArgumentParser(description='Training GNN on gene cell graph')
parser.add_argument('--data_path', type=str)
parser.add_argument('--metadata_path', type=str, required=True,
                    help='Path to the TSV metadata file (must contain "ID" and "CONDITION" columns)')
parser.add_argument('--omics_manifest', type=str, default=None,
                    help='Path to a YAML/JSON manifest listing an arbitrary number of omics '
                         '(see omics_manifest.py / README). Takes precedence over --omic1_path/--omic2_path.')
parser.add_argument('--omic1_path', type=str, default=None,
                    help='[legacy 2-omic mode] Path to the omic1 TSV matrix (features x samples, "ID" column '
                         'first). Ignored if --omics_manifest is given.')
parser.add_argument('--omic2_path', type=str, default=None,
                    help='[legacy 2-omic mode] Path to the omic2 TSV matrix (features x samples, "ID" column '
                         'first). Ignored if --omics_manifest is given.')
parser.add_argument('--topology', type=str, default=None, choices=['full', 'star'],
                    help="'full': every omic pair gets a direct feature-feature correlation graph "
                         "(O(k^2) mask cost, richer cross-omic signal - PRAMIGO's original design). "
                         "'star': omics only ever connect through sample nodes, never directly to each "
                         "other (O(k) mask cost). Overrides the manifest's top-level `topology:` if both "
                         "are given; defaults to 'full' if neither is set.")
parser.add_argument('--sample_mode', type=str, default=None, choices=['intersection', 'union'],
                    help="'intersection': only keep samples present in every omic (complete-case, "
                         "PRAMIGO's original behavior). 'union': keep any sample present in at least one "
                         "omic; a sample missing an omic's data gets no edges of that omic's relation type "
                         "(see README). Overrides the manifest's top-level `sample_mode:` if both are "
                         "given; defaults to 'intersection' if neither is set.")
parser.add_argument('--epoch', type=int, default=100)
# sampling times
parser.add_argument('--n_batch', type=int, default=25,
                    help='Number of batch (sampled graphs) for each epoch')

parser.add_argument('--omic1_rate', type=float, default=1,
                    help='[legacy 2-omic mode] Use `rate:` in --omics_manifest instead when using the manifest.')
parser.add_argument('--omic2_rate', type=float, default=1,
                    help='[legacy 2-omic mode] Use `rate:` in --omics_manifest instead when using the manifest.')

# Result
parser.add_argument('--data_name', type=str,
                    help='The name for dataset')
parser.add_argument('--result_dir', type=str, required=True,
                    help='The address for storing the models and optimization results.')
parser.add_argument('--reduction', type=str, default='AE',
                    help='the method for feature extraction, pca, raw, AE')
parser.add_argument('--in_dim', type=int, default=256,
                    help='Number of hidden dimension (AE)')
parser.add_argument('--omic1_corr_cutoff', type=float, default=0.7,
                    help='[legacy 2-omic mode] Correlation cutoff threshold (0-1) for omic1 feature-feature '
                         'graph edges. Use `corr_cutoff:` in --omics_manifest instead when using the manifest.')
parser.add_argument('--omic2_corr_cutoff', type=float, default=0.7,
                    help='[legacy 2-omic mode] Correlation cutoff threshold (0-1) for omic2 feature-feature '
                         'graph edges. Use `corr_cutoff:` in --omics_manifest instead when using the manifest.')
parser.add_argument('--leiden_resolution', type=float, default=1.5,
                    help='Resolution parameter for Leiden clustering of the top-feature attention subgraph')
parser.add_argument('--leiden_min_cluster_size', type=int, default=3,
                    help='Minimum number of features for a Leiden cluster to be kept in the cluster-network plot')

# GAE
parser.add_argument('--n_hid', type=int,
                    help='Number of hidden dimension (HGT)', default=128)
parser.add_argument('--ae_n_hid', type=int,
                    help='Number of hidden dimension (AE)', default=256)
parser.add_argument('--n_heads', type=int,
                    help='Number of attention head', default=8)
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout ratio')
parser.add_argument('--lr', type=float,
                    help='learning rate', default=0.0001)

parser.add_argument('--batch_size', type=int,
                    help='Number of output nodes for training', default=32)
parser.add_argument('--layer_type', type=str, default='hgt',
                    help='the layer type for GAE')
parser.add_argument('--loss', type=str, default='sup',
                    help='the loss for GAE')
parser.add_argument('--factor', type=float, default='0.5',
                    help='the attenuation factor')
parser.add_argument('--patience', type=int, default=5,
                    help='patience')
parser.add_argument('--rf', type=float, default='0.0',
                    help='the weights of regularization')
parser.add_argument('--cuda', type=int, default=1,
                    help='cuda 0 use GPU0 else cpu ')
parser.add_argument('--rep', type=str, default='T',
                    help='precision truncation')
parser.add_argument('--AEtype', type=int, default=1,
                    help='AEtype:1 embedding node autoencoder 2:HGT node autoencode')
parser.add_argument('--optimizer', type=str, default='adamw',
                    help='optimizer')
parser.add_argument('--resume', action='store_true',
                    help='Resume HGT training from the latest checkpoint in <result_dir>/model/ instead of starting fresh')
parser.add_argument('--checkpoint_every', type=int, default=1,
                    help='Save a resumable training checkpoint every N epochs')

args = parser.parse_args()

# Resolve the arbitrary-length omics list (--omics_manifest, or the legacy
# --omic1_path/--omic2_path pair), plus the --topology/--sample_mode flags, once,
# up front. Everything downstream is driven by OMIC_NAMES/OMICS instead of a fixed
# omic1/omic2 CLI surface - see omics_manifest.py for the manifest format.
try:
    OMICS, TOPOLOGY, SAMPLE_MODE = resolve_manifest(args)
except ManifestError as e:
    parser.error(str(e))
OMIC_NAMES = [o['name'] for o in OMICS]
OMIC_RATE = {o['name']: o['rate'] for o in OMICS}
OMIC_CORR_CUTOFF = {o['name']: o['corr_cutoff'] for o in OMICS}
OMIC_KIND = {o['name']: o['kind'] for o in OMICS}
OMIC_COLORS = get_omic_palette(OMIC_NAMES)
OMIC_MARKERS_BY_NAME = get_omic_markers(OMIC_NAMES)
OMIC_PLOTLY_SHAPES_BY_NAME = get_omic_plotly_shapes(OMIC_NAMES)
print(f"Omics ({len(OMIC_NAMES)}): {OMIC_NAMES} | topology={TOPOLOGY} | sample_mode={SAMPLE_MODE}")

#args.metadata_path = "/home/inigo/Desktop/BiomiX_HGT/EGA/Metadata/EGAS00001001746_metadata_CLL.tsv"
#args.omic1_path = "/home/inigo/Desktop/BiomiX_HGT/EGA_omic1.csv"
#args.omic2_path = "/home/inigo/Desktop/BiomiX_HGT/EGA_omic2.csv"
#args.result_dir = "/home/inigo/Desktop/BiomiX_HGT/EGA_biomix_function"

#gene_cell = minmax_scale(pd.read_csv('AUCs/aHD1.csv').iloc[: , 1:].to_numpy(), axis=0) # Initial adjacency matrix is the one generated in KNN

#print(f'GAS loaded!',gene_cell.shape)
#args.dropout = 0.2
#args.epoch = 100 # Hyperparameter specification
#args.n_hid = 104 # Hyperparameter specification
#args.n_heads = 13 # Hyperparameter specification
#args.lr = 0.001 # Hyperparameter specification
#args.n_batch = 32 # Hyperparameter specification
file0=f'epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_{args.lr}n_batch{args.n_batch}' # Text saving hypermarametrization
print(f'\n{file0}') #Print hyperparametrization
args.metadata_path

#args.result_dir = 'Output_MTB' # Sets parent directory for output
omic_dirs = {name: args.result_dir + f'/{name}/' for name in OMIC_NAMES} # Sets a per-omic output directory
sample_dir = args.result_dir+'/sample/' # Sets directory for sample output
model_dir = args.result_dir+'/model/' # Sets directory for model output
att_dir = args.result_dir+'/att/' # Sets directory for attention output
loss_dir = args.result_dir+'/loss/' # Sets directory for loss output
plots_dir = args.result_dir+'/plots/' # Sets directory for loss output
embbs_dir = args.result_dir+'/embeddings/' # Sets directory for loss output
supplementary_dir = args.result_dir+'/supplementary/' # Optional bonus figures, kept separate from the core plots
os.makedirs(args.result_dir, exist_ok=True)
for _omic_dir in omic_dirs.values():
    os.makedirs(_omic_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(att_dir , exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(embbs_dir, exist_ok=True)
os.makedirs(supplementary_dir, exist_ok=True)

#start_time = time.time() # Initialize time counting
#print('---0:00:00---scRNA starts.') # Prints the initialization 

# Setting working space GPU or CPU
if args.cuda == 0:
    if torch.cuda.is_available():
        device = torch.device("cuda:0") # Sets space to GPU
        print("cuda>>>")
    else:
        print("Warning: --cuda 0 requested GPU0 but no CUDA device is available. Falling back to CPU.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu") # Sets space to CPU

print(device) # Print device. IMPORTANT for incompatibilities.



################################################################
#
# METADATA PROCESSING BEFORE HGT
#
################################################################

#metadata = pd.read_csv("/home/inigo/Desktop/BiomiX_HGT/MTB/Metadata/MTBLS7623_Metadata.tsv", sep="	")
metadata = pd.read_csv(args.metadata_path, sep="	")
#metadata = metadata[metadata["CONDITION"] != "PTB_DM"].reset_index(drop=True)
sample_order = metadata.ID.tolist()
#trans = trans[['ID'] + sample_order]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample input (your real metadata)
labels = metadata['CONDITION']

# Step 1: Label encode to integers
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
# This will map: HC → 0, PTB → 1, PTB_DM → 2

# Step 2: One-hot encode
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(integer_labels.reshape(-1, 1))



################################################################
#
# HGT INFERENCE
#
################################################################

omic_dfs = {}
for o in OMICS:
    df = pd.read_csv(o['path'], sep="\t")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    omic_dfs[o['name']] = df

meta_samples = set(metadata["ID"])
omic_sample_sets = {name: set(df.columns[1:]) for name, df in omic_dfs.items()}  # exclude "ID" column

if SAMPLE_MODE == 'intersection':
    # Complete-case (PRAMIGO's original behavior): only samples present in the
    # metadata AND every single omic survive.
    common_samples = list(meta_samples.intersection(*omic_sample_sets.values()))
    sample_presence = {name: {s: True for s in common_samples} for name in OMIC_NAMES}
else:
    # Partial-coverage: keep any sample present in the metadata AND at least one
    # omic. A sample missing an omic's data gets that omic's feature block
    # zero-filled below purely so every omic matrix keeps one column per
    # common_sample - but critically, the KNN mask that turns into graph edges for
    # that omic<->sample relation is explicitly zeroed out for that sample further
    # down, so the *graph structure* carries the missingness (no edges of that
    # relation type at all), not an imputed value pretending to be real data.
    common_samples = list(meta_samples.intersection(set().union(*omic_sample_sets.values())))
    sample_presence = {name: {s: (s in omic_sample_sets[name]) for s in common_samples} for name in OMIC_NAMES}
    for name in OMIC_NAMES:
        n_missing = sum(1 for present in sample_presence[name].values() if not present)
        if n_missing:
            print(f"[sample_mode=union] {name}: {n_missing}/{len(common_samples)} samples have no "
                  f"data for this omic - zero-filled feature block, no '{name}<->sample' edges for them.")

if len(common_samples) == 0:
    raise SystemExit(
        f"No samples survive metadata + omics ({OMIC_NAMES}) under --sample_mode {SAMPLE_MODE}. "
        f"Check --sample_mode (try 'union' if your omics only partially overlap on samples) and that "
        f"sample IDs match exactly across --metadata_path and every omic file."
    )

metadata = metadata[metadata["ID"].isin(common_samples)].reset_index(drop=True)
metadata = metadata.set_index("ID").loc[common_samples].reset_index()

omic_matrices = {}
for name in OMIC_NAMES:
    df = omic_dfs[name].set_index(omic_dfs[name].columns[0])
    sub = df.reindex(columns=common_samples)
    missing_cols = [s for s in common_samples if not sample_presence[name][s]]
    if missing_cols:
        sub[missing_cols] = 0.0  # neutral placeholder for a sample this omic never measured
    omic_dfs[name] = sub.reset_index()
    omic_matrices[name] = sub.to_numpy()

# Shared across every condition-colored plot in this script (see PUBLICATION STYLE above).
CONDITION_COLORS = get_condition_palette(metadata['CONDITION'])

labels = metadata['CONDITION']
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(integer_labels.reshape(-1, 1))

# Per-omic autoencoder feature reduction - generalizes the original script's two
# hand-written reduction() calls (one per omic) to a loop over however many omics
# the manifest lists.
encoded = {}
for name in OMIC_NAMES:
    enc, enc2, losses, losses2 = reduction(args.reduction, omic_matrices[name], device, args.ae_n_hid)
    encoded[name] = enc

samples_stacked = np.concatenate([omic_matrices[name] for name in OMIC_NAMES], axis=0)
encoded_samples, encoded2_samples, losses, losses2 = reduction(args.reduction, samples_stacked, device, args.ae_n_hid) # Autoencoder for samples

def normalize_embeddings(X):
    # Normalize each row (embedding vector) to zero mean, unit variance
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)


# Concatenate every omic's feature embeddings and normalize jointly - the same
# `combined` step as the original 2-omic script, generalized to however many
# omics there are.
combined = torch.cat([encoded[name] for name in OMIC_NAMES], dim=0)
normalized = normalize_embeddings(combined)
_off = 0
for name in OMIC_NAMES:
    n = encoded[name].shape[0]
    encoded[name] = normalized[_off:_off + n, :]
    _off += n
# NOTE (preserved from original PRAMIGO): only the per-omic *feature* embeddings go
# through this cross-omic normalization pass - `encoded2_samples` (the per-sample
# embedding used below) is intentionally left as reduction()'s raw output, exactly
# as in the original 2-omic biomix_hgt.py (whose analogous `encoded2_sample` slice
# was computed but never fed back into anything downstream). Not "fixed" here,
# since that's a training-affecting numeric behavior outside the scope of this
# generalization.

for name in OMIC_NAMES:
    pd.DataFrame(encoded[name].detach().numpy()).to_csv(os.path.join(embbs_dir, f"ae_{name}_embeddings.csv"))
pd.DataFrame(encoded2_samples.detach().numpy()).to_csv(os.path.join(embbs_dir, "ae_samples_embeddings.csv"))
debuginfoStr('Feature extraction finished') # Print verbose


def compute_binary_correlation_mask(A, B, ae_n_hid, threshold=0.7):
    # Normalize embeddings for Pearson correlation
    A_norm = A
    B_norm = B
    # Pearson correlation = cosine similarity of standardized vectors
    corr = np.dot(A_norm, B_norm.T) / ae_n_hid  # ae_n_hid is the embedding dimension
    return (corr >= threshold).astype(np.uint8)  # Binary mask: 1 if corr >= threshold


# Each omic's own feature-feature correlation graph (self_masks[name]).
self_masks = {}
for name in OMIC_NAMES:
    m = compute_binary_correlation_mask(encoded[name].detach().numpy(), encoded[name].detach().numpy(),
                                         args.ae_n_hid, threshold=OMIC_CORR_CUTOFF[name])
    np.fill_diagonal(m, 0)
    self_masks[name] = m

# Sample-sample edges: kept as an all-zero placeholder relation, exactly like the
# original 2-omic script (no direct sample-sample correlation is computed here).
sample_mask = np.zeros([encoded2_samples.shape[0], encoded2_samples.shape[0]])

# NO MULTIOMIC CORRELATIONS - SEPARATE AEs ARTIFACTS
from sklearn.neighbors import NearestNeighbors

def compute_knn_mask(A, B, k=100, metric='cosine'):
    """
    For each row in A, find k nearest neighbors in B.
    Returns a binary matrix of shape (A.shape[0], B.shape[0]).
    """
    k = max(1, min(k, B.shape[0]))
    neigh = NearestNeighbors(n_neighbors=k, metric=metric)
    neigh.fit(B)
    dists, indices = neigh.kneighbors(A)
    mask = np.zeros((A.shape[0], B.shape[0]), dtype=np.uint8)
    for i, neighbors in enumerate(indices):
        mask[i, neighbors] = 1
    return mask

# omic <-> sample edges: always computed, regardless of --topology.
cross_sample_masks = {}
for name in OMIC_NAMES:
    m = compute_knn_mask(encoded[name].detach().numpy(), encoded2_samples.detach().numpy(),
                          k=round(encoded2_samples.shape[0] / 3))
    np.fill_diagonal(m, 0)
    if SAMPLE_MODE == 'union':
        missing_col_idx = [i for i, s in enumerate(common_samples) if not sample_presence[name][s]]
        if missing_col_idx:
            m[:, missing_col_idx] = 0  # no imputed edges for samples this omic never measured
    cross_sample_masks[name] = m

# omic <-> omic edges: only under --topology full (every pair gets a direct
# feature-feature KNN graph - O(k^2) mask cost, richer cross-omic signal); stays
# empty under --topology star (omics only ever connect through sample nodes -
# O(k) cost, no omic-omic pair is ever computed at all). See README.
cross_omic_masks = {}
if TOPOLOGY == 'full':
    for a, b in itertools.combinations(OMIC_NAMES, 2):
        m = compute_knn_mask(encoded[a].detach().numpy(), encoded[b].detach().numpy(),
                              k=round(encoded[b].shape[0] / 3))
        np.fill_diagonal(m, 0)
        cross_omic_masks[(a, b)] = m

_summary = ", ".join(f"{name} self-connect: {int(np.sum(self_masks[name]))}" for name in OMIC_NAMES)
_summary += f", samples self-connect: {int(np.sum(sample_mask))}"
_summary += ", " + ", ".join(f"{name}-samples connect: {int(np.sum(cross_sample_masks[name]))}" for name in OMIC_NAMES)
if cross_omic_masks:
    _summary += ", " + ", ".join(f"{a}-{b} connect: {int(np.sum(m))}" for (a, b), m in cross_omic_masks.items())
print(_summary)



################################################################
#
# GENERATE HETEROGENOUS GRAPH
#
################################################################

node_features_for_graph = {name: encoded[name].detach().numpy() for name in OMIC_NAMES}
node_features_for_graph['sample'] = encoded2_samples.detach().numpy()
graph = build_graph_generic(node_features_for_graph, cross_sample_masks, self_masks, sample_mask,
                             cross_omic_masks, OMIC_NAMES) # Generate graph based in autoencoder and KNN matrix
debuginfoStr('Build Graph finished') # Print verbose



################################################################
#   
# SUBGRAPHS SAMPLING FOR TRAINING
#
################################################################

#start = timer()
print("Start sampling!")
np.random.seed(seed)
jobs = []
sample_num=round(encoded2_samples.shape[0]/3)
omic_sizes = {name: max(1, int((encoded[name].shape[0]*OMIC_RATE[name])/args.n_batch)) for name in OMIC_NAMES} # Sets the number of features per omic in subgraph
print(f'sample_num: {sample_num}, omic_sizes: {omic_sizes}')
for _ in range(args.n_batch): # Iterates over subsampling
    p = sub_sample_generic(graph,
                            cross_sample_masks,
                            self_masks,
                            sample_mask,
                            cross_omic_masks,
                            OMIC_NAMES,
                            sample_num,
                            omic_sizes,
                            encoded2_samples.shape[0],
                            topology=TOPOLOGY) # Sub-graphs sampling as jobs to learn from in HGT
    jobs.append(p)


print("Sampling end!")
debuginfoStr('Cell Graph constructed and pruned')

# Split jobs into train, validation, and test sets
#train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
#n_jobs = len(jobs)
#n_train = int(n_jobs * train_ratio)
#n_val = int(n_jobs * val_ratio)
#n_test = n_jobs - n_train - n_val
#random.shuffle(jobs)
#train_jobs = jobs[:n_train]
#val_jobs = jobs[n_train:n_train + n_val]
#test_jobs = jobs[n_train + n_val:]
#print(f"Train jobs: {len(train_jobs)}, Val jobs: {len(val_jobs)}, Test jobs: {len(test_jobs)}")
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
n_jobs = len(jobs)
random.shuffle(jobs)

if n_jobs >= 3:
    n_val = max(1, int(n_jobs * val_ratio))
    n_test = max(1, int(n_jobs * test_ratio))
    n_train = n_jobs - n_val - n_test
else:
    # Too few sub-sampled jobs to hold out anything meaningfully;
    # reuse the training jobs for validation/test rather than leaving them empty.
    n_train = n_jobs
    n_val = 0
    n_test = 0

train_jobs = jobs[:n_train]
val_jobs = jobs[n_train:n_train + n_val] if n_val > 0 else train_jobs
test_jobs = jobs[n_train + n_val:] if n_test > 0 else train_jobs
print(f"Train jobs: {len(train_jobs)}, Val jobs: {len(val_jobs)}, Test jobs: {len(test_jobs)}")


################################################################
#   
# SUBGRAPHS SAMPLING FOR TRAINING
#
################################################################
#args.n_heads = 8
#args.n_hid = 128  # Ensure divisibility
# num_types/num_relations are derived from the graph itself (N omic node types + 1
# sample type; however many relation types the masks above actually produced, + 1
# reserved "self" slot) instead of being hand-computed and pasted in as literals -
# this is what lets the exact same code run for 2 omics or 15.
num_types = len(graph.get_types())
num_relations = len(graph.get_meta_graph()) + 1  # +1 for the reserved 'self' relation
gnn = GNN(conv_name=args.layer_type, in_dim=args.ae_n_hid,
                n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
                num_types=num_types, num_relations=num_relations, use_RTE=False, n_labels=one_hot_labels.shape[1],
                sample_type_id=len(OMIC_NAMES)  # 'sample' is inserted last into graph.node_feature - see build_graph_generic()
                ).to(device) # Loads HGT function
#gnn = GNN_class(conv_name=args.layer_type, in_dim=256,
#                n_hid=args.n_hid, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
#                num_types=2, num_relations=6, use_RTE=False, 
#                h_out=patient_loss.shape[1]
#                ).to(device) # Loads HGT function
# Options to use different optimization algorithms. Default: adamw
if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(gnn.parameters(), lr=args.lr)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(gnn.parameters(), lr=args.lr)
    

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=args.factor, patience=args.patience, verbose=True)
    
    

################################################################
#
# TRAIN HGT IN SUB-SAMPLING
#
################################################################


start = timer()
gnn.train() # The GNN arquitecture. It shows the layers and dimensions.
training_loss = []
kl_losses_history = {name: [] for name in OMIC_NAMES}
cross_entropy_losses = []
cosine_losses = []
total_losses = []
val_losses = []
lr_history = []

cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.5, reduction='sum').to(device)  # Initialize CosineEmbeddingLoss

patience = 50
patience_counter = 0
best_val_loss = float('inf')
best_model_state = None

################################################################
#
# RESUME FROM CHECKPOINT (if requested and available)
#
################################################################
# Saved every --checkpoint_every epochs so a training run killed by a time limit or an
# out-of-memory/preemption event can pick back up close to where it left off, instead
# of retraining from scratch. "Gradients" in the resumable sense means the optimizer's
# momentum/moment buffers (Adam-family optimizers), not raw .grad tensors - those are
# recomputed fresh from each batch's forward pass and aren't meaningful to persist.
checkpoint_path = model_dir + 'checkpoint_latest.pt'
start_epoch = 0

if args.resume:
    if os.path.exists(checkpoint_path):
        print(f"Resuming HGT training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        total_losses = checkpoint.get('total_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        cosine_losses = checkpoint.get('cosine_losses', [])
        cross_entropy_losses = checkpoint.get('cross_entropy_losses', [])
        kl_losses_history = checkpoint.get('kl_losses_history', {name: [] for name in OMIC_NAMES})
        lr_history = checkpoint.get('lr_history', [])
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        if torch.cuda.is_available() and checkpoint.get('cuda_random_state') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
        print(f"Resumed at epoch {start_epoch} (checkpoint had completed {len(total_losses)} epochs; "
              f"training will continue up to --epoch {args.epoch}).")
        if start_epoch >= args.epoch:
            print(f"Checkpoint already reached epoch {start_epoch}, which is >= --epoch {args.epoch}. "
                  f"Nothing to resume - pass a larger --epoch to keep training. Exiting.")
            sys.exit(0)
    else:
        print(f"--resume was set but no checkpoint was found at {checkpoint_path}; starting training from scratch.")


def save_checkpoint(current_epoch):
    torch.save({
        'epoch': current_epoch,
        'model_state_dict': gnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'total_losses': total_losses,
        'val_losses': val_losses,
        'cosine_losses': cosine_losses,
        'cross_entropy_losses': cross_entropy_losses,
        'kl_losses_history': kl_losses_history,
        'lr_history': lr_history,
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, checkpoint_path)


for epoch in np.arange(start_epoch, args.epoch): # Iterates over the epochs. Number set to 100.
    L = 0 # Initialization
    #check = 0
    gnn.train()
    for job in train_jobs: # Iterates over the graph-subsampling generated in the preprocessing
        #print(check)
        feature,time,edge_list,indxs = job # Initialize the job. Sets feature (cells and genes), time (time dealing in transformers), edges (connections) and indeces (no idea).
        node_dict = {} # Initialization
        node_feature = [] # Initialization
        node_type = [] # Initialization
        node_time = [] # Initialization
        edge_index = [] # Initialization
        edge_type = [] # Initialization
        edge_time = [] # Initialization
        
        node_num = 0 # Initialization
        types = graph.get_types()   # ['gene','cell']
        for t in types: # Iterates over genes and cells
            node_dict[t] = [node_num, len(node_dict)] # Creates a dictionary of cells and genes
            node_num += len(feature[t])
                # node_dict: {'gene':[0,0],'cell':[134,1]}
                 
        for t in types: # Iterates over genes and cellsc
            t_i = node_dict[t][1]
            node_feature.insert(t_i, torch.tensor(feature[t], dtype=torch.float32).to(device)) # Saves cells or genes matrices  
            node_time += list(time[t]) # Saves node time
            node_type += [node_dict[t][1] for _ in range(len(feature[t]))] # Saves node type
             
        edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())} # Reference dictionary of connections
        edge_dict['self'] = len(edge_dict) # Incoporates self lenght parameter to edge dictionary
        # {'g_c': 0, 'rev_g_c': 1 ,'self': 2}
        for target_type in edge_list:
            for source_type in edge_list[target_type]:
                for relation_type in edge_list[target_type][source_type]:
                    for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                        tid, sid = ti + \
                            node_dict[target_type][0], si + \
                            node_dict[source_type][0]
                        edge_index += [[sid, tid]]
                        edge_type += [edge_dict[relation_type]]
                        edge_time += [120]
                            
                        
                    
                
        node_feature = torch.cat(node_feature, 0) # Nodes features matrix in tensor format (N omics + samples, in type-index order)
        node_type = torch.LongTensor(node_type) # Nodes type matrix in tensor format
        edge_time = torch.LongTensor(edge_time) # Edges time matrix in tensor format (all the same)
        edge_index = torch.LongTensor(edge_index).t() # Edges indexes (nodes-nodes connections) in tensor format
        #edge_index[1][np.array(edge_type) == edge_dict['g_g']] = edge_index[1][np.array(edge_type) == edge_dict['g_g']]-node_dict['sample'][0]
        #edge_index[0][np.array(edge_type) == edge_dict['rev_g_g']] = edge_index[0][np.array(edge_type) == edge_dict['rev_g_g']]-node_dict['sample'][0]
        #edge_index[1][np.array(edge_type) == edge_dict['s_s']] = edge_index[1][np.array(edge_type) == edge_dict['s_s']]+node_dict['sample'][0]+node_dict['metabolite'][0]
        #edge_index[0][np.array(edge_type) == edge_dict['rev_s_s']] = edge_index[0][np.array(edge_type) == edge_dict['rev_s_s']]+node_dict['sample'][0]+node_dict['metabolite'][0]
        #edge_index[1][np.array(edge_type) == edge_dict['m_m']] = edge_index[1][np.array(edge_type) == edge_dict['m_m']]+node_dict['metabolite'][0]
        #edge_index[0][np.array(edge_type) == edge_dict['rev_m_m']] = edge_index[0][np.array(edge_type) == edge_dict['rev_m_m']]+node_dict['metabolite'][0]
        edge_type = torch.LongTensor(edge_type) # Edges type in tensor format
        #node_rep, cell_class = gnn.forward(node_feature, 
        node_rep, class_rep, g_32, g_64, g_128 = gnn.forward(node_feature,node_type.to(device),edge_time.to(device),edge_index.to(device),edge_type.to(device)) # Applies HGT to obtain nodes representations
            #if args.rep == 'T':
                #node_rep = torch.trunc(node_rep*10000000000)/10000000000 # NO IDEA.
                #if args.reduction == 'raw':
                #    for t in types:
                #        t_i = node_dict[t][1]
                #       print("t_i="+str(t_i))
                #        node_decoded_embedding[t_i] = torch.trunc(
                #            node_decoded_embedding[t_i]*10000000000)/10000000000
        omic_reps = {name: node_rep[node_type == i, ] for i, name in enumerate(OMIC_NAMES)} # Extracts each omic's node representations
        sample_matrix = node_rep[node_type == len(OMIC_NAMES), ]
        regularization_loss = 0 # Initialization of regularization loss
        for param in gnn.parameters():
            regularization_loss += torch.sum(torch.pow(param, 2)) # Saves the regularization loss over parameters
                
        if (args.loss == "sup"): # No other option
            decoders = {name: torch.mm(sample_matrix, omic_reps[name].t()) for name in OMIC_NAMES}
            adj = samples_stacked.T[indxs['sample'], ] # Extracts every omic's feature columns for the sampled samples
            omic_offsets = {}
            _off = 0
            for name in OMIC_NAMES:
                omic_offsets[name] = _off
                _off += encoded[name].shape[0]
            indexes = np.concatenate([indxs[name] + omic_offsets[name] for name in OMIC_NAMES])
            adj_per_omic = {name: torch.tensor(adj[:, indxs[name] + omic_offsets[name]], dtype=torch.float32).to(device)
                            for name in OMIC_NAMES} # per-omic columns are offset past every earlier omic's columns in `adj`
            adj = adj[:, indexes] # Extracts every sampled omic feature's column, in omic order
            adj = torch.tensor(adj, dtype=torch.float32).to(device) # Sets an adjacency matrix
            samples_labels = one_hot_labels[indxs['sample'], ]
            #samples_labels = one_hot_labels[node_type == 2, ]
            #cells_labels = patient_loss[indxs['cell'],:]
            if args.reduction == 'raw':
                if epoch % 2 == 0:
                    loss = F.kl_div(decoder.softmax(
                        dim=-1).log(), adj.softmax(dim=-1), reduction='sum')+args.rf*regularization_loss
                else:
                    loss = nn.MSELoss()(
                        node_feature[0], node_decoded_embedding[0])+args.rf*regularization_loss
                    for t_i in range(1, len(types)):
                        loss += nn.MSELoss()(node_feature[t_i],
                                                    node_decoded_embedding[t_i])
                            
                        
            else:
                #loss = F.kl_div(decoder_g.softmax(dim=-1).log(), adj_g.softmax(dim=-1), reduction='sum')
                #loss += F.kl_div(decoder_m.softmax(dim=-1).log(),adj_m.softmax(dim=-1), reduction='sum') # Calculates loss function based in KL divergence
                #loss += F.cross_entropy(class_rep.softmax(dim=-1),torch.FloatTensor(samples_labels).to(device))
                kl_losses = {name: F.kl_div(decoders[name].softmax(dim=-1).log(), adj_per_omic[name].softmax(dim=-1), reduction='sum')
                             for name in OMIC_NAMES}
                ce_loss = F.cross_entropy(class_rep,torch.FloatTensor(samples_labels).to(device))
                #loss = 0.2*kl_gene + 0.2*kl_metabo + 0.6*ce_loss
                #loss = kl_gene + kl_metabo + ce_loss
                #loss += F.kl_div(cell_class.softmax(dim=-1).log(),torch.FloatTensor(cells_labels).to(device).softmax(dim=-1), reduction='sum')
                # Compute CosineEmbeddingLoss for sample-sample pairs
                num_samples = sample_matrix.shape[0]
                max_pairs = 1000  # Limit pairs to avoid O(n^2) computation
                sample_idx_i = []
                sample_idx_j = []
                targets = []
                
                # Assume samples_labels is one-hot; convert to class indices
                sample_classes = torch.argmax(torch.LongTensor(samples_labels), dim=1)
                
                # Sample positive pairs (same class)
                for cls in torch.unique(sample_classes):
                    cls_idx = torch.where(sample_classes == cls)[0]
                    if len(cls_idx) >= 2:  # Need at least 2 samples for pairs
                        pairs = torch.combinations(cls_idx, r=2)
                        sample_idx_i.extend(pairs[:, 0].tolist())
                        sample_idx_j.extend(pairs[:, 1].tolist())
                        targets.extend([1] * len(pairs))
                    
                # Sample negative pairs (different classes)
                for i in range(num_samples):
                    for j in range(i + 1, num_samples):
                        if sample_classes[i] != sample_classes[j]:
                            sample_idx_i.append(i)
                            sample_idx_j.append(j)
                            targets.append(-1)
                
                # Limit total pairs
                if len(sample_idx_i) > max_pairs:
                    idx = torch.randperm(len(sample_idx_i))[:max_pairs]
                    sample_idx_i = [sample_idx_i[i] for i in idx]
                    sample_idx_j = [sample_idx_j[i] for i in idx]
                    targets = [targets[i] for i in idx]
                    
                if len(sample_idx_i) > 0:
                    sample_emb_i = sample_matrix[sample_idx_i]
                    sample_emb_j = sample_matrix[sample_idx_j]
                    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
                    cosine_loss = cosine_loss_fn(sample_emb_i, sample_emb_j, target_tensor)
                else:
                    cosine_loss = torch.tensor(0.0, device=device)
                    
                #loss = 0.1 * kl_gene + 0.1 * kl_metabo + 0.4 * ce_loss + 0.4 * cosine_loss
                loss = 0.3 * ce_loss + 0.7 * cosine_loss
                
        L += loss.item() # Saves loss 
        optimizer.zero_grad() # Initializies optimizer
        loss.backward() # Sets backpropagation to loss function
        optimizer.step() # Sets a step in optimizer
        #check += 1
        
    scheduler.step(L/(int(len(train_jobs))))
    lr_history.append(optimizer.param_groups[0]['lr'])
        # Validation phase
    gnn.eval()
    val_L = 0
    with torch.no_grad():
        for job in val_jobs:
            feature, time, edge_list, indxs = job
            node_dict = {}
            node_feature = []
            node_type = []
            node_time = []
            edge_index = []
            edge_type = []
            edge_time = []
            
            node_num = 0
            types = graph.get_types()
            for t in types:
                node_dict[t] = [node_num, len(node_dict)]
                node_num += len(feature[t])
                
            for t in types:
                t_i = node_dict[t][1]
                node_feature.insert(t_i, torch.tensor(feature[t], dtype=torch.float32).to(device))
                node_time += list(time[t])
                node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
                
            edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
            edge_dict['self'] = len(edge_dict)
            for target_type in edge_list:
                for source_type in edge_list[target_type]:
                    for relation_type in edge_list[target_type][source_type]:
                        for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                            tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                            edge_index += [[sid, tid]]
                            edge_type += [edge_dict[relation_type]]
                            edge_time += [120]
                            
            node_feature = torch.cat(node_feature, 0) # Nodes features matrix in tensor format (N omics + samples, in type-index order)
            node_type = torch.LongTensor(node_type).to(device)
            edge_time = torch.LongTensor(edge_time).to(device)
            edge_index = torch.LongTensor(edge_index).t().to(device)
            edge_type = torch.LongTensor(edge_type).to(device)
            node_rep, class_rep, g_32, g_64, g_128 = gnn.forward(
                node_feature, node_type, edge_time, edge_index, edge_type
            )
            omic_reps = {name: node_rep[node_type == i, :] for i, name in enumerate(OMIC_NAMES)}
            sample_matrix = node_rep[node_type == len(OMIC_NAMES), :]
            regularization_loss = 0
            for param in gnn.parameters():
                regularization_loss += torch.sum(torch.pow(param, 2))
                
            if args.loss == "sup":
                decoders = {name: torch.mm(sample_matrix, omic_reps[name].t()) for name in OMIC_NAMES}
                adj = samples_stacked.T[indxs['sample'], :] # Extracts every omic's feature columns for the sampled samples
                omic_offsets = {}
                _off = 0
                for name in OMIC_NAMES:
                    omic_offsets[name] = _off
                    _off += encoded[name].shape[0]
                indexes = np.concatenate([indxs[name] + omic_offsets[name] for name in OMIC_NAMES])
                adj_per_omic = {name: torch.tensor(adj[:, indxs[name] + omic_offsets[name]], dtype=torch.float32).to(device)
                                for name in OMIC_NAMES} # per-omic columns are offset past every earlier omic's columns in `adj`
                adj = adj[:, indexes] # Extracts every sampled omic feature's column, in omic order
                adj = torch.tensor(adj, dtype=torch.float32).to(device)
                samples_labels = one_hot_labels[indxs['sample'], :]
                if args.reduction == 'raw':
                    if epoch % 2 == 0:
                        val_loss = F.kl_div(decoder.softmax(dim=-1).log(), adj.softmax(dim=-1), reduction='sum') + args.rf * regularization_loss
                    else:
                        val_loss = nn.MSELoss()(node_feature[0], node_decoded_embedding[0]) + args.rf * regularization_loss
                        for t_i in range(1, len(types)):
                            val_loss += nn.MSELoss()(node_feature[t_i], node_decoded_embedding[t_i])
                else:
                    kl_losses = {name: F.kl_div(decoders[name].softmax(dim=-1).log(), adj_per_omic[name].softmax(dim=-1), reduction='sum')
                                 for name in OMIC_NAMES}
                    ce_loss = F.cross_entropy(class_rep, torch.FloatTensor(samples_labels).to(device))
                    num_samples = sample_matrix.shape[0]
                    max_pairs = 1000
                    sample_idx_i = []
                    sample_idx_j = []
                    targets = []
                    sample_classes = torch.argmax(torch.FloatTensor(samples_labels).to(device), dim=1)
                    for cls in torch.unique(sample_classes):
                        cls_idx = torch.where(sample_classes == cls)[0]
                        if len(cls_idx) >= 2:
                            pairs = torch.combinations(cls_idx, r=2)
                            sample_idx_i.extend(pairs[:, 0].tolist())
                            sample_idx_j.extend(pairs[:, 1].tolist())
                            targets.extend([1] * len(pairs))
                            
                    for i in range(num_samples):
                        for j in range(i + 1, num_samples):
                            if sample_classes[i] != sample_classes[j]:
                                sample_idx_i.append(i)
                                sample_idx_j.append(j)
                                targets.append(-1)
                                
                    if len(sample_idx_i) > max_pairs:
                        idx = torch.randperm(len(sample_idx_i))[:max_pairs]
                        sample_idx_i = [sample_idx_i[i] for i in idx]
                        sample_idx_j = [sample_idx_j[i] for i in idx]
                        targets = [targets[i] for i in idx]
                        
                    if len(sample_idx_i) > 0:
                        sample_emb_i = sample_matrix[sample_idx_i]
                        sample_emb_j = sample_matrix[sample_idx_j]
                        target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
                        cosine_loss = cosine_loss_fn(sample_emb_i, sample_emb_j, target_tensor)
                    else:
                        cosine_loss = torch.tensor(0.0, device=device)
                        
                    #val_loss = 0.2 * kl_gene + 0.2 * kl_metabo + 0.3 * ce_loss + 0.3 * cosine_loss
                    val_loss = 0.3 * ce_loss + 0.7 * cosine_loss
                    
                val_L += val_loss.item()
    #print('Epoch :', epoch+1, '|', 'train_loss:%.12f' % (L/(int(len(jobs)))/args.n_batch)) # Print loss in epoch
    #print('Epoch :', epoch+1, '|', 'train_loss:%.12f' % (L/(int(len(jobs))))) # Print loss in epoch
    #val_loss=1
    printed_val_loss = val_L/(int(len(val_jobs)))
    printed_train_loss = L/(int(len(train_jobs)))
    print(f'Epoch: {epoch+1} | Train Loss: {printed_train_loss:.12f}\nEpoch: {epoch+1} | Val Loss: {printed_val_loss:.12f}')
    #training_loss.append((L/(int(samples.shape[0]))/args.n_batch))
    for name in OMIC_NAMES:
        kl_losses_history[name].append(kl_losses[name].item())
    cross_entropy_losses.append(ce_loss.item())
    cosine_losses.append(cosine_loss.item())  # Track cosine loss
    total_losses.append(loss.item())
    val_losses.append(printed_val_loss)

    if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epoch:
        save_checkpoint(epoch)
        print(f"Checkpoint saved at epoch {epoch + 1} -> {checkpoint_path}")
    # Early stopping checks
    #if cosine_loss.item() < 1e-6 and ce_loss.item() < 1e-6:
    #    patience_counter += 1
    #    print(f'Early stopping condition: cosine_loss ({cosine_loss.item():.6f}) and ce_loss ({ce_loss.item():.6f}) near zero')
    #elif val_loss > loss:
    #    patience_counter += 1
    #    print(f'Early stopping condition: val_loss ({val_loss:.6f}) > train_loss ({loss:.6f})')
    #else:
    #    patience_counter = 0
    #    best_val_loss = min(best_val_loss, val_loss)
    #    best_model_state = gnn.state_dict()
    #    state = {'model': gnn.state_dict(), 'optimizer': scheduler.state_dict(),
    #    'epoch': epoch} # Saves a dictionary of model
    #    model0=f'BiomiX_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
    #    torch.save(state, model_dir+model0) # Saves model in folder
    #    
    #if patience_counter >= patience:
    #    print(f'Early stopping triggered after {patience} epochs of no improvement')
    #    gnn.load_state_dict(best_model_state)
    #    break
    

    
    



end = timer()
print(f"BiomiX-HGT took {round((end - start)/60,2)} mins to train")
unique_vals, counts = torch.unique(edge_type, return_counts=True) # Count value repetitions
count_dict = dict(zip(unique_vals.tolist(), counts.tolist())) # Combine into a dictionary (optional)
print(f"Edge dict is:\n{edge_dict}\nAnd count dict is:\n{count_dict}")

#import matplotlib.pyplot as plt

# Total/Validation loss carry the headline message (converging? overfitting?) so they
# get bold, saturated lines; the two loss components are supporting detail, so they're
# thin and muted to avoid competing for attention.
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(total_losses, label='Total Loss', color=OKABE_ITO[0], linewidth=2.5, zorder=3)
ax.plot(val_losses, label='Validation Loss', color=OKABE_ITO[1], linewidth=2.5, zorder=3)
ax.plot(cosine_losses, label='Cosine Loss (component)', color='#999999', linewidth=1.1, linestyle='--', zorder=2)
ax.plot(cross_entropy_losses, label='Cross Entropy (component)', color='#bbbbbb', linewidth=1.1, linestyle=':', zorder=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Convergence')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(plots_dir+"training_loss_components.pdf")
plt.close(fig)


state = {'model': gnn.state_dict(), 'optimizer': scheduler.state_dict(),
        'epoch': epoch} # Saves a dictionary of model
#model0=f'BiomiX_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
model0 = f'BiomiX_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
torch.save(state, model_dir+model0) # Saves model in folder
pd.DataFrame(total_losses).to_csv(loss_dir+model0+"_total_losses.csv")
#pd.DataFrame(kl_gene_losses).to_csv(loss_dir+model0+"_kl_gene_losses.csv")
#pd.DataFrame(kl_metabo_losses).to_csv(loss_dir+model0+"_kl_metabo_losses.csv")
pd.DataFrame(cosine_losses).to_csv(loss_dir+model0+"_cosine_losses.csv")
pd.DataFrame(cross_entropy_losses).to_csv(loss_dir+model0+"_cross_entropy_losses.csv")
pd.DataFrame(val_losses).to_csv(loss_dir+model0+"_val_losses.csv")
debuginfoStr('Graph Autoencoder training finished')

print(class_rep)
print(torch.FloatTensor(samples_labels))



################################################################
#
# APPLY HGT TO WHOLE GRAPH
#   
################################################################

# Load trained model
gnn.load_state_dict(state['model'])
gnn.eval()

# Determine batch size
sample_len = samples_stacked.T.shape[0]
if sample_len > 10000:
    ba = 500
else:
    ba = sample_len
    

#if (gene_cell.shape[1]>10000):
#        
#        if (gene_cell.shape[0]>10000):
#            ba = 500
#        else:
#            ba = gene_cell.shape[0]
#    else:
#        if (gene_cell.shape[0]>10000):
#            ba = 5000
#        else:
#            ba = gene_cell.shape[0] # Adapts batch to data dimensions

sample_embedding = []
omic_embedding = {name: [] for name in OMIC_NAMES}
attention = []
saliency = []
g32_embeddings = []
g64_embeddings = []
g128_embeddings = []
class_embeddings = []

node_features_full = {name: encoded[name].detach().numpy() for name in OMIC_NAMES}

#with torch.no_grad():
#with gnn.eval():
for i in range(0, sample_len, ba):
        # Sub-batch slice
    sample_batch = samples_stacked.T[i:i+ba]

        # Build (x, node_type, edge_time, edge_index, edge_type) for the fixed omic
        # feature nodes plus this batch of sample nodes - the N-omic generalization
        # of the original per-omic build_graph()/build_data() calls.
    batch_node_features = dict(node_features_full)
    batch_node_features['sample'] = encoded2_samples[i:i+ba, :].detach().numpy()
    batch_cross_sample_masks = {name: cross_sample_masks[name][:, i:i+ba] for name in OMIC_NAMES}
    batch_sample_mask = sample_mask[i:i+ba, i:i+ba]  # sub-mask for current batch

    x, node_type, edge_time, edge_index, edge_type = build_data_generic(
        batch_node_features,
        batch_cross_sample_masks,
        self_masks,
        batch_sample_mask,
        cross_omic_masks,
        OMIC_NAMES,
        edge_dict
    )

    gnn.eval()
    x = torch.cat([x[name] for name in OMIC_NAMES] + [x['sample']], 0) # Nodes features matrix in tensor format
    x.requires_grad = True
    node_rep, class_rep, g_32, g_64, g_128 = gnn.forward(x,
                                    node_type.to(device),edge_time.to(device),
                                    edge_index.to(device), edge_type.to(device)) # Applies HGT
    attention.append(gnn.att.detach().numpy()) # Saves attention
    for _idx, _name in enumerate(OMIC_NAMES):
        omic_embedding[_name].append(node_rep[node_type == _idx, ].detach().numpy()) # Extracts this omic's node representations
    sample_matrix = node_rep[node_type == len(OMIC_NAMES), ]
    sample_embedding.append(sample_matrix.detach().numpy())
    # Choose a scalar objective for gradients
    #score = class_rep.norm()  # or class_rep.mean() or a specific class prediction
    #score.backward()
    target = node_rep.mean() + class_rep.mean()
    target.backward()
    # Grab gradients
    #saliency.append(x.grad.data.abs())  # [N, F] gradient magnitude
    saliency.append(x.grad.abs().cpu().numpy())
    g32_embeddings.append(g_32.detach().numpy())
    g64_embeddings.append(g_64.detach().numpy())
    g128_embeddings.append(g_128.detach().numpy())
    class_embeddings.append(class_rep.detach().numpy())
                    



# SAVING RESULTS
# np.vstack naturally handles a final partial batch, so there is no need to special-case
# samples.shape[1] % ba != 0 (a previous version of this branch mixed up gene/metabo/sample
# matrices when the batch size didn't evenly divide the sample count).
omic_matrix = {name: np.vstack(omic_embedding[name]) for name in OMIC_NAMES}
sample_matrix = np.vstack(sample_embedding)
attention = np.vstack(attention)
saliency = np.vstack(saliency)
g32_embeddings = np.vstack(g32_embeddings)
g64_embeddings = np.vstack(g64_embeddings)
g128_embeddings = np.vstack(g128_embeddings)
class_embeddings = np.vstack(class_embeddings)


#cell_matrix = cell_matrix.detach().numpy()
for name in OMIC_NAMES:
    np.savetxt(omic_dirs[name]+file0, omic_matrix[name], delimiter=' ')
np.savetxt(sample_dir+file0, sample_matrix, delimiter=' ')
#np.savetxt(embeddings_dir+file0, sample_matrix, delimiter=' ')
positions = pd.DataFrame(edge_index.T)
df = pd.DataFrame(attention)
df2 = pd.concat([positions, df], axis=1)
attention = df2
#attention_save = np.concatenate([edge_index.T,attention],axis=1)
#np.savetxt(att_dir+file0, attention_save, delimiter=' ')
df2.to_csv(att_dir+file0+".csv",sep=",", index=True)
np.savetxt(embbs_dir+"g128_embeddings", g128_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"g64_embeddings", g64_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"g32_embeddings", g32_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"class_embeddings", class_embeddings, delimiter=' ')
#np.savetxt(embbs_dir+"node_embeddings", class_rep, delimiter=' ')



################################################################
#
# EVALUATION
#   
################################################################

# Compute ROC-AUC and F1-Score
#from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
# Convert class_embeddings to probabilities
class_probs = F.softmax(torch.tensor(class_embeddings, dtype=torch.float32), dim=1).numpy()
# Convert one_hot_labels to class indices
true_labels = np.argmax(one_hot_labels, axis=1)

# ROC-AUC
try:
    roc_auc = roc_auc_score(
        y_true=one_hot_labels,
        y_score=class_probs,
        multi_class='ovr',
        average='macro'
    )
    # Per-class ROC-AUC
    roc_auc_per_class = {}
    for i in range(one_hot_labels.shape[1]):
        if np.sum(one_hot_labels[:, i]) > 0:  # Only compute for classes with samples
            roc_auc_per_class[i] = roc_auc_score(
                one_hot_labels[:, i],
                class_probs[:, i]
            )
    print(f"ROC-AUC (macro): {roc_auc:.4f}")
    print(f"Per-class ROC-AUC: {roc_auc_per_class}")
except ValueError as e:
    print(f"Error computing ROC-AUC: {e}")
    roc_auc = 0.0
    roc_auc_per_class = {}

# F1-Score
pred_labels = np.argmax(class_probs, axis=1)
f1_macro = f1_score(true_labels, pred_labels, average='macro')
f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
f1_micro = f1_score(true_labels, pred_labels, average='micro')
print(f"F1-Score (macro): {f1_macro:.4f}, (weighted): {f1_weighted:.4f}, (micro): {f1_micro:.4f}")


# Compute and plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
# Row/column order must match the integer labels 0..k-1 that true_labels/pred_labels
# actually use, i.e. LabelEncoder's sorted class order - NOT metadata.CONDITION.unique()
# (pandas .unique() returns first-appearance order, which silently mislabels the plot
# whenever it differs from the alphabetical order LabelEncoder assigned).
condition_names = label_encoder.classes_
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
annot = np.array([[f"{c}\n({p:.0f}%)" for c, p in zip(row_c, row_p)]
                   for row_c, row_p in zip(cm, cm_pct)])

fig, ax = plt.subplots(figsize=(1.6 * len(condition_names) + 3, 1.4 * len(condition_names) + 3))
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
            xticklabels=condition_names, yticklabels=condition_names,
            cbar_kws={'label': 'Samples'}, linewidths=0.5, linecolor='white', ax=ax)
ax.set_xlabel('Predicted Condition')
ax.set_ylabel('True Condition')
ax.set_title(f'Confusion Matrix\nF1 (macro) = {f1_macro:.2f}  |  ROC-AUC (macro) = {roc_auc:.2f}')
plt.tight_layout()
plt.savefig(plots_dir+'Confusion_matrix_figure.pdf')
plt.close(fig)



#-----------------------------------------------------------------------
#
#----------------- Interpretability Plots ------------------------------
#
#-----------------------------------------------------------------------

edge_attn = torch.FloatTensor(attention.iloc[:,2:].mean(axis=1))
src_nodes = edge_index[0]
tgt_nodes = edge_index[1]
# Aggregate per-node attention (e.g., mean of outgoing edge attentions)
node_attention_sum = torch.zeros((x.shape[0], 1), device=edge_attn.device)
node_attention_count = torch.zeros((x.shape[0], 1), device=edge_attn.device)
node_attention_sum.index_add_(0, src_nodes, edge_attn)
node_attention_count.index_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float32).unsqueeze(1))

# Filter nodes with non-zero counts
connected_mask = (node_attention_count > 0).squeeze()
connected_nodes = torch.arange(x.shape[0], device=device)[connected_mask].cpu().numpy()
print(f"Filtering out {x.shape[0] - len(connected_nodes)} nodes with zero attention count")
# Filter node-related data
node_attention_sum = node_attention_sum[connected_mask].detach().cpu().numpy()
node_attention_count = node_attention_count[connected_mask].detach().cpu().numpy()
node_type_filtered = node_type[connected_mask].cpu().numpy()
saliency_filtered = saliency[connected_mask]
# Create node mapping for edge indices
#node_mapping = {old: new for new, old in enumerate(connected_nodes)}
# Filter attention DataFrame
#attention_filtered = attention[
#    attention['source'].isin(connected_nodes) & 
#    attention['target'].isin(connected_nodes)
#].copy()
#attention_filtered['source'] = attention_filtered['source'].map(node_mapping)
#attention_filtered['target'] = attention_filtered['target'].map(node_mapping)

node_attention_mean = abs(node_attention_sum) / node_attention_count
node_attention_mean = (node_attention_mean.T*(abs(saliency_filtered).mean(axis=1))).T
node_attention_mean = (node_attention_mean/node_attention_mean.sum())*100
node_attention_mean = abs(saliency_filtered).mean(axis=1)


#sample_node_attention = pd.concat([pd.DataFrame(trans.columns[1:]),pd.DataFrame(node_attention_mean[node_type == 2])], axis=1)
#sample_node_attention.columns = ['Sample_ID', 'Attention']
#node_attention_mean = (node_attention_mean/node_attention_mean[node_type != 2].sum())*100
#sample_node_attention = node_attention_mean[node_type == 2]
#gene_node_attention = node_attention_mean[node_type == 0]
# Per-omic Feature/Attention dataframe - generalizes the original script's two
# hand-written gene_node_attention/metabo_node_attention frames to N omics.
omic_node_attention = {}
for _oi, _name in enumerate(OMIC_NAMES):
    _df = pd.concat([
        omic_dfs[_name]['ID'].reset_index(drop=True),
        pd.DataFrame(node_attention_mean[node_type_filtered == _oi])
    ], axis=1)
    _df.columns = [f'{_name}_ID', 'Attention']
    omic_node_attention[_name] = _df
#sample_node_attention = pd.concat([pd.DataFrame(trans.columns[1:]),pd.DataFrame(node_attention_mean[node_type == 2].detach().numpy())], axis=1)
#import mygene
#mg = mygene.MyGeneInfo()
#gene_info = mg.querymany(gene_node_attention['Gene_ID'].tolist(), scopes='ensembl.gene', fields='symbol', species='human')
#gene_symbols = pd.DataFrame(gene_info)[['query', 'symbol']].drop_duplicates().rename(columns={'query': 'Gene_ID', 'symbol': 'Gene_Symbol'})
#gene_node_attention = pd.merge(gene_node_attention, gene_symbols, on='Gene_ID', how='left')
#gene_node_attention['Gene_Symbol'].fillna(gene_node_attention['Gene_ID'], inplace=True)  # fallback to ID if no match
#from matplotlib import gridspec
# --------------------------
# Prepare one Feature/Attention/Type dataframe per omic
# --------------------------
omic_feature_dfs = {}
for _name in OMIC_NAMES:
    _df = omic_node_attention[_name][[f'{_name}_ID', 'Attention']].copy()
    _df['Type'] = f'{_name}_feature'
    _df = _df.rename(columns={f'{_name}_ID': 'Feature'})
    omic_feature_dfs[_name] = _df
combined = pd.concat(list(omic_feature_dfs.values()), ignore_index=True)
combined = combined.sort_values('Attention', ascending=False).reset_index(drop=True)
# --------------------------
# Cumulative attention + saturation
# --------------------------
combined['Cumulative_Attention'] = combined['Attention'].cumsum()
combined['Cumulative_Attention_%'] = 100 * combined['Cumulative_Attention'] / combined['Attention'].sum()
saturation_threshold = 95
saturation_idx = (combined['Cumulative_Attention_%'] >= saturation_threshold).idxmax()
saturation_x = saturation_idx + 1
saturation_y = combined.loc[saturation_idx, 'Cumulative_Attention_%']
#top_features = combined.iloc[:saturation_x]
top_features = combined.iloc[:min(combined.shape[0],300)]
# --------------------------
# Prepare top 100 features per omic
# --------------------------
omic_top100 = {_name: _df.sort_values('Attention', ascending=False).head(min(100, _df.shape[0]))
                for _name, _df in omic_feature_dfs.items()}
omic_total_attention = {_name: _df['Attention'].sum() for _name, _df in omic_feature_dfs.items()}
# --------------------------
# Plot all into one PDF page
# --------------------------
fig = plt.figure(figsize=(6 * (len(OMIC_NAMES) + 1), 24))
gs = gridspec.GridSpec(2, len(OMIC_NAMES) + 1, height_ratios=[1, 0.6])
# Top 100 features, one panel per omic
for _oi, _name in enumerate(OMIC_NAMES):
    _ax = fig.add_subplot(gs[0, _oi])
    _ax.barh(omic_top100[_name]['Feature'], omic_top100[_name]['Attention'], color=OMIC_COLORS[_name])
    _ax.invert_yaxis()
    _ax.set_xlabel('Attention Score')
    _ax.set_title(f'Top 100 {_name} Features\nTotal: {omic_total_attention[_name]:.2f} | '
                  f'Mean: {omic_total_attention[_name] / len(omic_feature_dfs[_name]):.5f}')
    _ax.tick_params(axis='y', labelsize=5)
# Top Features Before Saturation
ax2 = fig.add_subplot(gs[0, len(OMIC_NAMES)])
_type_to_color = {f'{_name}_feature': OMIC_COLORS[_name] for _name in OMIC_NAMES}
colors = top_features['Type'].map(_type_to_color)
ax2.barh(top_features['Feature'], top_features['Attention'], color=colors)
ax2.invert_yaxis()
ax2.set_xlabel('Attention Score')
ax2.set_title(f'Features Before Saturation\nTop {top_features.shape[0]}/{saturation_x} Features to reach {saturation_threshold}%')
ax2.tick_params(axis='y', labelsize=3)
ax2.legend(handles=[Patch(facecolor=OMIC_COLORS[_name], label=_name) for _name in OMIC_NAMES],
           loc='lower right', fontsize=8)
# Cumulative Attention Curve
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(combined['Cumulative_Attention_%'], color='#333333', linewidth=1.5)
ax3.axvline(x=saturation_x, color=OKABE_ITO[4], linestyle='--', linewidth=1.2)
ax3.axhline(y=saturation_y, color=OKABE_ITO[4], linestyle='--', linewidth=1.2)
ax3.text(saturation_x + 2, saturation_y - 5, f'{saturation_x} features\n({saturation_y:.2f}%)',
         color=OKABE_ITO[4], fontsize=10, ha='left', va='top')
ax3.set_xlabel('Number of Top Features')
ax3.set_ylabel('Cumulative Attention (%)')
ax3.set_title('Cumulative Feature Attention with Saturation Threshold')
# Final layout & save
plt.tight_layout()
plt.savefig(plots_dir+'Interpretability_nodes_attention_figure.pdf')
plt.close()



#-----------------------------------------------------------------------
#
#----------------- Leiden Clustering of Top-Feature Attention Network --
#
#-----------------------------------------------------------------------
# Isolates the attention subgraph among the top (<=300) genes/metabolites from the
# plot above, clusters it with Leiden to find multi-omic "programs", and saves the
# cluster composition (CSV) and a contracted cluster network (PDF).

import igraph as ig
import leidenalg
from collections import defaultdict, Counter

# Map every graph feature-node index -> which omic it belongs to - generalizes the
# original script's single num_genes_total cutoff (idx < num_genes_total -> gene,
# else metabolite) to an arbitrary number of omics.
FEATURE_OMIC_SIZES = {name: omic_matrices[name].shape[0] for name in OMIC_NAMES}
FEATURE_INDEX_TO_OMIC = np.concatenate([[name] * FEATURE_OMIC_SIZES[name] for name in OMIC_NAMES])

# Map every graph node index -> its feature/sample name, in the same order used
# throughout (each omic in manifest order, then samples - see utils.build_data_generic()).
full_node_names = pd.concat(
    [omic_dfs[name]['ID'].reset_index(drop=True) for name in OMIC_NAMES] + [pd.Series(common_samples)],
    ignore_index=True
)

name_to_node_idx = {}
for i, name in enumerate(full_node_names):
    name_str = str(name).strip()
    if name_str not in name_to_node_idx:
        name_to_node_idx[name_str] = i

# Match the top attention features (genes + metabolites, from the plot above) to
# their global node index in the attention graph.
top_feature_node_idx = []
top_feature_names_found = []
for name in top_features['Feature']:
    name_str = str(name).strip()
    if name_str in name_to_node_idx:
        top_feature_node_idx.append(name_to_node_idx[name_str])
        top_feature_names_found.append(name_str)
print(f"Leiden clustering: matched {len(top_feature_node_idx)}/{len(top_features)} top features to graph nodes")

# Rebuild a clean src/tgt/mean_attention edge table from the same attention data
# used for the plot above (avoids relying on the attention CSV's column names).
edge_attention_df = pd.DataFrame({
    'src': attention.iloc[:, 0].astype(int).values,
    'tgt': attention.iloc[:, 1].astype(int).values,
    'mean_attention': edge_attn.numpy()
})

# Per-feature attention score (used both as a node attribute for the colormap below
# and to rank each cluster's top features later on).
feature_attention_lookup = dict(zip(top_features['Feature'], top_features['Attention']))

G_leiden = nx.Graph()
for idx, name in zip(top_feature_node_idx, top_feature_names_found):
    G_leiden.add_node(
        idx,
        name=name,
        type=FEATURE_INDEX_TO_OMIC[idx],
        attention=float(feature_attention_lookup.get(name, 0.0))
    )

top_idx_set = set(top_feature_node_idx)
sub_edges = edge_attention_df[
    edge_attention_df['src'].isin(top_idx_set) &
    edge_attention_df['tgt'].isin(top_idx_set) &
    (edge_attention_df['src'] != edge_attention_df['tgt'])
].copy()

# The attention graph is directed (src->tgt and tgt->src can carry different
# attention values); collapse both directions per undirected pair by averaging.
sub_edges['pair'] = list(zip(
    np.minimum(sub_edges['src'], sub_edges['tgt']),
    np.maximum(sub_edges['src'], sub_edges['tgt'])
))
edge_summary = sub_edges.groupby('pair')['mean_attention'].mean().reset_index()

for (src, tgt), weight in zip(edge_summary['pair'], edge_summary['mean_attention']):
    if weight <= 0:
        continue
    src_type = G_leiden.nodes[src]['type']
    tgt_type = G_leiden.nodes[tgt]['type']
    if src_type == tgt_type:
        edge_type = f'{src_type}-{src_type}'
    else:
        edge_type = '-'.join(sorted((src_type, tgt_type)))
    G_leiden.add_edge(src, tgt, weight=float(weight), edge_type=edge_type)

isolated_nodes = list(nx.isolates(G_leiden))
print(f"Leiden clustering: removing {len(isolated_nodes)} isolated nodes out of {G_leiden.number_of_nodes()}")
G_leiden.remove_nodes_from(isolated_nodes)

valid_clusters = set()  # default so downstream code can safely check "if valid_clusters:"
G_cluster = nx.Graph()
node_to_cluster = {}
cluster_df = pd.DataFrame(columns=['Feature', 'Node_Type', 'Attention', 'Leiden_Cluster',
                                    'Cluster_Size', 'Kept_In_Network_Plot'])
program_activity = pd.DataFrame(index=common_samples)

if G_leiden.number_of_nodes() == 0 or G_leiden.number_of_edges() == 0:
    print("Leiden clustering: no connected top-feature subgraph found, skipping cluster analysis.")
else:
    # ────────────────────────────────────────────────
    # Leiden clustering (igraph/leidenalg), weighted by mean attention
    # ────────────────────────────────────────────────
    nx_nodes = list(G_leiden.nodes())
    node_id_map = {node: i for i, node in enumerate(nx_nodes)}
    ig_edges = [(node_id_map[u], node_id_map[v]) for u, v in G_leiden.edges()]
    ig_weights = [G_leiden[u][v]['weight'] for u, v in G_leiden.edges()]

    g_ig = ig.Graph()
    g_ig.add_vertices(len(nx_nodes))
    g_ig.add_edges(ig_edges)
    g_ig.es['weight'] = ig_weights

    leiden_partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights=g_ig.es['weight'],
        resolution_parameter=args.leiden_resolution,
        seed=seed
    )
    for nx_node, cluster_id in zip(nx_nodes, leiden_partition.membership):
        G_leiden.nodes[nx_node]['leiden_cluster'] = cluster_id

    node_to_cluster = nx.get_node_attributes(G_leiden, 'leiden_cluster')
    cluster_sizes = Counter(node_to_cluster.values())
    valid_clusters = {cid for cid, size in cluster_sizes.items() if size >= args.leiden_min_cluster_size}
    print(f"Leiden clustering: found {len(cluster_sizes)} clusters, keeping {len(valid_clusters)} "
          f"with >= {args.leiden_min_cluster_size} features")

    # ────────────────────────────────────────────────
    # Save cluster composition to CSV (one row per feature)
    # ────────────────────────────────────────────────
    cluster_rows = [
        {
            'Feature': node_data['name'],
            'Node_Type': node_data['type'],
            'Attention': feature_attention_lookup.get(node_data['name'], np.nan),
            'Leiden_Cluster': node_data['leiden_cluster'],
            'Cluster_Size': cluster_sizes[node_data['leiden_cluster']],
            'Kept_In_Network_Plot': node_data['leiden_cluster'] in valid_clusters
        }
        for _, node_data in G_leiden.nodes(data=True)
    ]
    cluster_df = pd.DataFrame(cluster_rows).sort_values(
        ['Leiden_Cluster', 'Attention'], ascending=[True, False]
    ).reset_index(drop=True)
    cluster_csv_path = plots_dir + "leiden_clusters_composition.csv"
    cluster_df.to_csv(cluster_csv_path, index=False)
    print(f"Leiden clustering: cluster composition saved to {cluster_csv_path}")

    # ────────────────────────────────────────────────
    # Contract into a cluster-level network: one node per Leiden "program"
    # ────────────────────────────────────────────────
    G_cluster = nx.Graph()
    for cid in valid_clusters:
        members = [n for n, c in node_to_cluster.items() if c == cid]
        member_types = [G_leiden.nodes[n]['type'] for n in members]
        G_cluster.add_node(
            cid,
            size=len(members),
            dominant_type=max(set(member_types), key=member_types.count)
        )

    edge_accumulator = defaultdict(float)
    edge_type_accumulator = defaultdict(list)
    for u, v, edge_data in G_leiden.edges(data=True):
        cu, cv = node_to_cluster[u], node_to_cluster[v]
        if cu not in valid_clusters or cv not in valid_clusters or cu == cv:
            continue  # internal edges disappear after contraction
        key = tuple(sorted((cu, cv)))
        edge_accumulator[key] += edge_data.get('weight', 1.0)
        edge_type_accumulator[key].append(edge_data.get('edge_type', 'unknown'))

    for (cu, cv), weight in edge_accumulator.items():
        types = edge_type_accumulator[(cu, cv)]
        dominant_type = max(set(types), key=types.count)
        G_cluster.add_edge(cu, cv, weight=weight, edge_type=dominant_type, n_edges=len(types))

    # ────────────────────────────────────────────────
    # Plot: contracted cluster network
    # ────────────────────────────────────────────────
    if G_cluster.number_of_nodes() == 0:
        print("Leiden clustering: no clusters passed the minimum size filter, skipping network plot.")
    else:
        # G_cluster's edge weight is a SUM over every contracted original edge, so it can
        # range from 1 up into the hundreds - robust_layout() normalizes and inverts this
        # internally before it ever reaches Kamada-Kawai (see its docstring). Edge width
        # still encodes the true raw weight visually (see `widths` below).
        pos = robust_layout(G_cluster, seed=seed)
        n_cluster_nodes = G_cluster.number_of_nodes()
        cmap = plt.cm.get_cmap('tab20', n_cluster_nodes)
        node_color_by_id = {n: cmap(i) for i, n in enumerate(G_cluster.nodes())}

        def _size(n):
            return 120 + 400 * np.log1p(G_cluster.nodes[n]['size'])

        cluster_weights = np.array([G_cluster[u][v]['weight'] for u, v in G_cluster.edges()])
        widths = 0.2 + 2.5 * (cluster_weights / cluster_weights.max()) if len(cluster_weights) > 0 else 0.5

        # Top 5 highest-attention features per cluster, for the on-plot labels below.
        # cluster_df is already sorted by Attention (descending) within each cluster.
        top5_labels_per_cluster = (
            cluster_df[cluster_df['Kept_In_Network_Plot']]
            .groupby('Leiden_Cluster')['Feature']
            .apply(lambda s: s.head(5).tolist())
            .to_dict()
        )

        fig, ax = plt.subplots(figsize=(9.5, 9.5))
        nx.draw_networkx_edges(G_cluster, pos, width=widths, alpha=0.18, edge_color="#4a4a4a", ax=ax)
        # Marker shape encodes each program's dominant omic type, an attribute that
        # was already computed but previously unused visually.
        for dom_type in OMIC_NAMES:
            marker = OMIC_MARKERS_BY_NAME[dom_type]
            nodes_of_type = [n for n in G_cluster.nodes() if G_cluster.nodes[n]['dominant_type'] == dom_type]
            if not nodes_of_type:
                continue
            nx.draw_networkx_nodes(
                G_cluster, pos, nodelist=nodes_of_type, node_shape=marker,
                node_size=[_size(n) for n in nodes_of_type],
                node_color=[node_color_by_id[n] for n in nodes_of_type],
                edgecolors="black", linewidths=0.5, ax=ax
            )
        # Annotate each cluster with its top 5 highest-attention features, on a
        # white background so the labels stay legible over the muted edges.
        for cid, (x, y) in pos.items():
            labels = top5_labels_per_cluster.get(cid, [])
            if not labels:
                continue
            ax.annotate(
                "\n".join(labels),
                xy=(x, y),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=5,
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#cccccc', linewidth=0.5, alpha=0.85),
                zorder=5
            )
        _frame_axes_to_pos(ax, pos)
        ax.axis("off")
        ax.set_title(
            f"Multi-omic attention clusters (Leiden, top {len(top_feature_node_idx)} features, "
            f"{n_cluster_nodes} programs)",
            fontsize=12
        )
        # Legends: marker shape = dominant omic type, reference circles = program size.
        shape_legend = ax.legend(
            handles=[Line2D([0], [0], marker=OMIC_MARKERS_BY_NAME[_name], color='w', markerfacecolor='#888888',
                             markeredgecolor='black', markersize=9, label=f'{_name}-dominant program')
                     for _name in OMIC_NAMES],
            loc='upper left', fontsize=8, title='Program type', title_fontsize=8
        )
        ax.add_artist(shape_legend)
        size_ref = sorted(set(G_cluster.nodes[n]['size'] for n in G_cluster.nodes()))
        size_ref = [size_ref[0], size_ref[len(size_ref) // 2], size_ref[-1]] if len(size_ref) > 2 else size_ref
        ax.legend(
            handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc',
                             markeredgecolor='black', markersize=np.sqrt(120 + 400 * np.log1p(s)) / 3.5,
                             label=f'{s} features')
                     for s in size_ref],
            loc='lower left', fontsize=8, title='Program size', title_fontsize=8, labelspacing=1.2
        )
        plt.tight_layout()
        cluster_pdf_path = plots_dir + "leiden_cluster_network.pdf"
        plt.savefig(cluster_pdf_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Leiden clustering: cluster network plot saved to {cluster_pdf_path}")

    # ────────────────────────────────────────────────
    # Per-cluster attention subgraphs: one PDF page per Leiden cluster, nodes/edges
    # colored blue (low) -> red (high) by attention.
    # ────────────────────────────────────────────────
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as path_effects
    from matplotlib.backends.backend_pdf import PdfPages

    def draw_cluster_attention_subgraph(G_full, target_cluster, figsize=(11, 11), seed=7):
        """
        Subsets G_full to one Leiden cluster and draws its attention subnetwork: each
        omic gets its own marker shape (OMIC_MARKERS_BY_NAME), edges/nodes colored
        blue->red by attention (RdBu_r). Returns the matplotlib Figure, or None if the
        cluster is empty.
        """
        cluster_nodes = [n for n, d in G_full.nodes(data=True) if d.get('leiden_cluster') == target_cluster]
        if not cluster_nodes:
            return None

        G_sub = G_full.subgraph(cluster_nodes).copy()

        pos = robust_layout(G_sub, seed=seed)

        def norm_val(v, mn, mx):
            return (v - mn) / (mx - mn) if mx > mn else 0.5

        # Node attention comes from the 'attention' node attribute set when G_leiden was
        # built above (the per-feature attention score) - without it every node would
        # normalize to the same 0.5 midpoint and render as flat gray regardless of shape.
        node_att_values = [G_sub.nodes[n].get('attention', 0.0) for n in G_sub.nodes]
        min_node_att, max_node_att = (min(node_att_values), max(node_att_values)) if node_att_values else (0.0, 1.0)

        edge_weights_sub = [d.get('weight', 1.0) for _, _, d in G_sub.edges(data=True)]
        min_edge_w, max_edge_w = (min(edge_weights_sub), max(edge_weights_sub)) if edge_weights_sub else (1.0, 1.0)

        cmap = plt.cm.get_cmap('RdBu_r')

        fig, ax = plt.subplots(figsize=figsize, facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        ax.axis('off')

        # ─── Edges (color + width mapped to attention weight) ───
        for u, v, d in G_sub.edges(data=True):
            w = d.get('weight', 1.0)
            norm_w = norm_val(w, min_edge_w, max_edge_w)
            edge_color = cmap(norm_w)
            edge_width = 0.5 + norm_w * 4.5
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color=edge_color, lw=edge_width, alpha=0.6, zorder=1)

        # ─── Nodes (split by omic for different marker shapes) ───
        nodes_by_omic = {_name: [n for n in G_sub.nodes if G_sub.nodes[n].get('type') == _name] for _name in OMIC_NAMES}
        node_groups = [
            {'nodes': nodes_by_omic[_name], 'marker': OMIC_MARKERS_BY_NAME[_name],
             'base_size': 200, 'scale_size': 800, 'label': _name}
            for _name in OMIC_NAMES
        ]
        for group in node_groups:
            if not group['nodes']:
                continue
            g_nodes = group['nodes']
            g_atts = [G_sub.nodes[n].get('attention', 0.0) for n in g_nodes]
            g_norm_atts = [norm_val(a, min_node_att, max_node_att) for a in g_atts]
            node_colors_sub = [cmap(na) for na in g_norm_atts]
            node_sizes = [group['base_size'] + na * group['scale_size'] for na in g_norm_atts]
            border_colors = [tuple(max(0, c * 0.6) for c in rgba[:3]) + (1.0,) for rgba in node_colors_sub]
            ax.scatter(
                [pos[n][0] for n in g_nodes],
                [pos[n][1] for n in g_nodes],
                s=node_sizes,
                c=node_colors_sub,
                marker=group['marker'],
                edgecolors=border_colors,
                linewidths=1.2,
                label=group['label'],
                zorder=3
            )

        # ─── Labels, white-halo text for legibility over edges/nodes ───
        for n in G_sub.nodes:
            x, y = pos[n]
            name = G_sub.nodes[n].get('name', str(n))
            txt = ax.text(x, y, name, color='#000000', fontsize=6, fontweight='bold',
                          ha='center', va='center', zorder=4)
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='#FFFFFF')])

        # Explicit neutral-gray legend proxies for shape - the actual markers are colored
        # by attention (RdBu_r, see colorbar below), so a legend built from those same
        # per-point colors would misleadingly imply color varies by omic too.
        ax.legend(
            handles=[Line2D([0], [0], marker=OMIC_MARKERS_BY_NAME[_name], color='w', markerfacecolor='#999999',
                             markeredgecolor='black', markersize=10, label=_name)
                     for _name in OMIC_NAMES],
            loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#E0E0E0', fontsize=10
        )

        norm = mcolors.Normalize(
            vmin=min_node_att,
            vmax=max_node_att if max_node_att > min_node_att else min_node_att + 1
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb_ax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
        cb = plt.colorbar(sm, cax=cb_ax, orientation='horizontal')
        cb.set_label('Attention Score Intensity (Blue=Low, Red=High)', fontsize=9, labelpad=6)
        cb.ax.tick_params(labelsize=7)

        ax.set_title(
            f'Multi-Omic Attention Subgraph — Leiden Cluster {target_cluster}\n'
            f'({" · ".join(f"{len(nodes_by_omic[_name])} {_name}" for _name in OMIC_NAMES)})',
            fontsize=13, fontweight='bold', pad=18
        )
        _frame_axes_to_pos(ax, pos)
        ax.axis('off')  # re-assert: set_aspect inside _frame_axes_to_pos doesn't affect visibility
        return fig

    if not valid_clusters:
        print("Leiden clustering: no clusters passed the minimum size filter, skipping per-cluster subgraphs.")
    else:
        cluster_subgraphs_pdf_path = plots_dir + "leiden_cluster_subgraphs.pdf"
        with PdfPages(cluster_subgraphs_pdf_path) as pdf:
            for cid in sorted(valid_clusters):
                fig = draw_cluster_attention_subgraph(G_leiden, target_cluster=cid)
                if fig is None:
                    continue
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
                plt.close(fig)
        print(f"Leiden clustering: {len(valid_clusters)} per-cluster attention subgraphs saved to {cluster_subgraphs_pdf_path}")

    # ────────────────────────────────────────────────
    # Program activity: for each sample, weight every program member feature's
    # normalized input value by its attention score and sum within the program.
    # activity(sample, program) = sum over features f in program of
    #                              attention(f) * normalized_value(f, sample)
    # ────────────────────────────────────────────────
    if not valid_clusters:
        print("Leiden clustering: no clusters passed the minimum size filter, skipping program activity scores.")
    else:
        # `samples_stacked` is the (every omic's features stacked) x samples matrix
        # built from the raw per-omic input matrices earlier in the script,
        # row-aligned with the same global feature index used throughout this section
        # (FEATURE_INDEX_TO_OMIC[idx] gives idx's omic).
        program_activity = pd.DataFrame(index=common_samples)
        for cid in sorted(valid_clusters):
            member_idx = [n for n, c in node_to_cluster.items() if c == cid]
            member_attention = np.array([G_leiden.nodes[n]['attention'] for n in member_idx])
            member_expression = samples_stacked[member_idx, :]  # (n_members, n_samples)
            program_activity[f'Program_{cid}'] = member_attention @ member_expression
        program_activity.index.name = 'ID'
        program_activity_csv_path = plots_dir + "leiden_program_activity_scores.csv"
        program_activity.to_csv(program_activity_csv_path)
        print(f"Leiden clustering: program activity scores saved to {program_activity_csv_path}")



#-----------------------------------------------------------------------
#
#----------------- Embedding Plots ------------------------------
#
#-----------------------------------------------------------------------

#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
#import umap
#import torch
#import numpy as np
#import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
# ----------------------------------
# Helper: Project to 2D with UMAP
# ----------------------------------
def reduce_embedding(embedding, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(embedding)

# Convert to numpy if necessary
def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x

# ----------------------------------
# Prepare Embeddings & Labels
# ----------------------------------
labels = metadata.CONDITION.reset_index(drop=True)
node_names = omic_dfs[OMIC_NAMES[0]].columns[1:]

embeddings = {
    'Cell Nodes': to_numpy(node_rep[node_type == len(OMIC_NAMES)]),
    'g128 Embedding': to_numpy(g128_embeddings),
    'g64 Embedding': to_numpy(g64_embeddings),
    'g32 Embedding': to_numpy(g32_embeddings),
    'Class Rep': to_numpy(class_rep)
}
# ----------------------------------
# Create Multi-panel Figure
# ----------------------------------
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])  # Slightly more space for bottom
# Plot 1: Cell Nodes UMAP (Top row)
ax1 = fig.add_subplot(gs[0, :])
cell_nodes_umap = reduce_embedding(embeddings['Cell Nodes'], n_components=2)
for cond in labels.unique():
    idx = labels == cond
    ax1.scatter(cell_nodes_umap[idx, 0], cell_nodes_umap[idx, 1], label=cond, s=15, color=CONDITION_COLORS[str(cond)])

ax1.set_title('UMAP of Cell Node Embeddings')
ax1.legend(markerscale=2, fontsize=8)
# Plot 2: g128 Embedding UMAP
ax2 = fig.add_subplot(gs[1, 0])
umap128 = reduce_embedding(embeddings['g128 Embedding'], n_components=2)
for cond in labels.unique():
    idx = labels == cond
    ax2.scatter(umap128[idx, 0], umap128[idx, 1], label=cond, s=10, color=CONDITION_COLORS[str(cond)])

ax2.set_title('UMAP of g128 Embedding')
# Plot 3: g64 Embedding UMAP
ax3 = fig.add_subplot(gs[1, 1])
umap64 = reduce_embedding(embeddings['g64 Embedding'], n_components=2)
for cond in labels.unique():
    idx = labels == cond
    ax3.scatter(umap64[idx, 0], umap64[idx, 1], label=cond, s=10, color=CONDITION_COLORS[str(cond)])

ax3.set_title('UMAP of g64 Embedding')
# Plot 4: g32 Embedding UMAP
ax4 = fig.add_subplot(gs[1, 2])
umap32 = reduce_embedding(embeddings['g32 Embedding'], n_components=2)
for cond in labels.unique():
    idx = labels == cond
    ax4.scatter(umap32[idx, 0], umap32[idx, 1], label=cond, s=10, color=CONDITION_COLORS[str(cond)])

ax4.set_title('UMAP of g32 Embedding')
# Plot 5: Class Rep — choose UMAP or raw depending on dimensions
class_rep_np = embeddings['Class Rep']
D = class_rep_np.shape[1]
if D == 2:
    ax5 = fig.add_subplot(gs[2, :])
    for cond in labels.unique():
        idx = labels == cond
        ax5.scatter(class_rep_np[idx, 0], class_rep_np[idx, 1], label=cond, s=15, color=CONDITION_COLORS[str(cond)])

    ax5.set_title('2D Class Representation')
    ax5.legend(markerscale=2, fontsize=8)
elif D == 3:
    ax5 = fig.add_subplot(gs[2, :], projection='3d')
    for cond in labels.unique():
        idx = labels == cond
        ax5.scatter(class_rep_np[idx, 0], class_rep_np[idx, 1], class_rep_np[idx, 2], label=cond, s=15,
                     color=CONDITION_COLORS[str(cond)])

    ax5.set_title('3D Class Representation')
    ax5.legend(markerscale=2, fontsize=8)
    ax5.view_init(elev=20, azim=45)
    # Minimalist 3D panes (default matplotlib gray fill/gridlines read as cluttered in print).
    for axis in (ax5.xaxis, ax5.yaxis, ax5.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor('#dddddd')
    ax5.grid(False)
else:
    ax5 = fig.add_subplot(gs[2, :])
    reduced = reduce_embedding(class_rep_np, n_components=2)
    for cond in labels.unique():
        idx = labels == cond
        ax5.scatter(reduced[idx, 0], reduced[idx, 1], label=cond, s=15, color=CONDITION_COLORS[str(cond)])

    ax5.set_title(f'UMAP of Class Rep ({D}D)')
    ax5.legend(markerscale=2, fontsize=8)

# Final layout and save
plt.tight_layout()
plt.savefig(plots_dir+'embeddings_evaluations_figure.pdf')
plt.close()



#-----------------------------------------------------------------------
#
#----------------- Program Activity Report (per multi-omic program) ----
#
#-----------------------------------------------------------------------
# One PDF page per program, answering: "is there a real difference in this
# program's activity between conditions, and do we have the power to detect it?"
# Panels: (1) sample UMAP colored by condition, (2) same UMAP colored by program
# activity, (3) activity boxplot by condition with a non-parametric test + effect
# size + group sizes (a p-value alone says nothing about whether a null result is
# a real null or just an underpowered comparison - effect size and n give that
# context), (4) a decoupleR-style barplot of each program's feature contributions,
# signed by which condition group the feature trends higher in.

if not valid_clusters:
    print("Leiden clustering: no clusters passed the minimum size filter, skipping program activity report.")
else:
    from scipy.stats import mannwhitneyu, kruskal
    # matplotlib.colors (mcolors) and PdfPages were already imported earlier in the
    # Leiden clustering section above.

    condition_labels = metadata['CONDITION'].reset_index(drop=True).astype(str).values
    unique_conditions = sorted(pd.unique(condition_labels))
    condition_palette = CONDITION_COLORS  # shared across every condition-colored plot in this script

    def _contrast_groups(values, groups, group_names):
        # The two groups with the most different mean activity - used to give the
        # feature-contribution barplot a consistent, well-defined up/down direction
        # even when there are more than two conditions.
        means = {g: values[groups == g].mean() for g in group_names}
        return min(means, key=means.get), max(means, key=means.get)

    def _mannwhitney_effect(values, groups, g_low, g_high):
        v_low = values[groups == g_low]
        v_high = values[groups == g_high]
        stat, p = mannwhitneyu(v_low, v_high, alternative='two-sided')
        n_low, n_high = len(v_low), len(v_high)
        effect = abs(1 - (2 * stat) / (n_low * n_high))  # |rank-biserial correlation|
        return {'p': p, 'effect': effect, 'n_low': n_low, 'n_high': n_high}

    program_report_pdf_path = plots_dir + "leiden_program_activity_report.pdf"
    with PdfPages(program_report_pdf_path) as pdf:
        for cid in sorted(valid_clusters):
            activity_values = program_activity[f'Program_{cid}'].values

            member_idx = [n for n, c in node_to_cluster.items() if c == cid]
            member_names = [G_leiden.nodes[n]['name'] for n in member_idx]

            g_low, g_high = _contrast_groups(activity_values, condition_labels, unique_conditions)
            pairwise = _mannwhitney_effect(activity_values, condition_labels, g_low, g_high)
            stats_lines = [
                f"Mann-Whitney U: p={pairwise['p']:.3g}, |r|={pairwise['effect']:.2f}, "
                f"higher in {g_high} (n={pairwise['n_low']} vs {pairwise['n_high']})"
            ]
            if len(unique_conditions) > 2:
                kw_stat, kw_p = kruskal(*[activity_values[condition_labels == g] for g in unique_conditions])
                k, n_total = len(unique_conditions), len(activity_values)
                eps2 = (kw_stat - k + 1) / (n_total - k) if n_total > k else np.nan
                stats_lines.insert(0, f"Kruskal-Wallis (all {k} groups): p={kw_p:.3g}, ε²={eps2:.2f}")

            # Sign each feature's attention score by whether its normalized input value
            # trends higher in the high- or low-activity group (proxy for logFC direction;
            # the inputs are already filtered/normalized, not raw counts, so only the sign
            # of the group difference - not a literal fold change - is well defined here).
            low_mask = condition_labels == g_low
            high_mask = condition_labels == g_high
            signed_scores = []
            for idx in member_idx:
                mean_diff = samples_stacked[idx, high_mask].mean() - samples_stacked[idx, low_mask].mean()
                sign = np.sign(mean_diff) or 1.0
                signed_scores.append(G_leiden.nodes[idx]['attention'] * sign)
            signed_scores = np.array(signed_scores)
            order = np.argsort(signed_scores)[::-1]  # highest positive -> lowest negative
            ranked_names = [member_names[i] for i in order]
            ranked_scores = signed_scores[order]

            fig_height = max(10, 5 + 0.28 * len(member_idx))
            fig = plt.figure(figsize=(15, fig_height))
            gs_prog = gridspec.GridSpec(2, 3, height_ratios=[1, 1.3], hspace=0.4, wspace=0.35)

            # Panel 1: sample UMAP colored by condition
            ax0 = fig.add_subplot(gs_prog[0, 0])
            for cond in unique_conditions:
                idx_mask = condition_labels == cond
                ax0.scatter(cell_nodes_umap[idx_mask, 0], cell_nodes_umap[idx_mask, 1],
                            color=condition_palette[cond], label=cond, s=20, alpha=0.85)
            ax0.set_title('Sample UMAP – Condition', fontsize=11)
            ax0.legend(fontsize=7, markerscale=1.2)
            ax0.set_xlabel('UMAP1'); ax0.set_ylabel('UMAP2')

            # Panel 2: same UMAP colored by this program's activity score
            ax1 = fig.add_subplot(gs_prog[0, 1])
            sc = ax1.scatter(cell_nodes_umap[:, 0], cell_nodes_umap[:, 1],
                              c=activity_values, cmap='viridis', s=20)
            plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label='Activity')
            ax1.set_title(f'Sample UMAP – Program {cid} Activity', fontsize=11)
            ax1.set_xlabel('UMAP1'); ax1.set_ylabel('UMAP2')

            # Panel 3: activity boxplot by condition + non-parametric test/effect size
            ax2 = fig.add_subplot(gs_prog[0, 2])
            box_data = [activity_values[condition_labels == cond] for cond in unique_conditions]
            bp = ax2.boxplot(box_data, labels=unique_conditions, patch_artist=True, showfliers=False)
            for patch, cond in zip(bp['boxes'], unique_conditions):
                patch.set_facecolor(condition_palette[cond])
                patch.set_alpha(0.6)
            for i, cond in enumerate(unique_conditions):
                y = activity_values[condition_labels == cond]
                x = np.random.normal(i + 1, 0.05, size=len(y))
                ax2.scatter(x, y, s=10, color='black', alpha=0.5, zorder=3)
            ax2.set_title("\n".join(stats_lines), fontsize=8)
            ax2.set_xlabel('Condition'); ax2.set_ylabel('Program Activity')

            # Panel 4: decoupleR-style feature contribution barplot
            ax3 = fig.add_subplot(gs_prog[1, :])
            max_abs = max(np.max(np.abs(ranked_scores)), 1e-9) if len(ranked_scores) else 1.0
            bar_norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            bar_cmap = plt.cm.get_cmap('RdBu_r')
            bar_colors = [bar_cmap(bar_norm(v)) for v in ranked_scores]
            y_pos = np.arange(len(ranked_names))[::-1]
            ax3.barh(y_pos, ranked_scores, color=bar_colors, edgecolor='black', linewidth=0.3)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(ranked_names, fontsize=6)
            ax3.axvline(0, color='black', linewidth=0.8)
            ax3.set_xlabel(f'Attention score, signed by direction ({g_low} → {g_high})')
            ax3.set_title(f'Program {cid} feature contributions ({len(member_idx)} features)', fontsize=11)

            fig.suptitle(f'Multi-Omic Program {cid} Activity Report', fontsize=15, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight', dpi=200)
            plt.close(fig)
    print(f"Leiden clustering: program activity report saved to {program_report_pdf_path}")




#-----------------------------------------------------------------------
#
#----------------- Sample Features Interpretability --------------------
#
#-----------------------------------------------------------------------

#import numpy as np
#import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
#import seaborn as sns
#import matplotlib.pyplot as plt
#from scipy.cluster.hierarchy import linkage, leaves_list
#from scipy.spatial.distance import pdist
#import os





#-----------------------------------------------------------------------
#
#----------------- Network Visualization -------------------------------
#
#-----------------------------------------------------------------------

#import networkx as nx
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#from matplotlib.lines import Line2D
#from matplotlib.patches import Patch

# Step 1: Select top 100 features per omic by attention
top_n = 100
top_by_omic = {_name: omic_node_attention[_name].sort_values(by='Attention', ascending=False).head(top_n)
                for _name in OMIC_NAMES}
attention_network = pd.concat([df2.iloc[:, [0, 1]], pd.DataFrame(edge_attn.T)],axis=1)
attention_network.columns = ['source', 'target', 'weight']

# Get all samples
samples = metadata[['ID', 'CONDITION']].reset_index(drop=True)

# Create node lists
node_labels = pd.concat(
    [top_by_omic[_name][f'{_name}_ID'].reset_index(drop=True) for _name in OMIC_NAMES]
    + [samples['ID'].reset_index(drop=True)]
)
node_attention = np.concatenate(
    [top_by_omic[_name]['Attention'].values for _name in OMIC_NAMES]
    + [np.zeros(len(samples))]  # Samples may not have attention scores
)
node_types = np.concatenate(
    [np.full(len(top_by_omic[_name]), _oi) for _oi, _name in enumerate(OMIC_NAMES)]
    + [np.full(len(samples), len(OMIC_NAMES))]  # Samples get the next type index after every omic
)
node_shapes = (
    sum(([OMIC_PLOTLY_SHAPES_BY_NAME[_name]] * len(top_by_omic[_name]) for _name in OMIC_NAMES), []) +
    ['circle'] * len(samples)
)

# Step 2: Build Graph
G = nx.Graph()

num_samples = len(samples)
sample_type_idx = len(OMIC_NAMES)
# Node ids are offset by each omic's FULL feature count (matching the global feature
# index used throughout the script - see FEATURE_INDEX_TO_OMIC), not by the top-N
# subset size, so they line up with attention_network's source/target ids.
omic_indices = {}
_cum = 0
for _name in OMIC_NAMES:
    omic_indices[_name] = np.array(top_by_omic[_name].index) + _cum
    _cum += omic_node_attention[_name].shape[0]
sample_indices = np.array(range(num_samples)) + _cum
all_selected_indices = np.concatenate([omic_indices[_name] for _name in OMIC_NAMES] + [sample_indices])
# Add nodes with attributes
#for i, (label, shape, node_type, att) in zip(all_selected_indices, zip(node_labels, node_shapes, node_types, node_attention)):
#    G.add_node(i, label=label, shape=shape, node_type=node_type, importance=att)
for i, (label, shape, node_type) in zip(all_selected_indices, zip(node_labels, node_shapes, node_types)):
    G.add_node(i, label=label, shape=shape, node_type=node_type)

# Step 3: Filter edges to include only those between selected nodes
valid_nodes = set(all_selected_indices)
attention_filtered = attention_network[
    (attention_network['source'].isin(valid_nodes)) & (attention_network['target'].isin(valid_nodes))
].copy()

# A single global 80th-percentile cutoff is edge-type-blind: HGT attention weights come
# from per-relation-type attention heads, so gene-gene, gene-metabolite, gene-sample,
# metabolite-sample, and sample-sample edges routinely sit on very different absolute
# scales. Thresholding everything against one shared value lets whichever edge type
# happens to have the largest typical attention (in practice, usually sample-sample)
# crowd out survivors from every other type - most genes/metabolites are then left with
# at most one surviving edge each (scattered, near-isolated dyads once plotted), while a
# handful of sample nodes end up densely interconnected in one tight clump. Applying the
# same 80th-percentile cutoff *within* each edge type instead keeps every relation type
# proportionally represented among the survivors.
idx_to_type = dict(zip(all_selected_indices, node_types))
attention_filtered['source_type'] = attention_filtered['source'].map(idx_to_type)
attention_filtered['target_type'] = attention_filtered['target'].map(idx_to_type)
attention_filtered['edge_category'] = list(zip(
    np.minimum(attention_filtered['source_type'], attention_filtered['target_type']),
    np.maximum(attention_filtered['source_type'], attention_filtered['target_type'])
))
kept_chunks = []
for _category, group in attention_filtered.groupby('edge_category'):
    cat_threshold = np.percentile(group['weight'], 80)
    kept_chunks.append(group[group['weight'] >= cat_threshold])
attention_filtered = (
    pd.concat(kept_chunks, ignore_index=True) if kept_chunks else attention_filtered.iloc[0:0]
)

# Normalize edge weights for layout
#max_weight = attention_filtered['weight'].max()
#attention_filtered['weight_norm'] = attention_filtered['weight'] / max_weight

# Add edges with weights
for i, j in zip(attention_filtered['source'], attention_filtered['target']):
    G.add_edge(i, j, weight=1)

# Step 4: Remove isolated nodes
isolated_nodes = [node for node in G.nodes if G.degree[node] == 0]
G.remove_nodes_from(isolated_nodes)

# Update node attributes after removing isolated nodes
node_labels = [G.nodes[node]['label'] for node in G.nodes]
node_shapes = [G.nodes[node]['shape'] for node in G.nodes]
node_types = [G.nodes[node]['node_type'] for node in G.nodes]
#node_attention = [G.nodes[node]['importance'] for node in G.nodes]

# Step 5: Determine edge types
type_idx_to_name = dict(enumerate(OMIC_NAMES))

def _node_type_name(t):
    return 'sample' if t == sample_type_idx else type_idx_to_name[t]

edge_types = []
for u, v in G.edges:
    s_name = _node_type_name(G.nodes[u]['node_type'])
    t_name = _node_type_name(G.nodes[v]['node_type'])
    edge_types.append(f'{s_name}-{s_name}' if s_name == t_name else '-'.join(sorted((s_name, t_name))))

# Step 6: Plotting
plt.figure(figsize=(12, 12))

# Node positions
# After the top-N feature selection plus the 80th-percentile attention threshold, G
# regularly contains isolated nodes or several disconnected components (a few hub/
# sample nodes, many weakly- or un-linked ones). See the robust_layout() docstring
# near the top of the file for why that specifically breaks a plain Kamada-Kawai
# call, and how laying out each component separately fixes it.
pos = robust_layout(G, seed=42)
# Guard-rail: catch a degenerate/collapsed layout before it silently ships in a PDF.
_pos_xs, _pos_ys = zip(*pos.values())
if np.std(_pos_xs) < 0.05 or np.std(_pos_ys) < 0.05:
    print(f"WARNING: full network layout looks degenerate "
          f"(std_x={np.std(_pos_xs):.3f}, std_y={np.std(_pos_ys):.3f}) - "
          f"check graph connectivity before plotting.")

#import plotly.graph_objects as go

# node_shapes already stores final Plotly symbol names (OMIC_PLOTLY_SHAPES_BY_NAME /
# 'circle' for samples - see Step 1 above), so no shape_map translation is needed here.

# Color maps - reuse the same colorblind-safe, cross-figure-consistent palette as
# every other plot in this script (see PUBLICATION STYLE at the top of the file).
condition_color_map = {cond: hex_to_rgb_string(color) for cond, color in CONDITION_COLORS.items()}
node_type_colors = {_oi: hex_to_rgb_string(OMIC_COLORS[_name]) for _oi, _name in enumerate(OMIC_NAMES)}
node_type_colors[sample_type_idx] = None  # Samples by condition

edge_type_colors = {}
for _name in OMIC_NAMES:
    edge_type_colors[f'{_name}-{_name}'] = hex_to_rgb_string(OMIC_COLORS[_name])
for _a, _b in itertools.combinations(OMIC_NAMES, 2):
    # neutral - cross-omic edges shouldn't fight for attention
    edge_type_colors['-'.join(sorted((_a, _b)))] = 'rgb(140, 140, 140)'
for _name in OMIC_NAMES:
    # faint - de-emphasize the dense omic-sample fan-out
    edge_type_colors['-'.join(sorted((_name, 'sample')))] = 'rgb(210, 210, 210)'
edge_type_colors['sample-sample'] = 'rgb(60, 60, 60)'

# Prepare node data
node_x, node_y, node_colors, node_symbols, node_hover = [], [], [], [], []
for node in G.nodes(data=True):
    x, y = pos[node[0]]
    node_x.append(x)
    node_y.append(y)
    node_type = node[1]['node_type']
    label = node[1]['label']
    if node_type == sample_type_idx:
        condition = metadata.loc[metadata.ID == label, 'CONDITION'].values[0]
        node_colors.append(condition_color_map[condition])
        node_hover.append(f"Sample: {label}<br>Condition: {condition}")
    else:
        node_colors.append(node_type_colors[node_type])
        node_hover.append(f"{type_idx_to_name[node_type]}: {label}")
    node_symbols.append(node[1]['shape'])

# Prepare edge data by type
edge_traces = []
for edge_type in edge_type_colors:
    edge_x, edge_y = [], []
    for (u, v), et in zip(G.edges(), edge_types):
        if et == edge_type:
            try:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            except KeyError:
                print(f"Warning: Node {u} or {v} not in pos")
                continue
    if edge_x:
        edge_traces.append(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color=edge_type_colors[edge_type]),
            hoverinfo='none',
            name=edge_type.capitalize(),
            showlegend=True
        ))

# Create figure
fig = go.Figure()

# Add edge traces
for trace in edge_traces:
    fig.add_trace(trace)

# Add node trace
fig.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers',
    marker=dict(size=8, color=node_colors, symbol=node_symbols, line=dict(width=0.5, color='black')),
    text=node_hover,
    hoverinfo='text',
    name='Nodes',
    showlegend=False
))

# Add legend for nodes and conditions
legend_items = [
    go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=hex_to_rgb_string(OMIC_COLORS[_name]), symbol=OMIC_PLOTLY_SHAPES_BY_NAME[_name]),
        name=_name
    )
    for _name in OMIC_NAMES
] + [
    go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='gray', symbol='circle'),
        name='Sample'
    )
]
for cond, color in condition_color_map.items():
    legend_items.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=color, symbol='circle'),
        name=f'Sample: {cond}'
    ))

for item in legend_items:
    fig.add_trace(item)

# Update layout
fig.update_layout(
    title=f"Network Visualization: Top {top_n} Features per Omic ({', '.join(OMIC_NAMES)}), All Samples",
    showlegend=True,
    legend=dict(x=1, y=1, xanchor='left', yanchor='top'),
    hovermode='closest',
    plot_bgcolor='white',
    width=800,
    height=800,
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

# Save interactive plot
fig.write_html(plots_dir + "network_interactive.html")

# Debug: Print graph stats
print(f"Number of nodes: {len(G.nodes)}")
print(f"Number of edges: {len(G.edges)}")
print(f"Edge type counts:\n{pd.Series(edge_types).value_counts()}")
degrees = pd.Series(dict(G.degree()), name='degree')
print(f"Node degrees:\n{degrees.describe()}")
print(f"High-degree nodes (>50):\n{degrees[degrees > 50]}")



################################################################
#
# SUPPLEMENTARY VISUALIZATIONS
#
################################################################
# Two extra, optional figures per core visualization above, saved separately under
# supplementary_dir so they never clutter the main plots/ folder. Each block only
# depends on data already computed earlier in this script, and skips gracefully
# (printing why) when that data isn't available (e.g. no Leiden programs passed
# the size filter).

from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu, kruskal

print("\n=== Generating supplementary visualizations ===")

# ---------------------------------------------------------------------------
# 1. Training loss -> (a) loss vs. LR schedule, (b) relative loss-component share
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax1.plot(total_losses, color=OKABE_ITO[0], linewidth=2, label='Total Loss')
ax1.plot(val_losses, color=OKABE_ITO[1], linewidth=2, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper left', fontsize=9)
ax2 = ax1.twinx()
ax2.plot(lr_history, color='#555555', linewidth=1.3, linestyle='--', label='Learning Rate')
ax2.set_ylabel('Learning Rate')
ax2.spines['right'].set_visible(True)  # the one spine that actually carries information here
ax2.legend(loc='upper right', fontsize=9)
ax1.set_title('Training Loss vs. Learning-Rate Schedule')
plt.tight_layout()
plt.savefig(supplementary_dir + "loss_lr_schedule.pdf")
plt.close(fig)

ce_weighted = 0.3 * np.abs(np.array(cross_entropy_losses))
cos_weighted = 0.7 * np.abs(np.array(cosine_losses))
total_weighted_safe = np.where((ce_weighted + cos_weighted) == 0, 1, ce_weighted + cos_weighted)
ce_share = 100 * ce_weighted / total_weighted_safe
cos_share = 100 * cos_weighted / total_weighted_safe
fig, ax = plt.subplots(figsize=(9, 5))
epochs_x = np.arange(1, len(ce_share) + 1)
ax.stackplot(epochs_x, ce_share, cos_share, colors=[OKABE_ITO[3], OKABE_ITO[5]],
             labels=['Cross Entropy (weighted)', 'Cosine (weighted)'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Share of Total Loss (%)')
ax.set_ylim(0, 100)
ax.set_title('Relative Contribution of Each Loss Term Over Training')
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
plt.tight_layout()
plt.savefig(supplementary_dir + "loss_component_share.pdf")
plt.close(fig)
print("Supplementary: training-loss diagnostics saved.")

# ---------------------------------------------------------------------------
# 2. Confusion matrix -> (a) per-class ROC curves, (b) precision/recall/F1 bars
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
n_classes = one_hot_labels.shape[1]
for i in range(n_classes):
    if one_hot_labels[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(one_hot_labels[:, i], class_probs[:, i])
    auc_i = roc_auc_per_class.get(i, np.nan)
    ax.plot(fpr, tpr, color=OKABE_ITO[i % len(OKABE_ITO)], linewidth=1.8,
            label=f'{condition_names[i]} (AUC={auc_i:.2f})')
ax.plot([0, 1], [0, 1], color='#aaaaaa', linestyle='--', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'Per-Class ROC Curves (macro AUC = {roc_auc:.2f})')
ax.legend(loc='lower right', fontsize=8)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(supplementary_dir + "roc_curves.pdf")
plt.close(fig)

precision, recall, f1_per_class, support = precision_recall_fscore_support(
    true_labels, pred_labels, labels=np.arange(n_classes), zero_division=0
)
fig, ax = plt.subplots(figsize=(max(6, 1.3 * n_classes), 5))
x = np.arange(n_classes)
width = 0.25
ax.bar(x - width, precision, width, label='Precision', color=OKABE_ITO[0])
ax.bar(x, recall, width, label='Recall', color=OKABE_ITO[1])
ax.bar(x + width, f1_per_class, width, label='F1', color=OKABE_ITO[2])
ax.set_xticks(x)
ax.set_xticklabels(condition_names, rotation=20, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Per-Class Classification Performance')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(supplementary_dir + "per_class_performance.pdf")
plt.close(fig)
print("Supplementary: classifier ROC / per-class performance saved.")

# ---------------------------------------------------------------------------
# 3. Interpretability -> (a) attention concentration by omic, (b) attention distribution by omic
# ---------------------------------------------------------------------------
def _cumulative_pct(values):
    s = np.sort(np.asarray(values))[::-1]
    c = np.cumsum(s)
    total = c[-1] if len(c) and c[-1] > 0 else 1
    return 100 * np.arange(1, len(s) + 1) / max(len(s), 1), 100 * c / total

fig, ax = plt.subplots(figsize=(7, 5.5))
for _name in OMIC_NAMES:
    _rank_pct, _cum_pct = _cumulative_pct(omic_feature_dfs[_name]['Attention'].values)
    ax.plot(_rank_pct, _cum_pct, color=OMIC_COLORS[_name], linewidth=2,
            label=f'{_name} (n={len(omic_feature_dfs[_name])})')
ax.plot([0, 100], [0, 100], color='#cccccc', linestyle=':', linewidth=1, label='Uniform (no concentration)')
ax.set_xlabel('Features Ranked by Attention (% of that omic layer)')
ax.set_ylabel('Cumulative Attention (%)')
ax.set_title('Attention Concentration by Omic')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(supplementary_dir + "attention_concentration_by_omic.pdf")
plt.close(fig)

# Kruskal-Wallis across every omic when there are more than 2 (Mann-Whitney U is only
# defined for exactly 2 groups - the original script's fixed comparison).
_omic_attention_groups = [omic_feature_dfs[_name]['Attention'].values for _name in OMIC_NAMES]
if len(OMIC_NAMES) == 2:
    _, p_omic = mannwhitneyu(*_omic_attention_groups, alternative='two-sided')
    _stat_label = f'Mann-Whitney U: p={p_omic:.3g}'
else:
    _kw_stat, p_omic = kruskal(*_omic_attention_groups)
    _stat_label = f'Kruskal-Wallis: p={p_omic:.3g}'
fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(OMIC_NAMES)), 5.5))
parts = ax.violinplot(_omic_attention_groups, showmedians=True)
for pc, _name in zip(parts['bodies'], OMIC_NAMES):
    pc.set_facecolor(OMIC_COLORS[_name])
    pc.set_alpha(0.7)
ax.set_xticks(range(1, len(OMIC_NAMES) + 1))
ax.set_xticklabels(OMIC_NAMES)
ax.set_ylabel('Attention Score')
ax.set_title(f'Attention Score Distribution by Omic Type\n{_stat_label}')
plt.tight_layout()
plt.savefig(supplementary_dir + "attention_distribution_by_omic.pdf")
plt.close(fig)
print("Supplementary: omic-wise attention concentration/distribution saved.")

# ---------------------------------------------------------------------------
# 4. Leiden program network -> (a) ranked size bars, (b) inter-program connectivity heatmap
# ---------------------------------------------------------------------------
if not valid_clusters:
    print("Supplementary: skipping program-network figures (no valid Leiden clusters).")
else:
    size_summary = sorted(
        [(cid, G_cluster.nodes[cid]['size'], G_cluster.nodes[cid]['dominant_type']) for cid in valid_clusters],
        key=lambda t: t[1]
    )
    prog_labels = [f'Program {cid}' for cid, _, _ in size_summary]
    prog_sizes = [s for _, s, _ in size_summary]
    prog_colors = [OMIC_COLORS[t] for _, _, t in size_summary]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(size_summary))))
    ax.barh(prog_labels, prog_sizes, color=prog_colors, edgecolor='black', linewidth=0.4)
    ax.set_xlabel('Number of Features')
    ax.set_title('Multi-Omic Program Sizes')
    ax.legend(handles=[Patch(facecolor=OMIC_COLORS[_name], label=f'{_name}-dominant') for _name in OMIC_NAMES],
              loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(supplementary_dir + "program_size_ranked.pdf")
    plt.close(fig)

    cids = sorted(valid_clusters)
    conn = pd.DataFrame(0.0, index=[f'P{c}' for c in cids], columns=[f'P{c}' for c in cids])
    for u, v, edge_data_c in G_cluster.edges(data=True):
        conn.loc[f'P{u}', f'P{v}'] = edge_data_c['weight']
        conn.loc[f'P{v}', f'P{u}'] = edge_data_c['weight']
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(cids) + 2), max(4, 0.6 * len(cids) + 2)))
    sns.heatmap(conn, cmap=SEQ_CMAP, square=True, cbar_kws={'label': 'Attention weight'}, ax=ax)
    ax.set_title('Inter-Program Connectivity')
    plt.tight_layout()
    plt.savefig(supplementary_dir + "program_connectivity_heatmap.pdf")
    plt.close(fig)
    print("Supplementary: program size / connectivity figures saved.")

# ---------------------------------------------------------------------------
# 5. Per-program subgraphs -> (a) unsigned importance ranking, (b) member correlation heatmap
# ---------------------------------------------------------------------------
if not valid_clusters:
    print("Supplementary: skipping per-program feature figures (no valid Leiden clusters).")
else:
    # (every omic's features stacked) x samples, same row index as G_leiden node ids
    # (see the Leiden clustering section above) - recomputed locally since `samples`
    # was repurposed as a metadata frame in the Network Visualization section above.
    feature_matrix = np.concatenate([omic_matrices[_name] for _name in OMIC_NAMES], axis=0)

    importance_pdf_path = supplementary_dir + "program_feature_importance_ranked.pdf"
    with PdfPages(importance_pdf_path) as pdf:
        for cid in sorted(valid_clusters):
            member_idx = [n for n, c in node_to_cluster.items() if c == cid]
            names = [G_leiden.nodes[n]['name'] for n in member_idx]
            atts = np.array([G_leiden.nodes[n]['attention'] for n in member_idx])
            order = np.argsort(atts)[::-1]
            names_r, atts_r = [names[i] for i in order], atts[order]
            fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(names_r))))
            norm_atts = atts_r / atts_r.max() if atts_r.max() > 0 else atts_r
            colors_bar = plt.cm.get_cmap(SEQ_CMAP)(norm_atts)
            y_pos = np.arange(len(names_r))[::-1]
            ax.barh(y_pos, atts_r, color=colors_bar, edgecolor='black', linewidth=0.3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names_r, fontsize=7)
            ax.set_xlabel('Attention Score')
            ax.set_title(f'Program {cid}: Feature Importance (unsigned)')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"Supplementary: per-program feature importance saved to {importance_pdf_path}")

    corr_pdf_path = supplementary_dir + "program_feature_correlation_heatmap.pdf"
    with PdfPages(corr_pdf_path) as pdf:
        for cid in sorted(valid_clusters):
            member_idx = [n for n, c in node_to_cluster.items() if c == cid]
            if len(member_idx) < 2:
                continue
            names = [G_leiden.nodes[n]['name'] for n in member_idx]
            corr = np.corrcoef(feature_matrix[member_idx, :])
            fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(names) + 2), max(4, 0.35 * len(names) + 2)))
            sns.heatmap(corr, cmap=DIV_CMAP, vmin=-1, vmax=1, center=0,
                        xticklabels=names, yticklabels=names,
                        cbar_kws={'label': 'Pearson r'}, ax=ax, square=True)
            ax.tick_params(axis='both', labelsize=6)
            ax.set_title(f'Program {cid}: Feature-Feature Correlation (across samples)')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"Supplementary: per-program feature correlation heatmaps saved to {corr_pdf_path}")

# ---------------------------------------------------------------------------
# 6. Program activity -> (a) sample x program heatmap, (b) program-program correlation
# ---------------------------------------------------------------------------
if not valid_clusters or program_activity.shape[1] == 0:
    print("Supplementary: skipping program-activity heatmaps (no valid Leiden clusters).")
else:
    activity_std = program_activity.std(ddof=0).replace(0, 1)
    activity_z = (program_activity - program_activity.mean()) / activity_std
    activity_z = activity_z.replace([np.inf, -np.inf], np.nan).fillna(0)
    row_colors = metadata.set_index('ID').loc[program_activity.index, 'CONDITION'].astype(str).map(CONDITION_COLORS)
    row_colors.name = 'Condition'
    g_act = sns.clustermap(activity_z, cmap=DIV_CMAP, center=0, row_colors=row_colors,
                            figsize=(max(8, 0.5 * activity_z.shape[1] + 3), 9),
                            cbar_kws={'label': 'Activity (z-score)'}, xticklabels=True, yticklabels=False)
    g_act.ax_heatmap.set_xlabel('Program')
    g_act.ax_heatmap.set_ylabel('Sample')
    g_act.fig.suptitle('Program Activity Across Samples', y=1.02)
    g_act.savefig(supplementary_dir + "program_activity_heatmap.pdf", bbox_inches='tight')
    plt.close(g_act.fig)

    if program_activity.shape[1] > 1:
        prog_corr = program_activity.corr().fillna(0)
        g_corr = sns.clustermap(prog_corr, cmap=DIV_CMAP, vmin=-1, vmax=1, center=0,
                                 figsize=(max(6, 0.5 * prog_corr.shape[0] + 3), max(6, 0.5 * prog_corr.shape[0] + 3)),
                                 cbar_kws={'label': 'Pearson r'})
        g_corr.fig.suptitle('Program-Program Activity Correlation Across Samples', y=1.02)
        g_corr.savefig(supplementary_dir + "program_activity_correlation.pdf", bbox_inches='tight')
        plt.close(g_corr.fig)
    print("Supplementary: program activity heatmaps saved.")

# ---------------------------------------------------------------------------
# 7. Embeddings -> (a) silhouette score by representation depth, (b) PCA scree
# ---------------------------------------------------------------------------
condition_array_for_silhouette = labels.astype(str).values
depth_names = ['Cell Nodes', 'g128 Embedding', 'g64 Embedding', 'g32 Embedding']
sil_scores = []
for depth_name in depth_names:
    try:
        sil_scores.append(silhouette_score(embeddings[depth_name], condition_array_for_silhouette))
    except ValueError:
        sil_scores.append(np.nan)
fig, ax = plt.subplots(figsize=(7, 5))
bar_colors = plt.cm.get_cmap(SEQ_CMAP)(np.linspace(0.85, 0.25, len(depth_names)))
ax.bar(depth_names, sil_scores, color=bar_colors, edgecolor='black', linewidth=0.4)
ax.axhline(0, color='#999999', linewidth=0.8)
ax.set_ylabel('Silhouette Score (by Condition)')
ax.set_title('Condition Separability Across Representation Depths')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(supplementary_dir + "embedding_separability_silhouette.pdf")
plt.close(fig)

n_pca = max(1, min(10, embeddings['Cell Nodes'].shape[1], embeddings['Cell Nodes'].shape[0] - 1))
pca_full = PCA(n_components=n_pca)
pca_full.fit(embeddings['Cell Nodes'])
var_ratio = pca_full.explained_variance_ratio_ * 100
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(np.arange(1, len(var_ratio) + 1), var_ratio, color=OKABE_ITO[0], alpha=0.8, label='Individual')
ax2 = ax.twinx()
ax2.plot(np.arange(1, len(var_ratio) + 1), np.cumsum(var_ratio), color=OKABE_ITO[1],
         marker='o', markersize=4, label='Cumulative')
ax2.spines['right'].set_visible(True)
ax2.set_ylabel('Cumulative Variance Explained (%)')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('PCA Scree Plot – Cell Node Embedding')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
plt.tight_layout()
plt.savefig(supplementary_dir + "embedding_pca_scree.pdf")
plt.close(fig)
print("Supplementary: embedding separability / PCA scree figures saved.")

# ---------------------------------------------------------------------------
# 8. Network visualization -> hub-node summary
# ---------------------------------------------------------------------------
degree_series = pd.Series(dict(G.degree()))
top_hubs = degree_series.sort_values(ascending=False).head(20)
hub_types = [G.nodes[n]['node_type'] for n in top_hubs.index]
hub_names = [G.nodes[n]['label'] for n in top_hubs.index]
hub_colors = [OMIC_COLORS[type_idx_to_name[t]] if t in type_idx_to_name else '#888888' for t in hub_types]
fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(top_hubs))))
y_pos = np.arange(len(top_hubs))[::-1]
ax.barh(y_pos, top_hubs.values, color=hub_colors, edgecolor='black', linewidth=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels(hub_names, fontsize=7)
ax.set_xlabel('Degree (number of connections)')
ax.set_title('Top 20 Hub Nodes in the Attention Network')
ax.legend(handles=[Patch(facecolor=OMIC_COLORS[_name], label=_name) for _name in OMIC_NAMES] +
                   [Patch(facecolor='#888888', label='Sample')], loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(supplementary_dir + "network_hub_nodes.pdf")
plt.close(fig)
print("Supplementary: hub-node figure saved.")

print("=== Supplementary visualizations complete ===")


