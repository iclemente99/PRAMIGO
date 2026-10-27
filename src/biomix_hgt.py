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
from pyHGT.model import GNN, GNN_from_raw
from omics_manifest import resolve_manifest, ManifestError
from pramigo_figures import square_frame  # square axes, black frame, no grid
import itertools

from warnings import filterwarnings
filterwarnings("ignore")

import plotly.graph_objects as go
import seaborn as sns
import matplotlib
import matplotlib.ticker
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

# Omics and conditions get disjoint colour sets, so a network can show both at once
# (as in the PTB/CLL network figures: green transcripts, purple metabolites, black CpGs;
# samples as condition-coloured rings).
OMIC_PALETTE = ["#009E73", "#6a3d9a", "#000000", "#8c510a", "#999933", "#882255"]
CONDITION_PALETTE = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#7f7f7f"]

SEQ_CMAP = "viridis"         # every continuous/sequential scale (attention, activity, distance)
DIV_CMAP = "RdBu_r"          # every diverging/signed scale (blue=low/negative, red=high/positive)

# Matplotlib marker cycle for an arbitrary number of omics (PRAMIGO's original script
# hardcoded exactly 2: circle for genes, diamond for metabolites). Cycles if there are
# more omics than markers - readability of a many-omic figure is a separate, real
# concern (see design notes) that a bigger marker alphabet alone doesn't solve.
OMIC_MARKERS = ['^', 'D', 's', 'v', 'P', 'X', '*', 'p']  # 'o' is reserved for samples
# Plotly symbol cycle for the interactive network plot (network_interactive.html).
# 'circle' is reserved for sample nodes so it's deliberately excluded here.
OMIC_PLOTLY_SHAPES = ['triangle-up', 'square', 'diamond', 'star', 'pentagon', 'hexagon2', 'cross', 'x']


def get_omic_palette(omic_names):
    """Deterministic omic-name -> color mapping, generalizing the original script's
    fixed OMIC1_COLOR/OMIC2_COLOR constants to an arbitrary number of omics. Cycles
    through OKABE_ITO minus black (reserved for text/outlines) - the same palette
    CONDITION_COLORS draws from, which is fine since omic identity and condition are
    never colored in the same legend at once."""
    return {name: OMIC_PALETTE[i % len(OMIC_PALETTE)] for i, name in enumerate(omic_names)}


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
    return {cond: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, cond in enumerate(uniq)}


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
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of training subgraphs per epoch (each = all features + a random third of '
                         'the training samples)')
parser.add_argument('--val_ratio', type=float, default=0.15,
                    help='Fraction of samples (per condition) held out for validation')
parser.add_argument('--test_ratio', type=float, default=0.15,
                    help='Fraction of samples (per condition) held out for the final test metrics')

parser.add_argument('--omic1_rate', type=float, default=1,
                    help='[legacy 2-omic mode] Ignored (kept for compatibility): training subgraphs always keep every feature so they match the inference graph.')
parser.add_argument('--omic2_rate', type=float, default=1,
                    help='[legacy 2-omic mode] Ignored (kept for compatibility): training subgraphs always keep every feature so they match the inference graph.')

# Result
parser.add_argument('--data_name', type=str,
                    help='The name for dataset')
parser.add_argument('--result_dir', type=str, required=True,
                    help='The address for storing the models and optimization results.')
parser.add_argument('--reduction', type=str, default='AE', choices=['AE'],
                    help='Feature extraction method. Only AE is supported: with raw input, feature nodes '
                         '(dim = n_samples) and sample nodes (dim = n_features) cannot share one node-feature '
                         'tensor, and the old raw-mode loss relied on decoder outputs that no longer exist.')
parser.add_argument('--in_dim', type=int, default=256,
                    help='Number of hidden dimension (AE)')
parser.add_argument('--omic1_corr_cutoff', type=float, default=0.7,
                    help='[legacy 2-omic mode] Correlation cutoff threshold (0-1) for omic1 feature-feature '
                         'graph edges. Use `corr_cutoff:` in --omics_manifest instead when using the manifest.')
parser.add_argument('--omic2_corr_cutoff', type=float, default=0.7,
                    help='[legacy 2-omic mode] Correlation cutoff threshold (0-1) for omic2 feature-feature '
                         'graph edges. Use `corr_cutoff:` in --omics_manifest instead when using the manifest.')
parser.add_argument('--network_features_per_omic', type=int, nargs='+', default=[100],
                    help='Top features per omic (by attribution) drawn in plots/network_multiomic.pdf; one page '
                         'per value, e.g. --network_features_per_omic 10 20 50 100')
parser.add_argument('--leiden_resolution', type=float, default=1.5,
                    help='Resolution parameter for Leiden clustering of the top-feature attention subgraph')
parser.add_argument('--target_class', type=str, nargs='+', default=None,
                    help="Condition(s) whose programs are built (Leiden, program activity, report). "
                         "One or more CONDITION values (case-insensitive), or 'all'. Default: 'disease'; "
                         "if no condition is called 'disease', every condition gets its own programs.")
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
    common_samples = sorted(meta_samples.intersection(*omic_sample_sets.values()))  # sorted: set order depends on the per-process string hash seed
    sample_presence = {name: {s: True for s in common_samples} for name in OMIC_NAMES}
else:
    # Partial-coverage: keep any sample present in the metadata AND at least one
    # omic. A sample missing an omic's data gets that omic's feature block
    # zero-filled below purely so every omic matrix keeps one column per
    # common_sample - but critically, the KNN mask that turns into graph edges for
    # that omic<->sample relation is explicitly zeroed out for that sample further
    # down, so the *graph structure* carries the missingness (no edges of that
    # relation type at all), not an imputed value pretending to be real data.
    common_samples = sorted(meta_samples.intersection(set().union(*omic_sample_sets.values())))  # sorted: deterministic sample order across runs
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

# Target condition(s) for the program analysis (--target_class), resolved
# case-insensitively against the CONDITION values. Fails early on an explicit typo.
_class_lookup = {str(c).lower(): i for i, c in enumerate(label_encoder.classes_)}
if args.target_class is None or [t.lower() for t in args.target_class] == ['disease']:
    _requested = ['disease']
    _explicit = args.target_class is not None
else:
    _requested, _explicit = args.target_class, True
if [t.lower() for t in _requested] == ['all']:
    TARGET_CLASS_IDX = list(range(len(label_encoder.classes_)))
elif all(t.lower() in _class_lookup for t in _requested):
    TARGET_CLASS_IDX = list(dict.fromkeys(_class_lookup[t.lower()] for t in _requested))
elif not _explicit:
    print(f"NOTE: no condition is called 'disease' (conditions: {list(label_encoder.classes_)}); "
          f"building programs for every condition. Use --target_class to choose.")
    TARGET_CLASS_IDX = list(range(len(label_encoder.classes_)))
else:
    raise SystemExit(f"--target_class {args.target_class}: unknown condition(s). "
                     f"Available: {list(label_encoder.classes_)} (or 'all').")
print(f"Program analysis target condition(s): {[str(label_encoder.classes_[c]) for c in TARGET_CLASS_IDX]}")

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
    # Exact Pearson correlation between the rows of A and B: re-standardize each row
    # with the population std (ddof=0) so that dot / dim is exactly r. (The inputs were
    # z-scored with torch's unbiased std, which made the old dot / ae_n_hid equal to
    # r * (n-1)/n, i.e. a slightly stricter cutoff than the one requested.)
    def _standardize(X):
        X = np.asarray(X, dtype=np.float64)
        return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    corr = np.dot(_standardize(A), _standardize(B).T) / A.shape[1]
    # |r| >= threshold: strongly anti-correlated features are linked too, not only
    # positively correlated ones. The edge is unsigned; only its presence is encoded.
    return (np.abs(corr) >= threshold).astype(np.uint8)


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
# HELD-OUT SAMPLE SPLIT (train / validation / test)
#
################################################################
# The split is made over SAMPLES, stratified by CONDITION, before any training job is
# drawn. Training subgraphs only ever contain training samples; validation and test
# samples are never seen by the loss or the optimizer. Their labels are used only to
# score the model.
#
# The unsupervised preprocessing above (the autoencoders and the KNN/correlation masks)
# is still fitted on every sample's omic values, without labels. This is the
# transductive setting usual for graph models. Held-out samples also send no messages
# (there are no sample->feature or sample->sample edges), so they cannot influence the
# representation of any other node.

def stratified_sample_split(int_labels, val_ratio, test_ratio, rng):
    """Per class: shuffle, then take round(ratio * n_c) samples (at least 1 when the
    ratio is > 0) for test and for validation; the rest go to training. A class with
    fewer than 3 samples cannot be represented in all three sets, so it goes to
    training entirely (with a warning)."""
    train, val, test = [], [], []
    for c in np.unique(int_labels):
        idx = np.where(int_labels == c)[0]
        rng.shuffle(idx)
        if len(idx) < 3 and (val_ratio > 0 or test_ratio > 0):
            print(f"WARNING: class {label_encoder.classes_[c]!r} has only {len(idx)} sample(s); "
                  f"all of them go to training and the class is absent from validation/test.")
            train.extend(idx)
            continue
        n_test = max(1, int(round(test_ratio * len(idx)))) if test_ratio > 0 else 0
        n_val = max(1, int(round(val_ratio * len(idx)))) if val_ratio > 0 else 0
        if len(idx) - n_test - n_val < 1:
            raise SystemExit(f"Class {label_encoder.classes_[c]!r} ({len(idx)} samples) leaves no training "
                             f"samples with --val_ratio {val_ratio} / --test_ratio {test_ratio}.")
        test.extend(idx[:n_test])
        val.extend(idx[n_test:n_test + n_val])
        train.extend(idx[n_test + n_val:])
    return np.sort(train), np.sort(val), np.sort(test)

train_idx, val_idx, test_idx = stratified_sample_split(
    integer_labels, args.val_ratio, args.test_ratio, np.random.RandomState(seed))
sample_split = np.full(len(common_samples), 'train', dtype=object)
sample_split[val_idx] = 'val'
sample_split[test_idx] = 'test'
pd.DataFrame({'ID': common_samples, 'CONDITION': metadata['CONDITION'].values, 'split': sample_split}) \
    .to_csv(model_dir + 'sample_split.csv', index=False)
print(f"Sample split (stratified by CONDITION): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
      f"-> {model_dir}sample_split.csv")
if len(test_idx) == 0:
    print("WARNING: no held-out test samples (--test_ratio 0 or classes too small). The reported "
          "metrics will be computed on TRAINING samples and are not an estimate of generalization.")


################################################################
#
# ONE GRAPH BUILDER FOR TRAINING, VALIDATION AND INFERENCE
#
################################################################
# Every graph the model ever sees is "the whole feature graph + a subset of sample
# nodes": training jobs (random training samples), the validation graph (validation
# samples) and the inference batches (all samples). Every feature node and every
# feature->feature edge is always present. Sample nodes only receive messages. So
# each node's neighbourhood, and therefore each softmax over incoming edges, is the
# same during training and inference. (Previously each training job kept only about
# 1/n_batch of the features, the top ones by degree, while inference used them all.)
edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())} # relation name -> relation id
edge_dict['self'] = len(edge_dict) # reserved 'self' relation id

OMIC_FEATURE_OFFSETS = {}
_off = 0
for _name in OMIC_NAMES:
    OMIC_FEATURE_OFFSETS[_name] = _off
    _off += encoded[_name].shape[0]
n_feature_nodes = _off  # all omic features come before the samples in the global node index

# Feature-only part (feature nodes + feature->feature edges): built once, shared by every graph.
_feat_x, _feat_node_type, _, _feat_edge_index, _feat_edge_type = build_data_generic(
    {**{name: encoded[name].detach().numpy() for name in OMIC_NAMES},
     'sample': np.zeros((0, encoded2_samples.shape[1]), dtype=np.float32)},
    {name: cross_sample_masks[name][:, :0] for name in OMIC_NAMES},
    self_masks, sample_mask[:0, :0], cross_omic_masks, OMIC_NAMES, edge_dict)
FEATURE_X = torch.cat([_feat_x[name] for name in OMIC_NAMES], 0)
FEATURE_NODE_TYPE = _feat_node_type
FEATURE_EDGE_INDEX = _feat_edge_index
FEATURE_EDGE_TYPE = _feat_edge_type
del _feat_x, _feat_node_type, _feat_edge_index, _feat_edge_type
SAMPLE_X = encoded2_samples.detach().float().cpu()
_sample_rel = relation_name('sample', 'sample')

def build_sample_subgraph(sample_idx):
    """All feature nodes and feature->feature edges plus the samples in `sample_idx`.
    Node order: features (global index), then the given samples in the given order
    (local sample k = global sample sample_idx[k]). Returns CPU tensors
    (x, node_type, edge_time, edge_index, edge_type)."""
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    n_s = len(sample_idx)
    ei_parts = [FEATURE_EDGE_INDEX.numpy()]
    et_parts = [FEATURE_EDGE_TYPE.numpy()]
    for name in OMIC_NAMES:  # omic feature -> sample edges
        o_idx, s_idx = np.nonzero(cross_sample_masks[name][:, sample_idx])
        ei_parts.append(np.stack([o_idx + OMIC_FEATURE_OFFSETS[name], s_idx + n_feature_nodes]))
        et_parts.append(np.full(o_idx.shape[0], edge_dict[relation_name(name, 'sample')]))
    i_s, j_s = np.nonzero(sample_mask[np.ix_(sample_idx, sample_idx)])  # sample -> sample edges
    ei_parts.append(np.stack([i_s + n_feature_nodes, j_s + n_feature_nodes]))
    et_parts.append(np.full(i_s.shape[0], edge_dict[_sample_rel]))
    edge_index = torch.LongTensor(np.concatenate(ei_parts, axis=1))
    edge_type = torch.LongTensor(np.concatenate(et_parts))
    edge_time = torch.zeros(edge_type.shape[0], dtype=torch.long)
    x = torch.cat([FEATURE_X, SAMPLE_X[torch.as_tensor(sample_idx)]], 0)
    node_type = torch.cat([FEATURE_NODE_TYPE, torch.full((n_s,), len(OMIC_NAMES), dtype=torch.long)])
    return x, node_type, edge_time, edge_index, edge_type


################################################################
#
# TRAINING SUBGRAPHS (DeepMAPS-style, training samples only)
#
################################################################
# --n_batch subgraphs, each holding a random third of the TRAINING samples (at least 2,
# so the cosine loss has a pair) plus the whole feature graph. Drawn once, reused every
# epoch, from a dedicated seeded RNG (so --resume redraws exactly the same jobs).
if any(OMIC_RATE[name] != 1.0 for name in OMIC_NAMES):
    print(f"NOTE: per-omic `rate` ({OMIC_RATE}) is ignored. Training subgraphs always keep every "
          f"feature, so the graph seen in training matches the one used at inference.")
sample_num = int(min(len(train_idx), max(2, round(len(train_idx) / 3))))
_job_rng = np.random.RandomState(seed + 1)
train_jobs = [np.sort(_job_rng.choice(train_idx, sample_num, replace=False)) for _ in range(args.n_batch)]
_covered = len(np.unique(np.concatenate(train_jobs)))
print(f"Training subgraphs: {len(train_jobs)} x {sample_num} training samples (+ all {n_feature_nodes} "
      f"feature nodes); {_covered}/{len(train_idx)} training samples covered.")


################################################################
#   
# MODEL
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
cross_entropy_losses = []
cosine_losses = []
total_losses = []
val_losses = []
lr_history = []

cosine_loss_fn = nn.CosineEmbeddingLoss(margin=0.5, reduction='sum').to(device)  # Initialize CosineEmbeddingLoss

# Best-epoch selection: the model used for inference/interpretability is the one from
# the epoch with the lowest validation loss (held-out validation samples). With no
# validation samples, the final epoch is used.
best_val_loss = float('inf')
best_epoch = None          # 0-based epoch index of best_model_state
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
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch')
        best_model_state = checkpoint.get('best_model_state_dict')
        cosine_losses = checkpoint.get('cosine_losses', [])
        cross_entropy_losses = checkpoint.get('cross_entropy_losses', [])
        lr_history = checkpoint.get('lr_history', [])
        _ckpt_split = checkpoint.get('sample_split')
        _this_split = {'train': [common_samples[k] for k in train_idx],
                       'val': [common_samples[k] for k in val_idx],
                       'test': [common_samples[k] for k in test_idx]}
        if _ckpt_split is None or _ckpt_split != _this_split:
            raise SystemExit("--resume: the checkpoint was trained on a different (or unrecorded) "
                             "train/val/test sample split - resuming would leak held-out samples into "
                             "training. Start a fresh run instead.")
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
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'best_model_state_dict': best_model_state,
        'cosine_losses': cosine_losses,
        'cross_entropy_losses': cross_entropy_losses,
        'lr_history': lr_history,
        'sample_split': {'train': [common_samples[k] for k in train_idx],
                         'val': [common_samples[k] for k in val_idx],
                         'test': [common_samples[k] for k in test_idx]},
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, checkpoint_path)


def compute_loss(class_rep, sample_matrix, samples_labels):
    """Training objective, shared by training and validation:
    0.3 * cross-entropy + 0.7 * cosine-embedding loss over sample pairs
    (same class -> similar embeddings, different class -> dissimilar).
    Returns (loss, ce_loss, cosine_loss)."""
    ce_loss = F.cross_entropy(class_rep, torch.FloatTensor(samples_labels).to(device))
    num_samples = sample_matrix.shape[0]
    max_pairs = 1000  # Limit pairs to avoid O(n^2) computation
    sample_idx_i = []
    sample_idx_j = []
    targets = []
    sample_classes = torch.argmax(torch.LongTensor(samples_labels), dim=1)
    # Positive pairs (same class)
    for cls in torch.unique(sample_classes):
        cls_idx = torch.where(sample_classes == cls)[0]
        if len(cls_idx) >= 2:  # Need at least 2 samples for pairs
            pairs = torch.combinations(cls_idx, r=2)
            sample_idx_i.extend(pairs[:, 0].tolist())
            sample_idx_j.extend(pairs[:, 1].tolist())
            targets.extend([1] * len(pairs))
    # Negative pairs (different classes)
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
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
        cosine_loss = cosine_loss_fn(sample_matrix[sample_idx_i], sample_matrix[sample_idx_j], target_tensor)
    else:
        cosine_loss = torch.tensor(0.0, device=device)
    loss = 0.3 * ce_loss + 0.7 * cosine_loss
    return loss, ce_loss, cosine_loss


def forward_on_samples(sample_idx):
    """Runs the HGT on build_sample_subgraph(sample_idx). Returns
    (node_rep, class_rep, sample_matrix) - class_rep/sample_matrix rows follow sample_idx."""
    x, node_type, edge_time, edge_index, edge_type = build_sample_subgraph(sample_idx)
    node_type = node_type.to(device)
    node_rep, class_rep, g_32, g_64, g_128 = gnn.forward(
        x.to(device), node_type, edge_time.to(device), edge_index.to(device), edge_type.to(device))
    return node_rep, class_rep, node_rep[node_type == len(OMIC_NAMES), ]


for epoch in np.arange(start_epoch, args.epoch): # Iterates over the epochs
    # Training: one optimizer step per training subgraph. The logged values are the
    # MEANS over this epoch's subgraphs (previously: the last batch's total loss, and
    # CE/cosine values that the validation pass had overwritten).
    gnn.train()
    L, L_ce, L_cos = 0.0, 0.0, 0.0
    for job_idx in train_jobs:
        node_rep, class_rep, sample_matrix = forward_on_samples(job_idx)
        loss, ce_loss, cosine_loss = compute_loss(class_rep, sample_matrix, one_hot_labels[job_idx, ])
        L += loss.item()
        L_ce += ce_loss.item()
        L_cos += cosine_loss.item()
        optimizer.zero_grad() # Initializies optimizer
        loss.backward() # Sets backpropagation to loss function
        optimizer.step() # Sets a step in optimizer
    printed_train_loss = L / len(train_jobs)
    scheduler.step(printed_train_loss)
    lr_history.append(optimizer.param_groups[0]['lr'])

    # Validation: one forward pass over the held-out validation samples (same graph
    # construction as training/inference), no gradient.
    gnn.eval()
    if len(val_idx) > 0:
        with torch.no_grad():
            node_rep, class_rep, sample_matrix = forward_on_samples(val_idx)
            val_loss, _, _ = compute_loss(class_rep, sample_matrix, one_hot_labels[val_idx, ])
        printed_val_loss = val_loss.item()
    else:
        printed_val_loss = float('nan')
    print(f'Epoch: {epoch+1} | Train Loss: {printed_train_loss:.12f}\nEpoch: {epoch+1} | Val Loss: {printed_val_loss:.12f}')
    total_losses.append(printed_train_loss)
    cross_entropy_losses.append(L_ce / len(train_jobs))
    cosine_losses.append(L_cos / len(train_jobs))
    val_losses.append(printed_val_loss)

    if np.isfinite(printed_val_loss) and printed_val_loss < best_val_loss:
        best_val_loss = printed_val_loss
        best_epoch = int(epoch)
        best_model_state = {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}
        print(f"Epoch: {epoch+1} | new best validation loss {best_val_loss:.12f}")

    if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epoch:
        save_checkpoint(epoch)
        print(f"Checkpoint saved at epoch {epoch + 1} -> {checkpoint_path}")
end = timer()
print(f"BiomiX-HGT took {round((end - start)/60,2)} mins to train")
_all_edge_type = build_sample_subgraph(np.arange(encoded2_samples.shape[0]))[4]  # edge types of the full graph
unique_vals, counts = torch.unique(_all_edge_type, return_counts=True) # Count value repetitions
count_dict = dict(zip(unique_vals.tolist(), counts.tolist())) # Combine into a dictionary (optional)
print(f"Edge dict is:\n{edge_dict}\nAnd count dict is:\n{count_dict}")

#import matplotlib.pyplot as plt

# Total/Validation loss carry the headline message (converging? overfitting?) so they
# get bold, saturated lines; the two loss components are supporting detail, so they're
# thin and muted to avoid competing for attention.
fig, ax = plt.subplots(figsize=(6.5, 7.2))
ax.plot(total_losses, label='Training Loss (epoch mean)', color=OKABE_ITO[0], linewidth=2.5, zorder=3)
ax.plot(val_losses, label='Validation Loss (held-out samples)', color=OKABE_ITO[1], linewidth=2.5, zorder=3)
ax.plot(cosine_losses, label='Cosine Loss (component)', color='#999999', linewidth=1.1, linestyle='--', zorder=2)
ax.plot(cross_entropy_losses, label='Cross Entropy (component)', color='#bbbbbb', linewidth=1.1, linestyle=':', zorder=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
if best_epoch is not None:
    ax.axvline(best_epoch, color=OKABE_ITO[4], linestyle='--', linewidth=1.2,
               label=f'Selected model (epoch {best_epoch + 1}, lowest val loss)')
ax.set_title('Training Convergence')
square_frame(ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(plots_dir+"training_loss_components.pdf")
plt.close(fig)


if best_model_state is not None:
    selected_state, selected_epoch = best_model_state, best_epoch
    print(f"Model selection: using epoch {best_epoch + 1}/{len(val_losses)} "
          f"(lowest validation loss = {best_val_loss:.12f}).")
else:
    selected_state, selected_epoch = gnn.state_dict(), len(total_losses) - 1
    print("Model selection: no validation samples - using the final epoch.")
state = {'model': selected_state, 'optimizer': scheduler.state_dict(),
        'epoch': selected_epoch, 'best_val_loss': best_val_loss} # Saves a dictionary of model
#model0=f'BiomiX_epoch_{args.epoch}_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
model0 = f'BiomiX_n_hid_{args.n_hid}_nheads_{args.n_heads}_lr_01_n_batch{args.n_batch}'
torch.save(state, model_dir+model0) # Saves model in folder
pd.DataFrame(total_losses).to_csv(loss_dir+model0+"_total_losses.csv")
pd.DataFrame(cosine_losses).to_csv(loss_dir+model0+"_cosine_losses.csv")
pd.DataFrame(cross_entropy_losses).to_csv(loss_dir+model0+"_cross_entropy_losses.csv")
pd.DataFrame(val_losses).to_csv(loss_dir+model0+"_val_losses.csv")
debuginfoStr('Graph Autoencoder training finished')




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
# The trained model is applied to the FULL graph: all features + all samples. Up to
# 10000 samples this is a single pass. Above that, samples are processed in batches of
# 500 (memory only). This is mathematically identical to the single pass because samples
# never send messages, and the merge below plus the guard in the loop keep it that way.
if sample_len > 10000:
    ba = 500
else:
    ba = sample_len

# ----------------------------------------------------------------------------------
# Interpretability quantities extracted here (see README "Interpretability"):
#
#   alpha[e, l, h]  post-softmax attention of edge e = (j -> i) in HGT layer l, head h.
#                   This is the weight the model actually uses (the old output was the
#                   raw pre-softmax logit of layer 0 only, which is only defined up to a
#                   per-target shift, so its sign/absolute value carried no meaning).
#
#   push_c(e, s)    sum_h alpha[e,l,h] * d z~_c(s) / d alpha[e,l,h]: signed attention x
#                   gradient (in the spirit of Chefer et al., 2021) - how much edge e's
#                   message moves sample s towards class c. z~_c(s) = z_c(s) - mean_k z_k(s)
#                   is the centered class logit (the part of the logits the softmax
#                   prediction depends on; unlike log p_c it does not saturate).
#
#   attr[e, l, c]     = mean of push_c over TRAINING samples OF class c   (on-class:
#                       does the edge push class-c samples towards c?)
#   attr_off[e, l, c] = mean of push_c over TRAINING samples NOT of class c (off-class:
#                       does it also push the other samples towards c?)
#   score[e, c]       = attr - max(attr_off, 0), both summed over layers: class-specific
#                       evidence for c. A feature that pushes EVERY sample towards c (a
#                       bias, not a discriminator) is penalized; one that pushes c-samples
#                       towards c and leaves (or pushes away) the others keeps its score.
#                       Binary case: attr_off_D = -attr_H, so score_D = attr_D + min(attr_H, 0).
#   Only TRAINING samples are used, so programs/weights derived from these scores can be
#   tested on the held-out test samples without double-dipping.
#
#   Node values = sums over the node's OUTGOING edges (the messages it sends) of attr and
#   attr_off; the node score is then attr - max(attr_off, 0) on those sums.
# ----------------------------------------------------------------------------------
n_classes = one_hot_labels.shape[1]
class_names = [str(c) for c in label_encoder.classes_]
n_att_layers = len(gnn.gcs)
n_total_nodes = n_feature_nodes + sample_len
is_train_sample = np.zeros(sample_len, dtype=bool)
is_train_sample[train_idx] = True
class_counts = np.bincount(integer_labels[train_idx], minlength=n_classes)   # on-class denominators
off_counts = len(train_idx) - class_counts                                  # off-class denominators

sample_embedding = []
omic_matrix = {}
g32_embeddings = []
g64_embeddings = []
g128_embeddings = []
class_embeddings = []
# Per-batch edge records, stored with GLOBAL node indices so batches can be merged.
edge_src_parts, edge_tgt_parts, edge_type_parts = [], [], []
edge_alpha_parts = []  # [E_b, n_layers, n_heads]
edge_attr_parts = []   # [E_b, n_layers, n_classes] on-class push (sum over the batch's training samples)
edge_off_parts = []    # [E_b, n_layers, n_classes] off-class push (sum over the batch's training samples)

for i in range(0, sample_len, ba):
    # Same graph builder as the training subgraphs and the validation graph: the whole
    # feature graph plus this batch of samples (global samples i .. i+ba-1).
    x, node_type, edge_time, edge_index, edge_type = build_sample_subgraph(np.arange(i, min(i + ba, sample_len)))

    gnn.eval()
    node_rep, class_rep, g_32, g_64, g_128 = gnn.forward(x.to(device),
                                    node_type.to(device),edge_time.to(device),
                                    edge_index.to(device), edge_type.to(device)) # Applies HGT
    alphas = gnn.att  # one post-softmax [E_b, n_heads] tensor per layer, still in the autograd graph

    # Class-specific attention x gradient (see the block comment above the loop).
    batch_labels = integer_labels[i:i+ba]
    batch_train = is_train_sample[i:i+ba]
    centered_logits = class_rep - class_rep.mean(dim=1, keepdim=True)
    batch_attr = np.zeros((edge_index.shape[1], n_att_layers, n_classes), dtype=np.float64)
    batch_off = np.zeros((edge_index.shape[1], n_att_layers, n_classes), dtype=np.float64)
    for c in range(n_classes):
        for sel, out in ((np.where(batch_train & (batch_labels == c))[0], batch_attr),   # on-class
                         (np.where(batch_train & (batch_labels != c))[0], batch_off)):   # off-class
            if len(sel) == 0:
                continue
            target_c = centered_logits[torch.as_tensor(sel, device=centered_logits.device), c].sum()
            grads = torch.autograd.grad(target_c, alphas, retain_graph=True, allow_unused=True)
            for l, (a, g) in enumerate(zip(alphas, grads)):
                if g is not None:  # None = this layer's attention cannot reach the logits
                    out[:, l, c] = (a * g).sum(dim=1).detach().cpu().numpy()

    # Local -> global node index: feature nodes keep their index (same offsets in every
    # batch); local sample k of this batch is global sample i + k.
    src_local = edge_index[0].numpy()
    tgt_local = edge_index[1].numpy()
    edge_src_parts.append(np.where(src_local >= n_feature_nodes, src_local + i, src_local))
    edge_tgt_parts.append(np.where(tgt_local >= n_feature_nodes, tgt_local + i, tgt_local))
    edge_type_parts.append(edge_type.numpy())
    edge_alpha_parts.append(np.stack([a.detach().cpu().numpy() for a in alphas], axis=1))
    edge_attr_parts.append(batch_attr)
    edge_off_parts.append(batch_off)

    # Feature nodes only receive messages from other feature nodes, so their
    # representations are identical in every batch: keep them once. (Appending them for
    # every batch, as before, duplicated every omic embedding row per batch whenever
    # there were > 10000 samples.)
    if i == 0:
        for _idx, _name in enumerate(OMIC_NAMES):
            omic_matrix[_name] = node_rep[node_type.to(device) == _idx, ].detach().cpu().numpy()
    else:
        # Multi-batch guard: splitting the samples into batches is only exact because
        # feature nodes never receive messages from samples, so every batch computes the
        # same feature representations as the single full-graph pass. Verify it.
        for _idx, _name in enumerate(OMIC_NAMES):
            _rep_b = node_rep[node_type.to(device) == _idx, ].detach().cpu().numpy()
            if not np.allclose(_rep_b, omic_matrix[_name], atol=1e-5):
                raise RuntimeError("Multi-batch inference is not equivalent to the full graph "
                                   f"({_name} representations differ between batches) - "
                                   "a sample->feature or sample->sample edge must have been added.")
    sample_embedding.append(node_rep[node_type.to(device) == len(OMIC_NAMES), ].detach().cpu().numpy())
    g32_embeddings.append(g_32.detach().cpu().numpy())
    g64_embeddings.append(g_64.detach().cpu().numpy())
    g128_embeddings.append(g_128.detach().cpu().numpy())
    class_embeddings.append(class_rep.detach().cpu().numpy())


# SAVING RESULTS
# np.vstack naturally handles a final partial batch, so there is no need to special-case
# samples.shape[1] % ba != 0.
sample_matrix = np.vstack(sample_embedding)
g32_embeddings = np.vstack(g32_embeddings)
g64_embeddings = np.vstack(g64_embeddings)
g128_embeddings = np.vstack(g128_embeddings)
class_embeddings = np.vstack(class_embeddings)

# Merge the per-batch edge records into one table over the whole graph. Feature->feature
# edges appear once per batch: their alpha is identical in every batch (averaged here,
# a no-op), while their attribution is summed, since each batch contributes the gradients
# of its own samples' logits. Feature->sample edges appear in exactly one batch.
_edge_src = np.concatenate(edge_src_parts).astype(np.int64)
_edge_tgt = np.concatenate(edge_tgt_parts).astype(np.int64)
_edge_type = np.concatenate(edge_type_parts)
_edge_alpha = np.concatenate(edge_alpha_parts, axis=0)
_edge_attr = np.concatenate(edge_attr_parts, axis=0)
_edge_off = np.concatenate(edge_off_parts, axis=0)
_keys = _edge_src * n_total_nodes + _edge_tgt
_uniq_keys, _inv, _mult = np.unique(_keys, return_inverse=True, return_counts=True)
n_edges = len(_uniq_keys)
edge_src = _uniq_keys // n_total_nodes
edge_tgt = _uniq_keys % n_total_nodes
edge_rel = np.empty(n_edges, dtype=_edge_type.dtype)
edge_rel[_inv] = _edge_type
n_heads_att = _edge_alpha.shape[2]
edge_alpha = np.zeros((n_edges, n_att_layers, n_heads_att))
for _l in range(n_att_layers):
    for _h in range(n_heads_att):
        edge_alpha[:, _l, _h] = np.bincount(_inv, weights=_edge_alpha[:, _l, _h], minlength=n_edges) / _mult
edge_attr = np.zeros((n_edges, n_att_layers, n_classes))
edge_off = np.zeros((n_edges, n_att_layers, n_classes))
for _l in range(n_att_layers):
    for _c in range(n_classes):
        edge_attr[:, _l, _c] = np.bincount(_inv, weights=_edge_attr[:, _l, _c], minlength=n_edges)
        edge_off[:, _l, _c] = np.bincount(_inv, weights=_edge_off[:, _l, _c], minlength=n_edges)
edge_attr /= np.maximum(class_counts, 1)[None, None, :]  # sum over class-c training samples -> mean
edge_off /= np.maximum(off_counts, 1)[None, None, :]     # sum over other training samples -> mean
del _edge_src, _edge_tgt, _edge_type, _edge_alpha, _edge_attr, _edge_off, _keys, _uniq_keys, _inv, _mult

edge_attr_total = edge_attr.sum(axis=1)                    # [E, n_classes], summed over layers
edge_off_total = edge_off.sum(axis=1)                      # [E, n_classes], summed over layers
edge_score = edge_attr_total - np.maximum(edge_off_total, 0)   # [E, n_classes], class-specific evidence
edge_attr_magnitude = np.abs(edge_attr_total).sum(axis=1)  # [E], class-agnostic strength (overview plots only)

# Node-level metadata over the global node index (each omic's features in manifest
# order, then the samples; OMIC_FEATURE_OFFSETS is defined with the graph builder) -
# shared by every table/plot below.
full_node_names = pd.concat(
    [omic_dfs[name]['ID'].reset_index(drop=True) for name in OMIC_NAMES] + [pd.Series(common_samples)],
    ignore_index=True
)
assert len(full_node_names) == n_total_nodes, "node name table does not match the graph's node count"
node_type_names = np.concatenate(
    [[_name] * encoded[_name].shape[0] for _name in OMIC_NAMES] + [['sample'] * sample_len]
)
rel_id_to_name = {v: k for k, v in edge_dict.items()}
in_degree = np.bincount(edge_tgt, minlength=n_total_nodes)
out_degree = np.bincount(edge_src, minlength=n_total_nodes)

# Edge table (one row per directed edge j -> i of the whole graph).
edge_table = {
    'source': edge_src,
    'target': edge_tgt,
    'source_name': full_node_names.values[edge_src],
    'target_name': full_node_names.values[edge_tgt],
    'source_type': node_type_names[edge_src],
    'target_type': node_type_names[edge_tgt],
    'relation': [rel_id_to_name.get(r, str(r)) for r in edge_rel],
    'target_in_degree': in_degree[edge_tgt],
}
for _l in range(n_att_layers):
    for _h in range(n_heads_att):
        edge_table[f'alpha_L{_l}_H{_h}'] = edge_alpha[:, _l, _h]
for _l in range(n_att_layers):
    _alpha_mean = edge_alpha[:, _l, :].mean(axis=1)
    edge_table[f'alpha_mean_L{_l}'] = _alpha_mean
    # log(alpha * in_degree): > 0 the target attends to this edge more than uniformly
    # (alpha = 1/deg), < 0 less. Comparable across targets with different degrees.
    edge_table[f'alpha_log_ratio_L{_l}'] = np.log(np.maximum(_alpha_mean * in_degree[edge_tgt], 1e-12))
# Attention relative to uniform (alpha x in-degree; 1 = uniform), averaged over heads and layers.
# Label-free and defined for every edge, including edges into validation/test samples: the
# edge weight of the attention-network figures.
edge_att_ratio = (edge_alpha.mean(axis=2) * in_degree[edge_tgt][:, None]).mean(axis=1)
edge_table['attention_ratio'] = edge_att_ratio
for _l in range(n_att_layers):
    for _c, _cname in enumerate(class_names):
        edge_table[f'attr_L{_l}_{_cname}'] = edge_attr[:, _l, _c]
for _l in range(n_att_layers):
    for _c, _cname in enumerate(class_names):
        edge_table[f'attr_off_L{_l}_{_cname}'] = edge_off[:, _l, _c]
for _c, _cname in enumerate(class_names):
    edge_table[f'attr_{_cname}'] = edge_attr_total[:, _c]
    edge_table[f'attr_off_{_cname}'] = edge_off_total[:, _c]
    edge_table[f'score_{_cname}'] = edge_score[:, _c]
edge_table['attr_magnitude'] = edge_attr_magnitude
edge_df = pd.DataFrame(edge_table)
edge_df.to_csv(att_dir+file0+".csv", sep=",", index=False)

# Node table: node attribution = sum of its outgoing edges' attributions.
node_attr = np.zeros((n_total_nodes, n_classes))
node_off = np.zeros((n_total_nodes, n_classes))
for _c in range(n_classes):
    node_attr[:, _c] = np.bincount(edge_src, weights=edge_attr_total[:, _c], minlength=n_total_nodes)
    node_off[:, _c] = np.bincount(edge_src, weights=edge_off_total[:, _c], minlength=n_total_nodes)
node_score = node_attr - np.maximum(node_off, 0)      # class-specific evidence per node (programs)
node_attr_magnitude = np.abs(node_attr).sum(axis=1)  # class-agnostic importance (overview plots only)
node_table = {
    'node': np.arange(n_total_nodes),
    'node_name': full_node_names.values,
    'node_type': node_type_names,
    'out_degree': out_degree,
    'in_degree': in_degree,
}
for _c, _cname in enumerate(class_names):
    node_table[f'attr_{_cname}'] = node_attr[:, _c]
    node_table[f'attr_off_{_cname}'] = node_off[:, _c]
    node_table[f'score_{_cname}'] = node_score[:, _c]
node_table['Attribution'] = node_attr_magnitude
# The condition the node gives the most class-specific evidence for ('' if none is positive).
node_table['Supported_class'] = np.where(
    node_score.max(axis=1) > 0, np.array(class_names, dtype=object)[np.argmax(node_score, axis=1)], ''
)
node_df = pd.DataFrame(node_table)
node_df.to_csv(att_dir+file0+"_node_attribution.csv", sep=",", index=False)
print(f"Interpretability: {n_edges} edges x {n_att_layers} layers x {n_heads_att} heads extracted; "
      f"{int((out_degree[:n_feature_nodes] == 0).sum())} feature nodes have no outgoing edge (attribution 0).")

for name in OMIC_NAMES:
    np.savetxt(omic_dirs[name]+file0, omic_matrix[name], delimiter=' ')
np.savetxt(sample_dir+file0, sample_matrix, delimiter=' ')
np.savetxt(embbs_dir+"g128_embeddings", g128_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"g64_embeddings", g64_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"g32_embeddings", g32_embeddings, delimiter=' ')
np.savetxt(embbs_dir+"class_embeddings", class_embeddings, delimiter=' ')



################################################################
#
# EVALUATION
#   
################################################################

# Metrics are computed per split (train / val / test) from the whole-graph inference
# above. The headline numbers and figures (confusion matrix, ROC curves, per-class
# performance) use the HELD-OUT TEST samples only. Train/val numbers are written next to
# them in classification_metrics.csv for reference, plus 'all' (every sample together).
# train and all are optimistic by construction (they include samples the model was fit on).
class_probs_all = F.softmax(torch.tensor(class_embeddings, dtype=torch.float32), dim=1).numpy()
n_classes = one_hot_labels.shape[1]

def split_metrics(idx):
    y_true = integer_labels[idx]
    probs = class_probs_all[idx]
    y_pred = np.argmax(probs, axis=1)
    out = {'n_samples': len(idx), 'accuracy': np.nan, 'f1_macro': np.nan, 'f1_weighted': np.nan,
           'f1_micro': np.nan, 'roc_auc_macro': np.nan}
    if len(idx) == 0:
        return out
    out['accuracy'] = float(np.mean(y_true == y_pred))
    out['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    out['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    out['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    try:
        if n_classes == 2:
            out['roc_auc_macro'] = roc_auc_score(y_true, probs[:, 1])
        else:
            out['roc_auc_macro'] = roc_auc_score(one_hot_labels[idx], probs, multi_class='ovr', average='macro')
    except ValueError as e:  # e.g. a class missing from a tiny split
        print(f"ROC-AUC not computable on {len(idx)} samples: {e}")
    return out

metrics_rows = []
for _split_name, _idx in (('train', train_idx), ('val', val_idx), ('test', test_idx),
                          ('all', np.arange(len(common_samples)))):  # 'all' mixes training samples in: optimistic
    _m = split_metrics(_idx)
    metrics_rows.append({'split': _split_name, **_m})
    print(f"[{_split_name}] n={_m['n_samples']} | accuracy={_m['accuracy']:.4f} | F1 macro={_m['f1_macro']:.4f} "
          f"| ROC-AUC macro={_m['roc_auc_macro']:.4f}")
pd.DataFrame(metrics_rows).to_csv(plots_dir + 'classification_metrics.csv', index=False)

# Samples the headline figures are computed on.
if len(test_idx) > 0:
    eval_idx, eval_label = test_idx, f'held-out test samples (n={len(test_idx)})'
else:
    eval_idx, eval_label = train_idx, f'TRAINING samples (n={len(train_idx)}) - no held-out test set'
eval_one_hot = one_hot_labels[eval_idx]
class_probs = class_probs_all[eval_idx]
true_labels = integer_labels[eval_idx]
pred_labels = np.argmax(class_probs, axis=1)
_m = split_metrics(eval_idx)
roc_auc, f1_macro = _m['roc_auc_macro'], _m['f1_macro']
roc_auc_per_class = {}
for _c in range(n_classes):
    if 0 < eval_one_hot[:, _c].sum() < len(eval_idx):  # needs positives and negatives
        roc_auc_per_class[_c] = roc_auc_score(eval_one_hot[:, _c], class_probs[:, _c])

# Compute and plot confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(n_classes))
# Row/column order must match the integer labels 0..k-1 that true_labels/pred_labels
# actually use, i.e. LabelEncoder's sorted class order - NOT metadata.CONDITION.unique()
# (pandas .unique() returns first-appearance order, which silently mislabels the plot
# whenever it differs from the alphabetical order LabelEncoder assigned).
condition_names = label_encoder.classes_
_row = cm.sum(axis=1, keepdims=True)
cm_pct = np.divide(cm * 100.0, _row, out=np.zeros(cm.shape), where=_row > 0)
annot = np.array([[f"{c}\n({p:.0f}%)" for c, p in zip(row_c, row_p)]
                   for row_c, row_p in zip(cm, cm_pct)])

fig, ax = plt.subplots(figsize=(1.6 * len(condition_names) + 3.5, 1.6 * len(condition_names) + 3))
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', square=True,
            xticklabels=condition_names, yticklabels=condition_names,
            cbar_kws={'label': 'Samples', 'shrink': 0.8}, linewidths=0.5, linecolor='white', ax=ax)
square_frame(ax)
ax.set_xlabel('Predicted Condition')
ax.set_ylabel('True Condition')
ax.set_title(f'Confusion Matrix - {eval_label}\nF1 (macro) = {f1_macro:.2f}  |  ROC-AUC (macro) = {roc_auc:.2f}')
plt.tight_layout()
plt.savefig(plots_dir+'confusion_matrix.pdf')
plt.close(fig)



#-----------------------------------------------------------------------
#
#----------------- Interpretability Plots ------------------------------
#
#-----------------------------------------------------------------------

# Per-omic Feature/Attribution dataframe. Built by POSITION over the global node index
# (OMIC_FEATURE_OFFSETS), so every feature keeps its own score even if it has no
# outgoing edge. (The old version dropped edge-less nodes first and then concatenated
# the full ID column against the shorter score column, which silently shifted IDs.)
omic_node_attribution = {}
for _name in OMIC_NAMES:
    _start = OMIC_FEATURE_OFFSETS[_name]
    _n = encoded[_name].shape[0]
    assert omic_dfs[_name].shape[0] == _n, f"{_name}: ID column and graph features differ in length"
    omic_node_attribution[_name] = pd.DataFrame({
        'node': np.arange(_start, _start + _n),  # global node index (unambiguous, unlike IDs)
        f'{_name}_ID': omic_dfs[_name]['ID'].values,
        'Attribution': node_attr_magnitude[_start:_start + _n],
    })
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
    _df = omic_node_attribution[_name][['node', f'{_name}_ID', 'Attribution']].copy()
    _df['Type'] = f'{_name}_feature'
    _df = _df.rename(columns={f'{_name}_ID': 'Feature'})
    omic_feature_dfs[_name] = _df
combined = pd.concat(list(omic_feature_dfs.values()), ignore_index=True)
combined = combined.sort_values('Attribution', ascending=False).reset_index(drop=True)
# --------------------------
# Cumulative attention + saturation
# --------------------------
combined['Cumulative_Attribution'] = combined['Attribution'].cumsum()
combined['Cumulative_Attribution_%'] = 100 * combined['Cumulative_Attribution'] / combined['Attribution'].sum()
saturation_threshold = 95
saturation_idx = (combined['Cumulative_Attribution_%'] >= saturation_threshold).idxmax()
saturation_x = saturation_idx + 1
saturation_y = combined.loc[saturation_idx, 'Cumulative_Attribution_%']
#top_features = combined.iloc[:saturation_x]
top_features = combined.iloc[:min(combined.shape[0],300)]
# --------------------------
# Figure: top 10 features per omic, top 10 overall, cumulative curve (square panels)
# --------------------------
N_TOP_BARS = 10
omic_top = {_name: _df.sort_values('Attribution', ascending=False).head(N_TOP_BARS)
            for _name, _df in omic_feature_dfs.items()}
omic_total_attribution = {_name: _df['Attribution'].sum() for _name, _df in omic_feature_dfs.items()}
_n_panels = len(OMIC_NAMES) + 2
_ncols = _n_panels if _n_panels <= 4 else 3  # one row for up to 2 omics
_nrows = int(np.ceil(_n_panels / _ncols))
fig, _axes = plt.subplots(_nrows, _ncols, figsize=(5.4 * _ncols, 5.6 * _nrows), squeeze=False)
_axes = _axes.ravel()

def _bar_panel(ax, df, colors, title):
    y = np.arange(len(df))[::-1]
    ax.barh(y, df['Attribution'].values, color=colors, edgecolor='black', linewidth=0.5, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([str(f) if len(str(f)) <= 24 else str(f)[:23] + '…' for f in df['Feature']], fontsize=8)
    ax.set_ylim(-0.6, N_TOP_BARS - 0.4)
    ax.set_xlabel('Attribution  Σ_c |attention × gradient|', fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    square_frame(ax)

for _oi, _name in enumerate(OMIC_NAMES):
    _bar_panel(_axes[_oi], omic_top[_name], OMIC_COLORS[_name],
               f'Top {N_TOP_BARS} {_name} features\nTotal: {omic_total_attribution[_name]:.3g} | '
               f'Mean: {omic_total_attribution[_name] / len(omic_feature_dfs[_name]):.3g}')
_type_to_color = {f'{_name}_feature': OMIC_COLORS[_name] for _name in OMIC_NAMES}
_top_all = combined.head(N_TOP_BARS)
_ax = _axes[len(OMIC_NAMES)]
_bar_panel(_ax, _top_all, _top_all['Type'].map(_type_to_color).values,
           f'Top {N_TOP_BARS} features, all omics\n({saturation_x} features reach {saturation_threshold}% of the total)')
_ax.legend(handles=[Patch(facecolor=OMIC_COLORS[_name], edgecolor='black', label=_name) for _name in OMIC_NAMES],
           loc='lower right', fontsize=8)
# Cumulative attribution curve
ax3 = _axes[len(OMIC_NAMES) + 1]
ax3.plot(np.arange(1, len(combined) + 1), combined['Cumulative_Attribution_%'], color='#333333', linewidth=1.5)
ax3.axvline(x=saturation_x, color=OKABE_ITO[4], linestyle='--', linewidth=1.2)
ax3.axhline(y=saturation_y, color=OKABE_ITO[4], linestyle='--', linewidth=1.2)
ax3.text(saturation_x, saturation_y - 5, f' {saturation_x} features\n ({saturation_y:.1f}%)',
         color=OKABE_ITO[4], fontsize=9, ha='left', va='top')
ax3.set_xlabel('Number of top features', fontsize=9)
ax3.set_ylabel('Cumulative attribution (%)', fontsize=9)
ax3.set_title('Cumulative feature attribution', fontsize=11)
square_frame(ax3)
for _ax in _axes[_n_panels:]:
    _ax.set_visible(False)
plt.tight_layout()
plt.savefig(plots_dir+'interpretability_nodes_attention.pdf')
plt.close()



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
    'Cell Nodes': sample_matrix,  # every sample (node_rep only holds the last inference batch)
    'g128 Embedding': to_numpy(g128_embeddings),
    'g64 Embedding': to_numpy(g64_embeddings),
    'g32 Embedding': to_numpy(g32_embeddings),
    'Class Rep': class_embeddings  # every sample (class_rep only holds the last inference batch)
}
# ----------------------------------
# Figure: five square panels (sample node embedding, g128, g64, g32, class representation)
# ----------------------------------
def _scatter_by_condition(ax, xy, title, s=20):
    for cond in labels.unique():
        idx = (labels == cond).values
        ax.scatter(xy[idx, 0], xy[idx, 1], label=cond, s=s, color=CONDITION_COLORS[str(cond)],
                   edgecolors='white', linewidths=0.3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('UMAP1', fontsize=9)
    ax.set_ylabel('UMAP2', fontsize=9)
    square_frame(ax)

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
cell_nodes_umap = reduce_embedding(embeddings['Cell Nodes'], n_components=2)
_scatter_by_condition(ax1, cell_nodes_umap, 'UMAP of sample node embeddings')
ax1.legend(markerscale=1.5, fontsize=8)
umap128 = reduce_embedding(embeddings['g128 Embedding'], n_components=2)
_scatter_by_condition(fig.add_subplot(gs[0, 1]), umap128, 'UMAP of g128 embedding')
umap64 = reduce_embedding(embeddings['g64 Embedding'], n_components=2)
_scatter_by_condition(fig.add_subplot(gs[0, 2]), umap64, 'UMAP of g64 embedding')
umap32 = reduce_embedding(embeddings['g32 Embedding'], n_components=2)
_scatter_by_condition(fig.add_subplot(gs[1, 0]), umap32, 'UMAP of g32 embedding')
# Class representation: raw when 2-D, 3-D scatter when 3-D, UMAP otherwise
class_rep_np = embeddings['Class Rep']
D = class_rep_np.shape[1]
if D == 3:
    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    for cond in labels.unique():
        idx = (labels == cond).values
        ax5.scatter(class_rep_np[idx, 0], class_rep_np[idx, 1], class_rep_np[idx, 2], label=cond, s=20,
                    color=CONDITION_COLORS[str(cond)])
    ax5.set_title('3D class representation', fontsize=11)
    ax5.view_init(elev=20, azim=45)
    for axis in (ax5.xaxis, ax5.yaxis, ax5.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor('#000000')
    ax5.grid(False)
else:
    ax5 = fig.add_subplot(gs[1, 1])
    if D == 2:
        _scatter_by_condition(ax5, class_rep_np, '2D class representation')
        ax5.set_xlabel('logit 1', fontsize=9)
        ax5.set_ylabel('logit 2', fontsize=9)
    else:
        _scatter_by_condition(ax5, reduce_embedding(class_rep_np, n_components=2), f'UMAP of class representation ({D}D)')
plt.savefig(plots_dir+'embeddings_evaluations.pdf', bbox_inches='tight')
plt.close()



#-----------------------------------------------------------------------
#
#----------------- Multi-omic programs (one analysis per target condition)
#
#-----------------------------------------------------------------------
# For every target condition c in --target_class:
#   1. Features: the top <= 300 features with a POSITIVE class-specific score
#      score_c = attr_c - max(attr_off_c, 0) (see the inference block): evidence the model
#      uses to call c in c-samples, without also pushing the other samples towards c.
#   2. Edges: the graph's existing feature-feature edges between those features. The
#      weight of a pair is max(score_c(i->j), 0) + max(score_c(j->i), 0): attention is
#      not symmetric, so each direction is clipped before adding them. Leiden
#      (RBConfiguration) needs non-negative weights.
#   3. Programs: Leiden clusters with >= --leiden_min_cluster_size features, numbered
#      P1, P2, ... from the largest.
#   4. Activity per sample: A(s) = sum_f w_f * sigma_f * z(f, s) / sum_f w_f, where
#      w_f = score_c(f) is the importance (model), z the input matrix z-scored per feature
#      with the TRAINING samples' mean/SD (log1p-transformed first when the input is
#      non-negative, e.g. counts or intensities), and sigma_f = +-1 the direction (data):
#      the sign of f's loading on the program's first principal component of z over the
#      training samples (label-free; oriented so the highest-weight member is +). Up- and
#      down-regulated members add up instead of cancelling; w_f * sigma_f is the member's
#      signed weight in the activity.
#   5. Statistics per program (as in the PTB program figures): Mann-Whitney U, c vs rest,
#      and the difference of mean activity (c - rest, SD units) on ALL samples, with
#      Benjamini-Hochberg q across the condition's programs. Most samples trained the
#      model that defined the programs, so this is descriptive; the same test on the
#      HELD-OUT TEST samples only is reported next to it.
# Outputs: plots/programs_<condition>/ and supplementary/programs_<condition>/.

import re
import igraph as ig
import leidenalg
from collections import defaultdict, Counter
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu, kruskal
from pramigo_figures import bh_fdr, square_frame, draw_program_network, draw_program_members

# Global feature index -> omic name.
FEATURE_OMIC_SIZES = {name: omic_matrices[name].shape[0] for name in OMIC_NAMES}
FEATURE_INDEX_TO_OMIC = np.concatenate([[name] * FEATURE_OMIC_SIZES[name] for name in OMIC_NAMES])

condition_labels = metadata['CONDITION'].reset_index(drop=True).astype(str).values
unique_conditions = sorted(pd.unique(condition_labels))
is_heldout_test = np.zeros(len(common_samples), dtype=bool)
is_heldout_test[test_idx] = True

# Activity input: per-feature z-score with TRAINING statistics (log1p first if non-negative).
_X_act = samples_stacked.astype(float)
if np.nanmin(_X_act) >= 0:
    _X_act = np.log1p(_X_act)
    ACTIVITY_INPUT = 'log1p + z-scored inputs'
else:
    ACTIVITY_INPUT = 'z-scored inputs'
_mu = _X_act[:, train_idx].mean(axis=1, keepdims=True)
_sd = _X_act[:, train_idx].std(axis=1, keepdims=True)
_sd[~np.isfinite(_sd) | (_sd == 0)] = 1.0
Z_ACT = (_X_act - _mu) / _sd
del _X_act, _mu, _sd
print(f"Program activity computed on {ACTIVITY_INPUT} (training-sample mean/SD).")


def _mw(values, is_target):
    v_t, v_r = values[is_target], values[~is_target]
    if len(v_t) == 0 or len(v_r) == 0:
        return np.nan, np.nan
    _, p = mannwhitneyu(v_t, v_r, alternative='two-sided')
    return p, v_t.mean() - v_r.mean()


def draw_cluster_subgraph(G_full, target_cluster, pname, tname, figsize=(11, 11), seed=7):
    """One Leiden program's subnetwork, on the same red-white-blue scale as the report's signed
    weights: node colour = the member's signed activity weight (score x sigma; red raises the
    activity, blue lowers it), node size = |signed weight|; edge colour/width = the edge weight
    used by Leiden (always >= 0, so white to red). Omic -> marker shape."""
    cluster_nodes = [n for n, d in G_full.nodes(data=True) if d.get('leiden_cluster') == target_cluster]
    if not cluster_nodes:
        return None
    G_sub = G_full.subgraph(cluster_nodes).copy()
    pos = robust_layout(G_sub, seed=seed)
    cmap = plt.cm.get_cmap('RdBu_r')
    node_vals = np.array([G_sub.nodes[n].get('signed_weight', G_sub.nodes[n]['score']) for n in G_sub.nodes])
    vlim = float(np.abs(node_vals).max()) or 1.0
    node_norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)
    val_of = dict(zip(G_sub.nodes, node_vals))
    edge_ws = [d.get('weight', 1.0) for _, _, d in G_sub.edges(data=True)]
    wmax = max(edge_ws) if edge_ws else 1.0

    fig, ax = plt.subplots(figsize=figsize, facecolor='#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    for u, v, d in G_sub.edges(data=True):
        rel = d.get('weight', 1.0) / wmax
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=cmap(0.5 + 0.5 * rel), lw=0.5 + rel * 4.5,
                alpha=0.8, zorder=1)
    nodes_by_omic = {_name: [n for n in G_sub.nodes if G_sub.nodes[n].get('type') == _name] for _name in OMIC_NAMES}
    for _name in OMIC_NAMES:
        g_nodes = nodes_by_omic[_name]
        if not g_nodes:
            continue
        ax.scatter([pos[n][0] for n in g_nodes], [pos[n][1] for n in g_nodes],
                   s=[200 + 800 * abs(val_of[n]) / vlim for n in g_nodes], c=[cmap(node_norm(val_of[n])) for n in g_nodes],
                   marker=OMIC_MARKERS_BY_NAME[_name], edgecolors='black', linewidths=0.8, zorder=3)
    for n in G_sub.nodes:
        txt = ax.text(pos[n][0], pos[n][1], G_sub.nodes[n].get('name', str(n)), color='#000000', fontsize=6,
                      fontweight='bold', ha='center', va='center', zorder=4)
        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='#FFFFFF')])
    ax.legend(handles=[Line2D([0], [0], marker=OMIC_MARKERS_BY_NAME[_name], color='w', markerfacecolor='#999999',
                              markeredgecolor='black', markersize=10, label=_name) for _name in OMIC_NAMES],
              loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#E0E0E0', fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=node_norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=fig.add_axes([0.25, 0.04, 0.5, 0.02]), orientation='horizontal')
    cb.set_label(f"Nodes: signed weight score_{tname} × σ (red raises, blue lowers the activity)\n"
                 "Edges: Leiden weight (white → red)", fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"'{tname}' program {pname}\n"
                 f'({" · ".join(f"{len(nodes_by_omic[_name])} {_name}" for _name in OMIC_NAMES)})',
                 fontsize=13, fontweight='bold', pad=18)
    _frame_axes_to_pos(ax, pos)
    ax.axis('off')
    return fig


def plot_specificity_scatter(tc, tname, highlight_idx, path):
    """Diagnostic: on-class attribution (x) vs off-class attribution (y) for every feature.
    x > 0, y <= 0: specific evidence for the target; x > 0, y > 0: pushes every sample
    towards the target (non-specific, penalized in the score); x < 0: counter-evidence."""
    x_on = node_attr[:n_feature_nodes, tc]
    y_off = node_off[:n_feature_nodes, tc]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for _name in OMIC_NAMES:
        _sl = slice(OMIC_FEATURE_OFFSETS[_name], OMIC_FEATURE_OFFSETS[_name] + FEATURE_OMIC_SIZES[_name])
        ax.scatter(x_on[_sl], y_off[_sl], s=10, alpha=0.6, color=OMIC_COLORS[_name], label=_name)
    if len(highlight_idx):
        ax.scatter(x_on[highlight_idx], y_off[highlight_idx], s=28, facecolors='none', edgecolors='black',
                   linewidths=0.6, label='program members')
    ax.axhline(0, color='#888888', linewidth=0.8)
    ax.axvline(0, color='#888888', linewidth=0.8)
    ax.text(0.98, 0.02, 'specific evidence', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#444444')
    ax.text(0.98, 0.98, 'non-specific (pushes all\nsamples towards target)', transform=ax.transAxes, ha='right',
            va='top', fontsize=8, color='#444444')
    ax.text(0.02, 0.5, 'counter-evidence', transform=ax.transAxes, ha='left', va='center', fontsize=8, color='#444444')
    ax.set_xlabel(f"attr_{tname}: push towards '{tname}' in '{tname}' training samples")
    ax.set_ylabel(f"attr_off_{tname}: push towards '{tname}' in other training samples")
    ax.set_title(f"Attribution specificity for '{tname}' (per feature)")
    ax.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def run_program_analysis(tc):
    tname = class_names[tc]
    tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', tname)
    out_dir = plots_dir + f'programs_{tag}/'
    supp_dir = supplementary_dir + f'programs_{tag}/'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(supp_dir, exist_ok=True)
    is_target = condition_labels == tname
    print(f"\n=== Programs for target condition '{tname}' -> {out_dir} ===")

    # 1. Features with a positive class-specific score (top <= 300).
    f_score = node_score[:n_feature_nodes, tc]
    cand = np.where(f_score > 0)[0]
    top = cand[np.argsort(f_score[cand])[::-1][:300]]
    print(f"Programs [{tname}]: {len(cand)} features have a positive score; using the top {len(top)}")

    G_leiden = nx.Graph()
    for idx in top:
        G_leiden.add_node(int(idx), name=str(full_node_names.iloc[idx]).strip(), type=FEATURE_INDEX_TO_OMIC[idx],
                          score=float(node_score[idx, tc]), attr=float(node_attr[idx, tc]),
                          attr_off=float(node_off[idx, tc]))

    # 2. Edge weights: clip each direction at 0, then add the two directions of a pair.
    in_top = np.zeros(n_total_nodes, dtype=bool)
    in_top[top] = True
    m = in_top[edge_src] & in_top[edge_tgt] & (edge_src != edge_tgt)
    pair_df = pd.DataFrame({'a': np.minimum(edge_src[m], edge_tgt[m]), 'b': np.maximum(edge_src[m], edge_tgt[m]),
                            'w': np.maximum(edge_score[m, tc], 0)})
    pair_df = pair_df.groupby(['a', 'b'], as_index=False)['w'].sum()
    for a, b, w in zip(pair_df['a'], pair_df['b'], pair_df['w']):
        if w <= 0:
            continue
        ta, tb = G_leiden.nodes[a]['type'], G_leiden.nodes[b]['type']
        G_leiden.add_edge(int(a), int(b), weight=float(w),
                          edge_type=f'{ta}-{ta}' if ta == tb else '-'.join(sorted((ta, tb))))
    isolated = list(nx.isolates(G_leiden))
    G_leiden.remove_nodes_from(isolated)
    print(f"Programs [{tname}]: removed {len(isolated)} features without a positive-weight edge; "
          f"{G_leiden.number_of_nodes()} features, {G_leiden.number_of_edges()} edges left")
    plot_specificity_scatter(tc, tname, np.array(list(G_leiden.nodes()), dtype=int),
                             supp_dir + 'attribution_specificity_scatter.pdf')

    if G_leiden.number_of_nodes() == 0 or G_leiden.number_of_edges() == 0:
        print(f"Programs [{tname}]: no connected top-feature subgraph found, skipping cluster analysis.")
        return

    # 3. Leiden.
    nx_nodes = list(G_leiden.nodes())
    node_id_map = {node: k for k, node in enumerate(nx_nodes)}
    g_ig = ig.Graph()
    g_ig.add_vertices(len(nx_nodes))
    g_ig.add_edges([(node_id_map[u], node_id_map[v]) for u, v in G_leiden.edges()])
    g_ig.es['weight'] = [G_leiden[u][v]['weight'] for u, v in G_leiden.edges()]
    partition = leidenalg.find_partition(g_ig, leidenalg.RBConfigurationVertexPartition, weights=g_ig.es['weight'],
                                         resolution_parameter=args.leiden_resolution, seed=seed)
    for nx_node, cluster_id in zip(nx_nodes, partition.membership):
        G_leiden.nodes[nx_node]['leiden_cluster'] = cluster_id
    node_to_cluster = nx.get_node_attributes(G_leiden, 'leiden_cluster')
    cluster_sizes = Counter(node_to_cluster.values())
    valid_clusters = {cid for cid, size in cluster_sizes.items() if size >= args.leiden_min_cluster_size}
    print(f"Programs [{tname}]: Leiden found {len(cluster_sizes)} clusters, keeping {len(valid_clusters)} "
          f"with >= {args.leiden_min_cluster_size} features")
    # P1 = largest program (ties: lower Leiden id first)
    cids = sorted(valid_clusters, key=lambda c: (-cluster_sizes[c], c))
    pname = {c: f'P{k + 1}' for k, c in enumerate(cids)}

    def members_of(cid):  # sorted by score, highest first
        return sorted([n for n, c in node_to_cluster.items() if c == cid],
                      key=lambda n: -G_leiden.nodes[n]['score'])

    # 4. Program activity (weights from the model, directions from the data).
    program_activity = pd.DataFrame(index=common_samples)
    for cid in cids:
        mem = members_of(cid)
        w = np.array([G_leiden.nodes[n]['score'] for n in mem])
        Xc = Z_ACT[mem][:, train_idx]
        Xc = Xc - Xc.mean(axis=1, keepdims=True)
        sigma = np.ones(len(mem))
        if Xc.shape[1] >= 2 and np.any(Xc):
            u_svd, _, _ = np.linalg.svd(Xc, full_matrices=False)
            sigma = np.sign(u_svd[:, 0])
            sigma[sigma == 0] = 1.0
            if sigma[np.argmax(w)] < 0:
                sigma = -sigma
        for n, s_f, w_f in zip(mem, sigma, w):
            G_leiden.nodes[n]['direction'] = int(s_f)
            G_leiden.nodes[n]['signed_weight'] = float(w_f * s_f)
        program_activity[pname[cid]] = (w * sigma) @ Z_ACT[mem] / w.sum()
    program_activity.index.name = 'ID'

    # Composition CSV (one row per clustered feature).
    cluster_rows = [{
        'Feature': d['name'], 'Node_Type': d['type'], 'node': n,
        f'score_{tname}': d['score'], f'attr_{tname}': d['attr'], f'attr_off_{tname}': d['attr_off'],
        'Leiden_Cluster': d['leiden_cluster'], 'Program': pname.get(d['leiden_cluster'], ''),
        'Cluster_Size': cluster_sizes[d['leiden_cluster']],
        'Kept_In_Network_Plot': d['leiden_cluster'] in valid_clusters,
        'Activity_direction': d.get('direction', np.nan), 'Activity_weight': d.get('signed_weight', np.nan),
    } for n, d in G_leiden.nodes(data=True)]
    cluster_df = pd.DataFrame(cluster_rows).sort_values(['Cluster_Size', 'Leiden_Cluster', f'score_{tname}'],
                                                         ascending=[False, True, False]).reset_index(drop=True)
    cluster_df.to_csv(out_dir + 'leiden_clusters_composition.csv', index=False)
    print(f"Programs [{tname}]: cluster composition saved to {out_dir}leiden_clusters_composition.csv")

    if not valid_clusters:
        print(f"Programs [{tname}]: no clusters passed the minimum size filter, skipping program plots.")
        return

    _act_out = program_activity.copy()
    _act_out.insert(0, 'split', sample_split)
    _act_out.insert(0, 'CONDITION', condition_labels)
    _act_out.to_csv(out_dir + 'leiden_program_activity_scores.csv')

    # 5. Statistics: all samples (BH across this condition's programs) + held-out test samples.
    stat_rows = []
    for cid in cids:
        v = program_activity[pname[cid]].values
        p_all, d_all = _mw(v, is_target)
        p_te, d_te = _mw(v[is_heldout_test], is_target[is_heldout_test])
        stat_rows.append({'Program': pname[cid], 'Leiden_Cluster': cid, 'size': cluster_sizes[cid],
                          'delta_mean_all': d_all, 'p_all': p_all,
                          'n_target_all': int(is_target.sum()), 'n_rest_all': int((~is_target).sum()),
                          'delta_mean_test': d_te, 'p_test': p_te,
                          'n_target_test': int((is_target & is_heldout_test).sum()),
                          'n_rest_test': int((~is_target & is_heldout_test).sum())})
    stats_df = pd.DataFrame(stat_rows)
    stats_df.insert(5, 'q_all', bh_fdr(stats_df['p_all'].values))
    stats_df['q_test'] = bh_fdr(stats_df['p_test'].values)
    stats_df.to_csv(out_dir + 'program_statistics.csv', index=False)
    stats_by = stats_df.set_index('Program')

    # Contracted program network (option-4 style: omic-composition pies, labels beside nodes).
    G_cluster = nx.Graph()
    for cid in cids:
        mem = members_of(cid)
        types = [G_leiden.nodes[n]['type'] for n in mem]
        G_cluster.add_node(cid, size=len(mem), dominant_type=max(set(types), key=types.count))
    acc, acc_types = defaultdict(float), defaultdict(list)
    for u, v, d in G_leiden.edges(data=True):
        cu, cv = node_to_cluster[u], node_to_cluster[v]
        if cu not in valid_clusters or cv not in valid_clusters or cu == cv:
            continue
        acc[tuple(sorted((cu, cv)))] += d['weight']
        acc_types[tuple(sorted((cu, cv)))].append(d['edge_type'])
    for (cu, cv), w in acc.items():
        G_cluster.add_edge(cu, cv, weight=w, edge_type=max(set(acc_types[(cu, cv)]), key=acc_types[(cu, cv)].count))
    pos_c = nx.spring_layout(G_cluster, k=1.2, iterations=500, seed=7, weight='weight')
    programs_info = []
    for cid in cids:
        mem = members_of(cid)
        types = [G_leiden.nodes[n]['type'] for n in mem]
        programs_info.append({'key': cid, 'name': pname[cid], 'size': len(mem),
                              'counts': {t: types.count(t) for t in OMIC_NAMES},
                              'top': [G_leiden.nodes[n]['name'] for n in mem[:5]]})
    fig = draw_program_network(
        programs_info, pos_c, [(u, v, d['weight']) for u, v, d in G_cluster.edges(data=True)], OMIC_NAMES,
        OMIC_COLORS, f"Multi-omic programs for '{tname}' in the PRAMIGO attention network",
        f"{len(cids)} Leiden programs (≥ {args.leiden_min_cluster_size} features; "
        f"{sum(p['size'] for p in programs_info)} features) · node size = features (log) · "
        f"edge width = summed inter-program score · labels = top 5 features by score_{tname}")
    fig.savefig(out_dir + "leiden_cluster_network.pdf", bbox_inches="tight")
    plt.close(fig)

    with PdfPages(out_dir + "leiden_cluster_subgraphs.pdf") as pdf:
        for cid in cids:
            fig = draw_cluster_subgraph(G_leiden, cid, pname[cid], tname)
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
                plt.close(fig)

    # Program report: one page per program, 2 x 2 square panels.
    #   A sample UMAP by condition       B sample UMAP by program activity
    #   C top members, signed weight     D activity by condition + statistics
    with PdfPages(out_dir + "leiden_program_activity_report.pdf") as pdf:
        for cid in cids:
            pn = pname[cid]
            activity_values = program_activity[pn].values
            mem = members_of(cid)
            st = stats_by.loc[pn]
            fig = plt.figure(figsize=(10, 10.6))
            gsr = gridspec.GridSpec(2, 2, left=0.17, right=0.94, bottom=0.12, top=0.88, wspace=0.45, hspace=0.38)

            ax0 = fig.add_subplot(gsr[0, 0])
            for cond in unique_conditions:
                mk = condition_labels == cond
                ax0.scatter(cell_nodes_umap[mk, 0], cell_nodes_umap[mk, 1], color=CONDITION_COLORS[cond],
                            label=cond, s=22, alpha=0.9, edgecolors='white', linewidths=0.3)
            ax0.set_title('A  Sample UMAP – condition', fontsize=11, loc='left')
            ax0.legend(fontsize=7, markerscale=1.2, loc='best')
            ax0.set_xlabel('UMAP1'); ax0.set_ylabel('UMAP2')
            square_frame(ax0)

            ax1 = fig.add_subplot(gsr[0, 1])
            _lim = float(np.nanmax(np.abs(activity_values))) or 1.0
            sc = ax1.scatter(cell_nodes_umap[:, 0], cell_nodes_umap[:, 1], c=activity_values, cmap='RdBu_r',
                             norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-_lim, vmax=_lim), s=26,
                             edgecolors='black', linewidths=0.3)
            plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label=f'{pn} activity')
            ax1.set_title(f'B  Sample UMAP – {pn} activity', fontsize=11, loc='left')
            ax1.set_xlabel('UMAP1'); ax1.set_ylabel('UMAP2')
            square_frame(ax1)

            ax2 = fig.add_subplot(gsr[1, 0])
            e = draw_program_members(
                ax2, [G_leiden.nodes[n]['name'] for n in mem], [G_leiden.nodes[n]['signed_weight'] for n in mem],
                [G_leiden.nodes[n]['type'] for n in mem], OMIC_NAMES, OMIC_MARKERS_BY_NAME,
                f'C  {pn} (n = {len(mem)}): signed weight, top 10', n_rows=10, fs=7.5)
            ax2.set_title(ax2.get_title(loc='left'), fontsize=11, loc='left', fontweight='bold')
            ax2.set_xlabel(f'score_{tname} × σ (×10$^{{{e}}}$)', fontsize=8)
            handles = [Patch(fc=plt.get_cmap('RdBu_r')(0.9), ec='black', lw=0.6, label='higher value raises activity'),
                       Patch(fc=plt.get_cmap('RdBu_r')(0.1), ec='black', lw=0.6, label='higher value lowers activity')]
            handles += [Line2D([], [], ls='', marker=OMIC_MARKERS_BY_NAME[t], ms=4, color='black', label=t)
                        for t in OMIC_NAMES[1:]]
            ax2.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.5,
                       frameon=False, handlelength=1.2, columnspacing=1.0)

            ax3 = fig.add_subplot(gsr[1, 1])
            box_conds = [g for g in unique_conditions if np.any(condition_labels == g)]
            bp = ax3.boxplot([activity_values[condition_labels == g] for g in box_conds], labels=box_conds,
                             patch_artist=True, showfliers=False, widths=0.55)
            for patch, cond in zip(bp['boxes'], box_conds):
                patch.set_facecolor(CONDITION_COLORS[cond])
                patch.set_alpha(0.55)
            for med in bp['medians']:
                med.set_color('black')
            rng_jit = np.random.RandomState(0)
            for k_, cond in enumerate(box_conds):
                mk = condition_labels == cond
                xj = rng_jit.normal(k_ + 1, 0.06, size=mk.sum())
                yv = activity_values[mk]
                held = is_heldout_test[mk]
                ax3.scatter(xj[~held], yv[~held], s=14, color='black', alpha=0.7, zorder=3)
                ax3.scatter(xj[held], yv[held], s=18, facecolors='white', edgecolors='black', linewidths=0.8, zorder=4)
            ax3.set_title(f'D  {pn} activity: {tname} vs rest', fontsize=11, loc='left')
            ax3.set_xlabel('Condition'); ax3.set_ylabel(f'{pn} activity (SD units)')
            square_frame(ax3)
            fmt = lambda x: 'n/a' if not np.isfinite(x) else f'{x:.3g}'
            ax3.text(0.0, -0.20,
                     f"All samples ({int(st.n_target_all)} vs {int(st.n_rest_all)}): Δ mean = {fmt(st.delta_mean_all)}, "
                     f"Mann–Whitney p = {fmt(st.p_all)}, BH q = {fmt(st.q_all)}\n"
                     f"Held-out test ({int(st.n_target_test)} vs {int(st.n_rest_test)}): Δ mean = {fmt(st.delta_mean_test)}, "
                     f"p = {fmt(st.p_test)}\n● training/validation sample   ○ held-out test sample",
                     transform=ax3.transAxes, fontsize=7, va='top', ha='left', color='#333333')

            fig.text(0.06, 0.955, f"'{tname}' program {pn} (Leiden cluster {cid}, {len(mem)} features)",
                     fontsize=14, fontweight='bold', va='bottom')
            fig.text(0.06, 0.945, f"A(s) = Σ w·σ·z(s) / Σ w, w = score_{tname} (attention × gradient, training "
                     f"samples), z = {ACTIVITY_INPUT} (training mean/SD), σ = sign of the member's loading on the "
                     "training PC1 (top-weight member +).", fontsize=7.5, color='#4a4a4a', va='top', wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Programs [{tname}]: network, subgraphs, activity scores, statistics and report saved to {out_dir}")

    # Supplementary (per target condition).
    size_summary = sorted([(pname[cid], G_cluster.nodes[cid]['size'], G_cluster.nodes[cid]['dominant_type'])
                           for cid in cids], key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(size_summary))))
    ax.barh([p for p, _, _ in size_summary], [s_ for _, s_, _ in size_summary],
            color=[OMIC_COLORS[t] for _, _, t in size_summary], edgecolor='black', linewidth=0.4)
    ax.set_xlabel('Number of Features')
    ax.set_title(f"'{tname}' Program Sizes")
    ax.legend(handles=[Patch(facecolor=OMIC_COLORS[_name], label=f'{_name}-dominant') for _name in OMIC_NAMES],
              loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(supp_dir + "program_size_ranked.pdf")
    plt.close(fig)

    labels_p = [pname[c] for c in cids]
    conn = pd.DataFrame(0.0, index=labels_p, columns=labels_p)
    for u, v, d in G_cluster.edges(data=True):
        conn.loc[pname[u], pname[v]] = d['weight']
        conn.loc[pname[v], pname[u]] = d['weight']
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(cids) + 2), max(4, 0.6 * len(cids) + 2)))
    sns.heatmap(conn, cmap=SEQ_CMAP, square=True, cbar_kws={'label': 'Sum of class-specific edge weights'}, ax=ax)
    ax.set_title(f"'{tname}' Inter-Program Connectivity")
    plt.tight_layout()
    plt.savefig(supp_dir + "program_connectivity_heatmap.pdf")
    plt.close(fig)

    with PdfPages(supp_dir + "program_feature_correlation_heatmap.pdf") as pdf:
        for cid in cids:
            mem = members_of(cid)
            if len(mem) < 2:
                continue
            names_m = [G_leiden.nodes[n]['name'] for n in mem]
            corr = np.corrcoef(Z_ACT[mem][:, train_idx]) if len(train_idx) > 1 else np.eye(len(mem))
            fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(mem) + 2), max(4, 0.35 * len(mem) + 2)))
            sns.heatmap(corr, cmap=DIV_CMAP, vmin=-1, vmax=1, center=0, xticklabels=names_m, yticklabels=names_m,
                        cbar_kws={'label': 'Pearson r'}, ax=ax, square=True)
            ax.tick_params(axis='both', labelsize=6)
            ax.set_title(f"'{tname}' {pname[cid]}: Feature-Feature Correlation (training samples)")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    activity_z = (program_activity - program_activity.mean()) / program_activity.std(ddof=0).replace(0, 1)
    activity_z = activity_z.replace([np.inf, -np.inf], np.nan).fillna(0)
    row_colors = pd.Series(condition_labels, index=program_activity.index).map(CONDITION_COLORS)
    row_colors.name = 'Condition'
    g_act = sns.clustermap(activity_z, cmap=DIV_CMAP, center=0, row_colors=row_colors,
                           col_cluster=activity_z.shape[1] > 1, row_cluster=activity_z.shape[0] > 1,
                           figsize=(max(8, 0.5 * activity_z.shape[1] + 3), 9),
                           cbar_kws={'label': 'Activity (z-score)'}, xticklabels=True, yticklabels=False)
    g_act.ax_heatmap.set_xlabel('Program')
    g_act.ax_heatmap.set_ylabel('Sample')
    g_act.fig.suptitle(f"'{tname}' Program Activity Across Samples", y=1.02)
    g_act.savefig(supp_dir + "program_activity_heatmap.pdf", bbox_inches='tight')
    plt.close(g_act.fig)
    if program_activity.shape[1] > 1:
        prog_corr = program_activity.corr().fillna(0)
        g_corr = sns.clustermap(prog_corr, cmap=DIV_CMAP, vmin=-1, vmax=1, center=0,
                                figsize=(max(6, 0.5 * prog_corr.shape[0] + 3), max(6, 0.5 * prog_corr.shape[0] + 3)),
                                cbar_kws={'label': 'Pearson r'})
        g_corr.fig.suptitle(f"'{tname}' Program-Program Activity Correlation", y=1.02)
        g_corr.savefig(supp_dir + "program_activity_correlation.pdf", bbox_inches='tight')
        plt.close(g_corr.fig)
    print(f"Programs [{tname}]: supplementary program figures saved to {supp_dir}")


for _tc in TARGET_CLASS_IDX:
    run_program_analysis(_tc)


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

# Step 1: Select top 100 features per omic by attribution
top_n = 100
top_by_omic = {_name: omic_node_attribution[_name].sort_values(by='Attribution', ascending=False).head(top_n)
                for _name in OMIC_NAMES}
# Edge weight = attention relative to uniform (alpha x in-degree, mean over heads and layers):
# comparable across targets with different degrees, label-free, and defined for every sample
# (attribution is computed on training samples only, so it would drop held-out samples).
attention_network = pd.DataFrame({'source': edge_src, 'target': edge_tgt, 'weight': edge_att_ratio})

# Get all samples
samples = metadata[['ID', 'CONDITION']].reset_index(drop=True)

# Create node lists
node_labels = pd.concat(
    [top_by_omic[_name][f'{_name}_ID'].reset_index(drop=True) for _name in OMIC_NAMES]
    + [samples['ID'].reset_index(drop=True)]
)
node_attention = np.concatenate(
    [top_by_omic[_name]['Attribution'].values for _name in OMIC_NAMES]
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
# Node ids are the global node index (each omic's features in manifest order, then
# the samples), so they line up with attention_network's source/target ids.
omic_indices = {_name: top_by_omic[_name]['node'].values for _name in OMIC_NAMES}
sample_indices = np.array(range(num_samples)) + n_feature_nodes
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

# Add edges with their attribution weight (both directions of a pair are summed, as in
# the Leiden graph). robust_layout() normalizes these weights for the layout.
for i, j, w in zip(attention_filtered['source'], attention_filtered['target'], attention_filtered['weight']):
    if G.has_edge(i, j):
        G[i][j]['weight'] += float(w)
    else:
        G.add_edge(i, j, weight=float(w))

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

# Static attention network (network_multiomic.pdf), one page per --network_features_per_omic:
# the top-N features per omic by attribution plus all samples, edges between them with the
# same per-edge-type 80th-percentile rule as network_interactive.html. The cutoffs come from
# the largest N, so every smaller page is a subgraph of the largest one; nodes left without
# an edge are dropped.
from pramigo_figures import draw_multiomic_network
_net_sizes = sorted(set(args.network_features_per_omic))
_node_type_of = np.concatenate([[name] * FEATURE_OMIC_SIZES[name] for name in OMIC_NAMES] + [['sample'] * sample_len])
_cond_of_node = np.concatenate([np.full(n_feature_nodes, ''), condition_labels])

def _network_selection(n_top):
    keep = np.zeros(n_total_nodes, dtype=bool)
    for _name in OMIC_NAMES:
        _sel = omic_node_attribution[_name].sort_values('Attribution', ascending=False).head(n_top)
        keep[_sel['node'].values] = True
    keep[n_feature_nodes:] = True
    m_ = keep[edge_src] & keep[edge_tgt] & (edge_src != edge_tgt)
    e_ = pd.DataFrame({'source': edge_src[m_], 'target': edge_tgt[m_], 'weight': edge_att_ratio[m_]})
    # undirected: a pair keeps its stronger direction
    e_[['source', 'target']] = np.sort(e_[['source', 'target']].values, axis=1)
    e_ = e_.groupby(['source', 'target'], as_index=False)['weight'].max()
    e_['category'] = [tuple(sorted((_node_type_of[a], _node_type_of[b]))) for a, b in zip(e_.source, e_.target)]
    return e_

_e_big = _network_selection(_net_sizes[-1])
_cutoffs = _e_big.groupby('category')['weight'].quantile(0.8).to_dict()
_lw_all = np.log(_e_big.loc[_e_big.weight >= _e_big.category.map(_cutoffs), 'weight'])
_lw_range = (float(_lw_all.min()), float(_lw_all.max())) if len(_lw_all) else None
with PdfPages(plots_dir + "network_multiomic.pdf") as _pdf:
    for _n in _net_sizes:
        _e = _network_selection(_n)
        _e = _e[_e.weight >= _e.category.map(_cutoffs)]
        _linked = np.unique(np.concatenate([_e.source.values, _e.target.values]))
        _nodes = pd.DataFrame({'node': _linked, 'label': full_node_names.values[_linked],
                               'type': _node_type_of[_linked], 'condition': _cond_of_node[_linked]})
        _fig = draw_multiomic_network(
            _nodes, _e, OMIC_NAMES, OMIC_COLORS, OMIC_MARKERS_BY_NAME, CONDITION_COLORS, top_n=_n,
            lw_range=_lw_range,
            footer_note=(f"Top {_n} features per omic by PRAMIGO attribution (Σ over classes of |attention × gradient|, "
                         "training samples); edges = HGT attention relative to uniform (α × in-degree, mean of heads "
                         f"and layers), kept when ≥ the 80th percentile of its edge type in the top-{_net_sizes[-1]} network. "
                         "Edge colour = omic (grey = between two omics); "
                         "edge width and opacity scale with log attention; node size scales with degree. Shaded areas: "
                         "modules (connected components). Labels: 5 features per omic and module with the highest "
                         "summed edge weight."))
        _pdf.savefig(_fig, bbox_inches="tight")
        plt.close(_fig)
        print(f"network_multiomic: top {_n} per omic -> {len(_nodes)} nodes, {len(_e)} edges")

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
    if i not in roc_auc_per_class:  # class absent (or the only class) in the evaluated samples
        continue
    fpr, tpr, _ = roc_curve(eval_one_hot[:, i], class_probs[:, i])
    auc_i = roc_auc_per_class.get(i, np.nan)
    ax.plot(fpr, tpr, color=OKABE_ITO[i % len(OKABE_ITO)], linewidth=1.8,
            label=f'{condition_names[i]} (AUC={auc_i:.2f})')
ax.plot([0, 1], [0, 1], color='#aaaaaa', linestyle='--', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'Per-Class ROC Curves - {eval_label}\n(macro AUC = {roc_auc:.2f})')
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
ax.set_title(f'Per-Class Classification Performance - {eval_label}')
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
    _rank_pct, _cum_pct = _cumulative_pct(omic_feature_dfs[_name]['Attribution'].values)
    ax.plot(_rank_pct, _cum_pct, color=OMIC_COLORS[_name], linewidth=2,
            label=f'{_name} (n={len(omic_feature_dfs[_name])})')
ax.plot([0, 100], [0, 100], color='#cccccc', linestyle=':', linewidth=1, label='Uniform (no concentration)')
ax.set_xlabel('Features Ranked by Attribution (% of that omic layer)')
ax.set_ylabel('Cumulative Attribution (%)')
ax.set_title('Attribution Concentration by Omic')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(supplementary_dir + "attention_concentration_by_omic.pdf")
plt.close(fig)

# Kruskal-Wallis across every omic when there are more than 2 (Mann-Whitney U is only
# defined for exactly 2 groups - the original script's fixed comparison).
_omic_attention_groups = [omic_feature_dfs[_name]['Attribution'].values for _name in OMIC_NAMES]
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
ax.set_ylabel('Attribution')
ax.set_title(f'Attribution Distribution by Omic Type\n{_stat_label}')
plt.tight_layout()
plt.savefig(supplementary_dir + "attention_distribution_by_omic.pdf")
plt.close(fig)
print("Supplementary: omic-wise attention concentration/distribution saved.")

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
ax.set_title('Top 20 Hub Nodes in the Attribution Network')
ax.legend(handles=[Patch(facecolor=OMIC_COLORS[_name], label=_name) for _name in OMIC_NAMES] +
                   [Patch(facecolor='#888888', label='Sample')], loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(supplementary_dir + "network_hub_nodes.pdf")
plt.close(fig)
print("Supplementary: hub-node figure saved.")

print("=== Supplementary visualizations complete ===")


