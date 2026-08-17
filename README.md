# PRAMIGO: A Heterogeneous Graph Transformer approach to target multi-omic integrated programs

<p align="center">
  <img src="docs/pramigopy.png" width="900">
</p>

PRAMIGO builds a heterogeneous graph over **samples, genes, and metabolites** (or any two omic layers), trains a Heterogeneous Graph Transformer (HGT) on it, and returns embeddings, attention-based interpretability, and cluster-level "programs" of co-regulated features — all from a single command.

---

## 📥 Installation

```bash
git clone https://github.com/iclemente99/PRAMIGO
cd PRAMIGO

curl -LsSf https://astral.sh/uv/install.sh | sh   # once, to install uv
uv venv hgt_env --python 3.8
source hgt_env/bin/activate                        # macOS / Linux
# hgt_env\Scripts\activate                          # Windows (PowerShell/cmd)
uv pip install -r env/biomixhgt_env.txt
```

That's it — no manual dependency wrangling beyond this.

---

## 📊 Input data

PRAMIGO expects three TSV files:

| File | Required columns / shape |
|---|---|
| `--metadata_path` | Samples as rows, with at least `ID` (sample identifier) and `CONDITION` (group label) |
| `--omic1_path` | Features × samples, `ID` column first (e.g. gene expression) |
| `--omic2_path` | Features × samples, `ID` column first (e.g. metabolomics) |

> ⚠️ **The omic matrices must already be filtered and normalized.** PRAMIGO does **not** perform any filtering, batch correction, or normalization of its own — it consumes the matrices as-is and builds the correlation graph directly from them. Feed it raw or poorly-normalized data and the feature-feature correlation graph (and everything downstream of it) will reflect that noise. A standard choice is per-feature (row-wise) z-scoring after your usual QC/filtering pipeline — see [Simulated toy data](#-simulated-toy-data) below for a worked example of exactly this format.

---

## 🚀 Running it

### From the command line

```bash
python src/biomix_hgt.py \
  --metadata_path "./data/EGAS00001001746/EGAS00001001746_metadata_CLL.tsv" \
  --omic1_path "./data/EGAS00001001746/EGAS00001001746_transcriptomics.tsv" \
  --omic2_path "./data/EGAS00001001746/EGAS00001001746_methylomics.tsv" \
  --result_dir "./data/EGA_biomix"
```

### From a Python script

`src/biomix_hgt.py` is a standalone script (not a package with an importable API), so the supported way to drive it from Python is the same way it's launched from the shell — via `subprocess`, pointed at whatever interpreter has the `hgt_env` dependencies installed:

```python
import subprocess, sys

subprocess.run([
    sys.executable, "src/biomix_hgt.py",
    "--metadata_path", "./data/EGAS00001001746/EGAS00001001746_metadata_CLL.tsv",
    "--omic1_path", "./data/EGAS00001001746/EGAS00001001746_transcriptomics.tsv",
    "--omic2_path", "./data/EGAS00001001746/EGAS00001001746_methylomics.tsv",
    "--result_dir", "./data/EGA_biomix",
], check=True)
```

This is useful for calling PRAMIGO as one step in a larger Python pipeline (e.g. looping over hyperparameters or datasets).

### Resuming a training run

Training checkpoints itself every epoch by default (model, optimizer, scheduler, RNG state, and full loss history), so a killed or time-limited run isn't wasted:

```bash
python src/biomix_hgt.py --resume \
  --metadata_path ... --omic1_path ... --omic2_path ... \
  --result_dir "./data/EGA_biomix" --epoch 200
```

`--resume` picks up from `<result_dir>/model/checkpoint_latest.pt` and continues until `--epoch`. If no checkpoint exists yet, it just starts fresh. If the checkpoint already reached `--epoch`, it exits immediately rather than doing wasted work.

### Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `--epoch` | 100 | Number of training epochs |
| `--n_batch` | 25 | Number of sub-sampled graphs per epoch |
| `--n_hid` | 128 | HGT hidden dimension |
| `--ae_n_hid` | 256 | Autoencoder hidden dimension (feature reduction step) |
| `--n_heads` | 8 | Attention heads |
| `--n_layers` | 2 | Number of HGT layers |
| `--lr` | 0.0001 | Learning rate |
| `--omic1_corr_cutoff`, `--omic2_corr_cutoff` | 0.7 | Correlation threshold for building each omic's feature-feature graph edges |
| `--leiden_resolution` | 1.5 | Resolution for Leiden clustering of the attention-weighted feature subgraph |
| `--leiden_min_cluster_size` | 3 | Minimum cluster size to keep in the cluster-network plots |
| `--cuda` | 1 | **`0` runs on GPU 0; any other value runs on CPU** |
| `--resume` | off | Resume from `<result_dir>/model/checkpoint_latest.pt` |
| `--checkpoint_every` | 1 | Save a resumable checkpoint every N epochs |

Run `python src/biomix_hgt.py --help` for the complete list, including reduction method, dropout, optimizer choice, and regularization weight.

---

## 📁 Outputs

Everything is written under `--result_dir`, organized into subfolders:

| Folder | Contents |
|---|---|
| `model/` | `checkpoint_latest.pt` (resumable training state) and the final trained model |
| `loss/` | Per-epoch CSVs: total, validation, cosine, and cross-entropy loss history |
| `gene/`, `metabo/`, `sample/` | Learned embedding matrices for each node type |
| `embeddings/` | Classifier-head embeddings at each hidden layer (128 / 64 / 32-dim) plus final class embeddings |
| `att/` | Per-node attention scores as CSV — the main interpretability output |
| `plots/` | Core visualizations (see below), always generated |
| `supplementary/` | Additional diagnostic plots, always generated |

**Core visualizations (`plots/`):**
- `training_loss_components.pdf` — total/validation loss plus loss components over training
- `Confusion_matrix_figure.pdf` — classification performance on the held-out test split
- `Interpretability_nodes_attention_figure.pdf` — top attended genes/metabolites
- `embeddings_evaluations_figure.pdf` — embedding quality diagnostics
- `network_interactive.html` — interactive, browsable version of the integrated graph
- `leiden_clusters_composition.csv`, `leiden_cluster_network_nature_style.pdf`, `leiden_cluster_subgraphs.pdf`, `leiden_program_activity_scores.csv`, `leiden_program_activity_report.pdf` — feature "programs" found by Leiden clustering of the attention subgraph. These only appear if at least one cluster meets `--leiden_min_cluster_size`; on very small graphs it's normal for no cluster to form.

**Supplementary plots (`supplementary/`):** learning-rate schedule, loss-component share over time, ROC curves, per-class performance, attention concentration/distribution by omic layer, embedding separability (silhouette) and PCA scree, plus static publication-style and hub-node network figures.

---

## 🧪 Simulated toy data

`src/simulate_data.py` generates a small synthetic dataset for quickly exercising the whole pipeline without needing real data — useful for smoke-testing an install or a code change in seconds rather than minutes:

```bash
python src/simulate_data.py
```

This writes to `data/simulated_toy/`:
- **9 samples** across **3 condition groups** (3 patients each)
- **2 omic layers, 20 features each** — 10 differentially-expressed (DE) marker features per omic (each tied to exactly one group via a fixed mean shift) and 10 pure-noise features
- Every matrix is written **already row-wise (per-feature) z-scored** — exactly the normalized format `biomix_hgt.py` expects (see [Input data](#-input-data) above)
- `simulated_ground_truth.tsv` records which features are DE and which group they mark, so downstream attention/interpretability output can be checked against a known answer

Run PRAMIGO on it directly:

```bash
python src/biomix_hgt.py \
  --metadata_path data/simulated_toy/simulated_metadata.tsv \
  --omic1_path data/simulated_toy/simulated_omic1.tsv \
  --omic2_path data/simulated_toy/simulated_omic2.tsv \
  --result_dir data/simulated_toy_result \
  --epoch 4 --n_batch 2 --n_hid 8 --ae_n_hid 8 --n_heads 2 --n_layers 2
```

### Test suite

`src/test_biomix_pipeline.py` is an end-to-end integration test built on this simulated data:

```bash
pytest src/test_biomix_pipeline.py -v -s
```

It provisions a fresh `uv` environment from `env/biomixhgt_env.txt`, then runs the full pipeline — loading, training (including resume-from-checkpoint), output saving, and visualization — for both a 2-group and a 3-group condition setup, and checks that every expected file above gets created. It's a slow test (a few minutes even on toy-sized data), so it's meant for validating a setup or a code change, not for a tight edit/test loop. It uses pytest's `tmp_path`/`tmp_path_factory` fixtures for its result directories, so outputs from a plain `pytest` run land in your OS temp folder, not the repo — run the direct command above (or pass `--result_dir` to your own invocation) if you want to keep and inspect the plots.

---

## ✍️ Citation & Acknowledgements

This work was developed at LBAI-UBO. Please cite accordingly if used in academic research.

## 🖥️ Maintainers

Iñigo Clemente Larramendi — inigo.clementelarramendi@univ-brest.fr
