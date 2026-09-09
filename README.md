# PRAMIGO: A Heterogeneous Graph Transformer approach to target multi-omic integrated programs

<p align="center">
  <img src="docs/pramigopy.png" width="900">
</p>

PRAMIGO builds a heterogeneous graph over **samples and an arbitrary number of omic layers**, trains a Heterogeneous Graph Transformer (HGT) on it, and returns embeddings, attention-based interpretability, and cluster-level "programs" of co-regulated features — all from a single command.

> This version of **PRAMIGO** generalizes the pipeline from a fixed 2-omic CLI (`--omic1_path`/`--omic2_path`) to an arbitrary-length list of omics, driven by a small YAML/JSON manifest (`--omics_manifest`). The original 2-omic flags still work unchanged (see [Input data](#-input-data) below) — nothing about the 2-omic path changes for existing users.

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

PRAMIGO_gen expects a metadata TSV plus **either** an `--omics_manifest` (any number of omics) **or** the legacy `--omic1_path`/`--omic2_path` pair (exactly 2 omics).

| File | Required columns / shape |
|---|---|
| `--metadata_path` | Samples as rows, with at least `ID` (sample identifier) and `CONDITION` (group label) |
| each omic file | Features × samples, `ID` column first (e.g. gene expression, metabolomics, methylomics, ...) |

> ⚠️ **The omic matrices must already be filtered and normalized.** PRAMIGO does **not** perform any filtering, batch correction, or normalization of its own — it consumes the matrices as-is and builds the correlation graph directly from them. Feed it raw or poorly-normalized data and the feature-feature correlation graph (and everything downstream of it) will reflect that noise. A standard choice is per-feature (row-wise) z-scoring after your usual QC/filtering pipeline — see [Simulated toy data](#-simulated-toy-data) below for a worked example of exactly this format. 

> As a general recommendation, we suggest using a **row-wise normalized matrix restricted to the significantly different omic features** identified between groups within each modality.

### The omics manifest (`--omics_manifest`)

For anything beyond 2 omics, point `--omics_manifest` at a YAML or JSON file listing however many omic layers your run needs instead of adding more `--omicN_path`-style flags:

```yaml
topology: full              # 'full' (default) or 'star' - see Topology below
sample_mode: intersection   # 'intersection' (default) or 'union' - see Sample coverage below
omics:
  - name: transcriptomics
    path: ./data/rnaseq.tsv
    rate: 1.0                # optional, defaults to 1.0 - fraction of features sampled per training batch
    corr_cutoff: 0.7          # optional, defaults to 0.7 - feature-feature correlation threshold for this omic
    kind: transcriptomics     # optional modality tag, defaults to `name` - see "Same omic twice" below
  - name: methylomics
    path: ./data/methyl.tsv
    corr_cutoff: 0.6
  - name: metabolomics
    path: ./data/metab.tsv
```

`name` must be unique across the manifest — it becomes that omic's node type, output subfolder (`<result_dir>/<name>/`), and plot legend label, so keep it short and filesystem-safe. `topology` and `sample_mode` can also be set at the top level of the manifest as shown, or overridden from the command line with `--topology`/`--sample_mode` (the CLI flag always wins if both are given).

If `--omics_manifest` is omitted, `--omic1_path`/`--omic2_path` (plus their `--omic1_rate`/`--omic2_corr_cutoff`/etc. siblings) are automatically wrapped into an equivalent 2-entry manifest, so every existing 2-omic command still works unchanged.

### Topology: `full` vs. `star` (`--topology`)

Controls whether omic layers connect **directly to each other**, in addition to connecting through sample nodes:

- **`full` (default, matches original PRAMIGO behavior):** every pair of omics gets its own direct feature-feature correlation graph (a specific gene can be wired straight to a specific metabolite). Cost grows as O(k²) in the number of omics — fine for 2-4 omics, worth watching beyond that.
- **`star`:** omics only ever connect through sample nodes, never directly to each other. Cost is O(k) in the number of omics, at the price of losing that direct cross-omic co-regulation edge.

```bash
python src/biomix_hgt.py --omics_manifest manifest.yaml --metadata_path ... --result_dir ... --topology star
```

### Sample coverage: `intersection` vs. `union` (`--sample_mode`)

Controls which samples are kept when not every sample has data in every omic:

- **`intersection` (default, matches original PRAMIGO behavior):** complete-case — only samples present in the metadata *and every single omic* survive. With many omics, this can shrink the usable cohort fast (every extra omic multiplies the chance a sample is missing at least one layer).
- **`union`:** keep any sample present in the metadata and *at least one* omic. A sample missing a given omic's data gets that omic's feature block zero-filled (so every omic matrix still has one column per kept sample) and, critically, gets **no graph edges** of that omic-sample relation type for that sample — the missingness is carried structurally by the graph (no edges), not by an imputed value pretending to be real data.

```bash
python src/biomix_hgt.py --omics_manifest manifest.yaml --metadata_path ... --result_dir ... --sample_mode union
```

### Same omic twice (`kind`)

Tagging two manifest entries with the same `kind` (e.g. two transcriptomics files from different platforms/cohorts of the *same* patients) groups them in reports/plots without merging them into one feature space — each is still its own node type/file. This is safe for the same-cohort, two-platform case. Merging **different cohorts'** data of the same modality is a separate multi-cohort/batch-effect problem (differing sample sets, potential feature-ID mismatches, and a real risk of the graph picking up cohort identity instead of biology) that PRAMIGO_gen does not attempt to solve automatically — do any needed harmonization/batch-correction upstream before feeding the files in.

---

## 🚀 Running it

### From the command line

Legacy 2-omic flags (unchanged from PRAMIGO):

```bash
python src/biomix_hgt.py \
  --metadata_path "./data/EGAS00001001746/EGAS00001001746_metadata_CLL.tsv" \
  --omic1_path "./data/EGAS00001001746/EGAS00001001746_transcriptomics.tsv" \
  --omic2_path "./data/EGAS00001001746/EGAS00001001746_methylomics.tsv" \
  --result_dir "./data/EGA_biomix"
```

Or with a manifest (any number of omics — see [The omics manifest](#the-omics-manifest---omics_manifest) above):

```bash
python src/biomix_hgt.py \
  --metadata_path "./data/EGAS00001001746/EGAS00001001746_metadata_CLL.tsv" \
  --omics_manifest "./data/EGAS00001001746/manifest.yaml" \
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
| `--omics_manifest` | none | YAML/JSON manifest listing an arbitrary number of omics (takes precedence over `--omic1_path`/`--omic2_path`) |
| `--topology` | `full` | `full` (every omic pair gets direct edges) or `star` (omics only connect through samples) |
| `--sample_mode` | `intersection` | `intersection` (complete-case) or `union` (partial coverage, missing omics get no edges) |
| `--omic1_corr_cutoff`, `--omic2_corr_cutoff` | 0.7 | [legacy 2-omic mode] Correlation threshold for building each omic's feature-feature graph edges — use `corr_cutoff:` per-omic in `--omics_manifest` instead |
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
| `<omic_name>/` (one per omic, named after its manifest `name` — `omic1/`/`omic2/` in legacy 2-omic mode), `sample/` | Learned embedding matrices for each node type |
| `embeddings/` | Classifier-head embeddings at each hidden layer (128 / 64 / 32-dim) plus final class embeddings |
| `att/` | Per-node attention scores as CSV — the main interpretability output |
| `plots/` | Core visualizations (see below), always generated |
| `supplementary/` | Additional diagnostic plots, always generated |

**Core visualizations (`plots/`):**
- `training_loss_components.pdf` — total/validation loss plus loss components over training
- `Confusion_matrix_figure.pdf` — classification performance on the held-out test split
- `Interpretability_nodes_attention_figure.pdf` — top attended features per omic
- `embeddings_evaluations_figure.pdf` — embedding quality diagnostics
- `network_interactive.html` — interactive, browsable version of the integrated graph
- `leiden_clusters_composition.csv`, `leiden_cluster_network.pdf`, `leiden_cluster_subgraphs.pdf`, `leiden_program_activity_scores.csv`, `leiden_program_activity_report.pdf` — feature "programs" found by Leiden clustering of the attention subgraph. These only appear if at least one cluster meets `--leiden_min_cluster_size`; on very small graphs it's normal for no cluster to form.

**Supplementary plots (`supplementary/`):** learning-rate schedule, loss-component share over time, ROC curves, per-class performance, attention concentration/distribution by omic layer, embedding separability (silhouette) and PCA scree, plus a hub-node network figure.

### About `leiden_cluster_network.pdf`

**Input:** the *contracted* program graph — one node per Leiden-derived multi-omic program (a cluster of co-attended genes/metabolites), sized by how many features it contains, with edges weighted by the total attention linking one program's members to another's (the exact same numbers also sit in `leiden_clusters_composition.csv` and `supplementary/program_connectivity_heatmap.pdf`).

**Goal:** give a quick, at-a-glance map of how many distinct multi-omic programs were found, their relative sizes, and how strongly they relate to one another — a map of *programs*, not of individual features (that's what `leiden_cluster_subgraphs.pdf` is for).

> ⚠️ **Please look carefully at this figure before relying on it.** Network layouts in Python (Kamada-Kawai/spring-based, as used here) are quite sensitive to the degree and weight distribution of the underlying graph — a small number of programs with widely varying connection strengths can legitimately produce a layout with a few long, stretched-out edges even when the underlying attention data is perfectly valid. If the plot ever looks off to you, please don't take it at face value: cross-check it against the underlying numbers yourself, namely `leiden_clusters_composition.csv` (which features are in which program) and `supplementary/program_connectivity_heatmap.pdf` (the exact attention weight between every pair of programs, as a matrix — no 2D layout involved). We're continuing to look for a better default here, so feedback on this specific plot is very welcome.

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

This work was developed through the BiomiX consortium (https://github.com/IxI-97/BiomiX). Please cite accordingly if used in academic research.

## 🖥️ Maintainers

Iñigo Clemente Larramendi — inigo.clementelarramendi@univ-brest.fr
