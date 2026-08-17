"""
Generates a tiny synthetic multi-omic dataset for exercising PRAMIGO's
biomix_hgt.py pipeline end-to-end without needing real data.

Design: 3 condition groups x 3 patients each (9 samples total), 2 omic layers
with 20 features each. Within each omic layer, 10 features are simulated as
differentially expressed (DE) markers of one specific group (a clear, fixed
mean shift in that group's samples only), and the other 10 are pure noise
with no group signal. Every matrix is written already row-wise (per-feature)
z-scored, matching the "filtered and normalized" input format biomix_hgt.py
expects (it does not normalize omic matrices itself).

Outputs (data/simulated_toy/):
  simulated_metadata.tsv       - ID, CONDITION (9 rows)
  simulated_omic1.tsv          - ID + 9 sample columns (20 rows)
  simulated_omic2.tsv          - ID + 9 sample columns (20 rows)
  simulated_ground_truth.tsv   - which features are DE and of which group,
                                 for validating downstream interpretability
                                 (attention/logFC direction) against a known answer.
"""
import os
import numpy as np
import pandas as pd

SEED = 0
N_GROUPS = 3
N_PER_GROUP = 3
N_FEATURES_PER_OMIC = 20
N_DE_PER_OMIC = 10
GROUP_EFFECT_SIZE = 2.5  # fixed mean shift applied to a marker feature's own group
NOISE_SD = 1.0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'simulated_toy')


def simulate_omic(rng, n_features, n_de, group_names, sample_groups, omic_name):
    n_samples = len(sample_groups)
    raw = rng.normal(loc=0.0, scale=NOISE_SD, size=(n_features, n_samples))

    # Each DE feature is a marker of exactly one group (round-robin assignment), with a
    # clear fixed upward shift in that group's samples only - an unambiguous ground
    # truth for later validation (e.g. "is this feature correctly identified as up in
    # GroupA"), as opposed to noisy per-group effects that could be ambiguous by chance.
    marker_group = [group_names[i % len(group_names)] for i in range(n_de)]
    for feat_idx, group in enumerate(marker_group):
        cols = [j for j, g in enumerate(sample_groups) if g == group]
        raw[feat_idx, cols] += GROUP_EFFECT_SIZE

    # Row-wise (per-feature) z-score: a standard normalized representation, and exactly
    # what biomix_hgt.py expects to receive already done.
    normalized = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-8)

    feature_ids = [f"{omic_name}_DE_{i+1}" if i < n_de else f"{omic_name}_noise_{i - n_de + 1}"
                   for i in range(n_features)]
    marker_of = [marker_group[i] if i < n_de else None for i in range(n_features)]
    is_de = [i < n_de for i in range(n_features)]
    return feature_ids, is_de, marker_of, normalized


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    group_names = [f"Group{chr(65 + g)}" for g in range(N_GROUPS)]  # GroupA, GroupB, GroupC
    sample_ids = [f"S{i + 1}" for i in range(N_GROUPS * N_PER_GROUP)]
    sample_groups = [group_names[g] for g in range(N_GROUPS) for _ in range(N_PER_GROUP)]

    metadata = pd.DataFrame({"ID": sample_ids, "CONDITION": sample_groups})
    metadata_path = os.path.join(OUT_DIR, "simulated_metadata.tsv")
    metadata.to_csv(metadata_path, sep="\t", index=False)

    ground_truth_rows = []
    for omic_name, omic_filename in [("omic1", "simulated_omic1.tsv"), ("omic2", "simulated_omic2.tsv")]:
        feature_ids, is_de, marker_of, matrix = simulate_omic(
            rng, N_FEATURES_PER_OMIC, N_DE_PER_OMIC, group_names, sample_groups, omic_name
        )
        df = pd.DataFrame(matrix, columns=sample_ids)
        df.insert(0, "ID", feature_ids)
        df.to_csv(os.path.join(OUT_DIR, omic_filename), sep="\t", index=False)
        ground_truth_rows += [
            {"Omic": omic_name, "Feature": fid, "Is_Differentially_Expressed": de, "Marker_Of_Group": mk}
            for fid, de, mk in zip(feature_ids, is_de, marker_of)
        ]

    ground_truth_path = os.path.join(OUT_DIR, "simulated_ground_truth.tsv")
    pd.DataFrame(ground_truth_rows).to_csv(ground_truth_path, sep="\t", index=False)

    print(f"Simulated dataset written to {os.path.abspath(OUT_DIR)}")
    print(f"  Metadata: {metadata_path}")
    print(f"  {len(sample_ids)} samples across {N_GROUPS} groups ({N_PER_GROUP} each): {group_names}")
    print(f"  2 omics x {N_FEATURES_PER_OMIC} features ({N_DE_PER_OMIC} DE / {N_FEATURES_PER_OMIC - N_DE_PER_OMIC} noise each)")
    print(f"  Ground truth: {ground_truth_path}")


if __name__ == "__main__":
    main()
