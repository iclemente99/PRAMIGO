"""
Extended end-to-end integration tests for PRAMIGO's N-omic generalization:
`--omics_manifest`, `--topology full|star`, and `--sample_mode
intersection|union`. This complements test_biomix_pipeline.py, which only
exercises the legacy 2-omic `--omic1_path`/`--omic2_path` CLI surface.

Reuses test_biomix_pipeline.py's `provisioned_python` fixture (so the same
README-documented venv is provisioned once per pytest session and shared
across both files) plus its constants/helpers, and simulate_data.py's
`simulate_omic()` to generate extra synthetic omics on top of the real
simulated_toy omic1/omic2 - written into pytest tmp dirs, never into the
repo's data/ folder.

Two kinds of tests live here:
  * Fast, subprocess-free unit tests of omics_manifest.py itself
    (TestOmicsManifestParsing, TestManifestYamlAndErrorHandling) - run in a
    fraction of a second, no venv/provisioning needed.
  * Slow end-to-end tests that actually run biomix_hgt.py as a subprocess
    (three/four omics, --topology star, --sample_mode union vs intersection,
    two omics sharing the same `kind`) - same "slow integration test" caveat
    as test_biomix_pipeline.py.

Run manually with:
    pytest src/test_multi_biomix_pipeline.py -v -s
or together with the base suite:
    pytest src/test_biomix_pipeline.py src/test_multi_biomix_pipeline.py -v -s

Requires the simulated data to exist first: `python src/simulate_data.py`.
Requires `uv` on PATH for the subprocess-based tests; they skip (rather than
fail) if it isn't available, exactly like test_biomix_pipeline.py.
"""
import os
import sys
import json
import itertools
import subprocess
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from simulate_data import simulate_omic
import omics_manifest
from omics_manifest import ManifestError

from test_biomix_pipeline import (
    provisioned_python,
    run_biomix,
    _assert_success,
    SRC_DIR,
    BIOMIX_SCRIPT,
    METADATA_3GROUP,
    OMIC1_PATH,
    OMIC2_PATH,
    N_SAMPLES,
    N_FEATURES_PER_OMIC,
    N_DE_PER_OMIC,
    FAST_HPARAMS,
)


# ---------------------------------------------------------------------------
# Helpers shared by every test class below.
# ---------------------------------------------------------------------------
def _read_metadata_groups(metadata_path=METADATA_3GROUP):
    metadata = pd.read_csv(metadata_path, sep="\t")
    sample_ids = metadata["ID"].tolist()
    sample_groups = metadata["CONDITION"].tolist()
    group_names = metadata["CONDITION"].unique().tolist()
    return sample_ids, sample_groups, group_names


def generate_synthetic_omic(out_dir, omic_name, seed, drop_sample_ids=None,
                             n_features=N_FEATURES_PER_OMIC, n_de=N_DE_PER_OMIC):
    """Writes one more synthetic omic TSV using simulate_data.py's own
    simulate_omic() (same DE-marker/noise recipe, same 9-sample/3-group
    structure as the real omic1/omic2), for scenarios that need more than the
    2 pre-generated omics. Deterministic via `seed`. `drop_sample_ids`
    optionally removes sample columns entirely, to simulate a sample this omic
    was never measured for (the fixture --sample_mode union/intersection is
    tested against)."""
    sample_ids, sample_groups, group_names = _read_metadata_groups()
    rng = np.random.default_rng(seed)
    feature_ids, is_de, marker_of, matrix = simulate_omic(
        rng, n_features, n_de, group_names, sample_groups, omic_name
    )
    df = pd.DataFrame(matrix, columns=sample_ids)
    df.insert(0, "ID", feature_ids)
    if drop_sample_ids:
        df = df.drop(columns=list(drop_sample_ids))
    out_path = os.path.join(str(out_dir), f"simulated_{omic_name}.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def omic_entry(name, path, corr_cutoff=0.3, rate=1.0, kind=None):
    """One `omics:` list entry for a manifest dict. corr_cutoff defaults to the
    same relaxed 0.3 FAST_HPARAMS uses for the legacy
    --omic1_corr_cutoff/--omic2_corr_cutoff, so tiny toy-sized correlation
    graphs don't end up nearly disconnected."""
    entry = {"name": name, "path": str(path), "corr_cutoff": corr_cutoff, "rate": rate}
    if kind is not None:
        entry["kind"] = kind
    return entry


def write_manifest(path, omics, topology=None, sample_mode=None):
    manifest = {"omics": omics}
    if topology is not None:
        manifest["topology"] = topology
    if sample_mode is not None:
        manifest["sample_mode"] = sample_mode
    with open(path, "w") as fh:
        json.dump(manifest, fh)
    return str(path)


def run_biomix_manifest(python_executable, metadata_path, manifest_path, result_dir, extra_args=None):
    """Same convention as test_biomix_pipeline.run_biomix(), but drives
    biomix_hgt.py through --omics_manifest instead of the legacy
    --omic1_path/--omic2_path pair. FAST_HPARAMS's --omic1_corr_cutoff/
    --omic2_corr_cutoff are harmless no-ops here (biomix_hgt.py ignores them
    whenever --omics_manifest is given - see manifest_from_legacy_args())."""
    cmd = [
        str(python_executable), BIOMIX_SCRIPT,
        "--metadata_path", str(metadata_path),
        "--omics_manifest", str(manifest_path),
        "--result_dir", str(result_dir),
    ] + FAST_HPARAMS + (extra_args or [])
    return subprocess.run(cmd, cwd=SRC_DIR, capture_output=True, text=True, timeout=900)


def _pairwise_relation_keys(omic_names):
    """Every cross-omic relation name --topology full is expected to create
    edges for (utils.relation_name(a, b) == f'{a}_{b}'), generated in the same
    order build_graph_generic() iterates itertools.combinations(omic_names, 2)
    - i.e. the same order the omics appear in the manifest."""
    return [f"{a}_{b}" for a, b in itertools.combinations(omic_names, 2)]


def _read_embedding_matrix(result_dir, subfolder):
    folder = os.path.join(str(result_dir), subfolder)
    files = os.listdir(folder)
    assert len(files) == 1, f"Expected exactly one embedding file in {subfolder}/, found {files}"
    return pd.read_csv(os.path.join(folder, files[0]), sep=" ", header=None)


def _args(**overrides):
    """Minimal stand-in for argparse.Namespace, matching biomix_hgt.py's actual
    argparse defaults for every attribute omics_manifest.py reads."""
    base = dict(
        omics_manifest=None, topology=None, sample_mode=None,
        omic1_path=None, omic2_path=None,
        omic1_rate=1.0, omic2_rate=1.0,
        omic1_corr_cutoff=0.7, omic2_corr_cutoff=0.7,
    )
    base.update(overrides)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# Fast, subprocess-free checks of omics_manifest.py itself - the piece that
# replaced PRAMIGO's per-omic-index CLI surface (--omic1_path, --omic2_path,
# ... and a hand-pasted num_types/num_relations) with a manifest driven by an
# arbitrary-length `omics:` list. No provisioned env or real data needed.
# ---------------------------------------------------------------------------
class TestOmicsManifestParsing:

    def test_manifest_accepts_three_omics(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "m.json", [
            omic_entry("a", "a.tsv"), omic_entry("b", "b.tsv"), omic_entry("c", "c.tsv"),
        ])
        omics, topology, sample_mode = omics_manifest.resolve_manifest(_args(omics_manifest=manifest_path))
        assert [o["name"] for o in omics] == ["a", "b", "c"]
        assert topology == "full"
        assert sample_mode == "intersection"

    def test_manifest_accepts_an_arbitrary_number_of_omics(self, tmp_path):
        """Nothing caps the omic count at any fixed number of 'slots' - the
        whole point of the manifest replacing a fixed CLI surface."""
        names = [f"omic{i}" for i in range(7)]
        manifest_path = write_manifest(tmp_path / "m.json", [omic_entry(n, f"{n}.tsv") for n in names])
        omics, _, _ = omics_manifest.resolve_manifest(_args(omics_manifest=manifest_path))
        assert [o["name"] for o in omics] == names

    def test_manifest_rejects_fewer_than_two_omics(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "m.json", [omic_entry("a", "a.tsv")])
        with pytest.raises(ManifestError):
            omics_manifest.resolve_manifest(_args(omics_manifest=manifest_path))

    def test_manifest_rejects_duplicate_names(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "m.json", [
            omic_entry("a", "a.tsv"), omic_entry("a", "b.tsv"),
        ])
        with pytest.raises(ManifestError):
            omics_manifest.load_manifest(manifest_path)

    def test_manifest_rejects_reserved_name(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "m.json", [
            omic_entry("sample", "a.tsv"), omic_entry("b", "b.tsv"),
        ])
        with pytest.raises(ManifestError):
            omics_manifest.load_manifest(manifest_path)

    def test_omic_defaults_applied_when_rate_and_corr_cutoff_omitted(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "m.json", [
            {"name": "a", "path": "a.tsv"}, {"name": "b", "path": "b.tsv"},
        ])
        manifest = omics_manifest.load_manifest(manifest_path)
        for o in manifest["omics"]:
            assert o["rate"] == omics_manifest.DEFAULT_RATE
            assert o["corr_cutoff"] == omics_manifest.DEFAULT_CORR_CUTOFF

    def test_kind_defaults_to_name_when_omitted(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "m.json", [
            omic_entry("rnaseq", "a.tsv"), omic_entry("b", "b.tsv"),
        ])
        manifest = omics_manifest.load_manifest(manifest_path)
        assert manifest["omics"][0]["kind"] == "rnaseq"

    def test_kind_can_be_shared_by_two_omics(self, tmp_path):
        """'Same omic twice' (same-cohort, two platforms/batches of one
        modality): two entries tagged with the same `kind` stay two
        independent entries in the manifest - `kind` is a grouping label
        only, parsing never merges them into one."""
        manifest_path = write_manifest(tmp_path / "m.json", [
            omic_entry("rnaseq_batch1", "a.tsv", kind="transcriptomics"),
            omic_entry("rnaseq_batch2", "b.tsv", kind="transcriptomics"),
        ])
        manifest = omics_manifest.load_manifest(manifest_path)
        assert [o["name"] for o in manifest["omics"]] == ["rnaseq_batch1", "rnaseq_batch2"]
        assert [o["kind"] for o in manifest["omics"]] == ["transcriptomics", "transcriptomics"]

    def test_cli_topology_and_sample_mode_override_manifest(self, tmp_path):
        manifest_path = write_manifest(
            tmp_path / "m.json",
            [omic_entry("a", "a.tsv"), omic_entry("b", "b.tsv")],
            topology="full", sample_mode="intersection",
        )
        omics, topology, sample_mode = omics_manifest.resolve_manifest(
            _args(omics_manifest=manifest_path, topology="star", sample_mode="union")
        )
        assert topology == "star"
        assert sample_mode == "union"

    def test_manifest_topology_and_sample_mode_used_when_cli_not_set(self, tmp_path):
        manifest_path = write_manifest(
            tmp_path / "m.json",
            [omic_entry("a", "a.tsv"), omic_entry("b", "b.tsv")],
            topology="star", sample_mode="union",
        )
        omics, topology, sample_mode = omics_manifest.resolve_manifest(_args(omics_manifest=manifest_path))
        assert topology == "star"
        assert sample_mode == "union"

    def test_defaults_are_full_and_intersection_when_nothing_set(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "m.json", [omic_entry("a", "a.tsv"), omic_entry("b", "b.tsv")])
        _, topology, sample_mode = omics_manifest.resolve_manifest(_args(omics_manifest=manifest_path))
        assert topology == "full"
        assert sample_mode == "intersection"

    def test_legacy_two_omic_args_wrapped_into_equivalent_manifest(self):
        omics, topology, sample_mode = omics_manifest.resolve_manifest(
            _args(omic1_path="a.tsv", omic2_path="b.tsv")
        )
        assert [o["name"] for o in omics] == ["omic1", "omic2"]
        assert topology == "full"
        assert sample_mode == "intersection"

    def test_legacy_args_require_both_omic_paths(self):
        with pytest.raises(ManifestError):
            omics_manifest.resolve_manifest(_args(omic1_path="a.tsv", omic2_path=None))
        with pytest.raises(ManifestError):
            omics_manifest.resolve_manifest(_args(omic1_path=None, omic2_path=None))


class TestManifestYamlAndErrorHandling:

    def test_yaml_manifest_parses_when_pyyaml_available(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        manifest_dict = {
            "topology": "star",
            "omics": [
                {"name": "a", "path": "a.tsv", "corr_cutoff": 0.5},
                {"name": "b", "path": "b.tsv"},
            ],
        }
        yaml_path = tmp_path / "manifest.yaml"
        yaml_path.write_text(yaml.safe_dump(manifest_dict))
        manifest = omics_manifest.load_manifest(str(yaml_path))
        assert manifest["topology"] == "star"
        assert [o["name"] for o in manifest["omics"]] == ["a", "b"]
        assert manifest["omics"][0]["corr_cutoff"] == 0.5

    def test_yaml_manifest_without_pyyaml_gives_actionable_error(self, tmp_path, monkeypatch):
        """Forces the ImportError branch regardless of whether PyYAML happens
        to be installed in whatever interpreter runs this test - the
        provisioned venv built exactly per README.md/env/biomixhgt_env.txt
        does NOT include PyYAML, so real users hitting this is an expected
        case, not a hypothetical one."""
        monkeypatch.setitem(sys.modules, "yaml", None)
        yaml_path = tmp_path / "manifest.yaml"
        yaml_path.write_text("omics:\n  - name: a\n    path: a.tsv\n  - name: b\n    path: b.tsv\n")
        with pytest.raises(ManifestError, match="PyYAML"):
            omics_manifest.load_manifest(str(yaml_path))

    def test_missing_manifest_file_gives_clear_error(self, tmp_path):
        with pytest.raises(ManifestError):
            omics_manifest.load_manifest(str(tmp_path / "does_not_exist.json"))


# ---------------------------------------------------------------------------
# Synthetic data shared by the subprocess-based tests below: reuse the real
# simulated_toy omic1/omic2 (same 9 samples / 3 groups as
# test_biomix_pipeline.py) and add extra synthetic omics on top, generated
# fresh into a session-scoped tmp dir so nothing pollutes the repo's data/
# folder.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def extra_omics_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("extra_omics")


@pytest.fixture(scope="session")
def omic3_path(extra_omics_dir):
    return generate_synthetic_omic(extra_omics_dir, "omic3", seed=100)


@pytest.fixture(scope="session")
def omic4_path(extra_omics_dir):
    return generate_synthetic_omic(extra_omics_dir, "omic4", seed=200)


@pytest.fixture(scope="session")
def omic3_missing_one_sample_path(extra_omics_dir):
    """Same recipe as omic3_path but missing sample S9's column entirely - the
    fixture the --sample_mode intersection-vs-union comparison is built on."""
    return generate_synthetic_omic(extra_omics_dir, "omic3_partial", seed=100, drop_sample_ids=["S9"])


@pytest.fixture(scope="session")
def duplicate_kind_omic_path(extra_omics_dir):
    """A second, independently-generated 'transcriptomics-like' omic (own
    features, own noise draw) - tagged with the same `kind` as omic1 in
    TestTwoOmicsSameKind to exercise the same-cohort, two-platform 'same omic
    twice' case."""
    return generate_synthetic_omic(extra_omics_dir, "omic1b", seed=300)


THREE_OMIC_NAMES = ["omic1", "omic2", "omic3"]
FOUR_OMIC_NAMES = ["omic1", "omic2", "omic3", "omic4"]


# ---------------------------------------------------------------------------
# Three omics, default (full) topology.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def three_omics_full_run(provisioned_python, tmp_path_factory, omic3_path):
    result_dir = tmp_path_factory.mktemp("three_omics_full_result")
    manifest_path = write_manifest(
        tmp_path_factory.mktemp("three_omics_full_manifest") / "manifest.json",
        [omic_entry("omic1", OMIC1_PATH), omic_entry("omic2", OMIC2_PATH), omic_entry("omic3", omic3_path)],
        topology="full",
    )
    proc = run_biomix_manifest(provisioned_python, METADATA_3GROUP, manifest_path, result_dir)
    return proc, result_dir


class TestThreeOmicsFullTopology:

    def test_process_completes_successfully(self, three_omics_full_run):
        proc, _ = three_omics_full_run
        _assert_success(proc)

    def test_reports_three_omics_and_full_topology(self, three_omics_full_run):
        proc, _ = three_omics_full_run
        assert "Omics (3): ['omic1', 'omic2', 'omic3']" in proc.stdout
        assert "topology=full" in proc.stdout

    def test_per_omic_output_dirs_named_after_manifest_not_gene_metabo(self, three_omics_full_run):
        """The original PRAMIGO hardcoded 'gene'/'metabo' output folders;
        PRAMIGO's generalized script names one folder per omic after its
        manifest `name` instead."""
        _, result_dir = three_omics_full_run
        for name in THREE_OMIC_NAMES:
            assert os.path.isdir(os.path.join(result_dir, name)), f"Missing output dir for omic '{name}'"
        assert not os.path.isdir(os.path.join(result_dir, "gene"))
        assert not os.path.isdir(os.path.join(result_dir, "metabo"))

    def test_embeddings_saved_with_expected_shapes_for_every_omic(self, three_omics_full_run):
        _, result_dir = three_omics_full_run
        for name in THREE_OMIC_NAMES:
            matrix = _read_embedding_matrix(result_dir, name)
            assert matrix.shape[0] == N_FEATURES_PER_OMIC
        sample_matrix = _read_embedding_matrix(result_dir, "sample")
        assert sample_matrix.shape[0] == N_SAMPLES

    def test_full_topology_creates_every_pairwise_cross_omic_relation(self, three_omics_full_run):
        proc, _ = three_omics_full_run
        for rel in _pairwise_relation_keys(THREE_OMIC_NAMES):
            assert rel in proc.stdout, f"Expected relation '{rel}' in Edge dict under --topology full"

    def test_core_visualizations_saved(self, three_omics_full_run):
        _, result_dir = three_omics_full_run
        plots_dir = os.path.join(result_dir, "plots")
        for filename in [
            "training_loss_components.pdf",
            "Confusion_matrix_figure.pdf",
            "Interpretability_nodes_attention_figure.pdf",
            "embeddings_evaluations_figure.pdf",
            "network_interactive.html",
        ]:
            assert os.path.exists(os.path.join(plots_dir, filename)), f"Missing expected output: {filename}"

    def test_supplementary_visualizations_saved(self, three_omics_full_run):
        _, result_dir = three_omics_full_run
        supplementary_dir = os.path.join(result_dir, "supplementary")
        for filename in [
            "attention_concentration_by_omic.pdf",
            "attention_distribution_by_omic.pdf",
            "network_hub_nodes.pdf",
        ]:
            assert os.path.exists(os.path.join(supplementary_dir, filename)), f"Missing supplementary output: {filename}"


# ---------------------------------------------------------------------------
# Same three omics, --topology star overriding the manifest's own 'full'.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def three_omics_star_run(provisioned_python, tmp_path_factory, omic3_path):
    result_dir = tmp_path_factory.mktemp("three_omics_star_result")
    # Manifest says 'full' - the --topology CLI flag should win (resolve_manifest()).
    manifest_path = write_manifest(
        tmp_path_factory.mktemp("three_omics_star_manifest") / "manifest.json",
        [omic_entry("omic1", OMIC1_PATH), omic_entry("omic2", OMIC2_PATH), omic_entry("omic3", omic3_path)],
        topology="full",
    )
    proc = run_biomix_manifest(provisioned_python, METADATA_3GROUP, manifest_path, result_dir,
                                extra_args=["--topology", "star"])
    return proc, result_dir


class TestTopologyStarOverride:

    def test_process_completes_successfully(self, three_omics_star_run):
        proc, _ = three_omics_star_run
        _assert_success(proc)

    def test_cli_flag_overrides_manifest_topology(self, three_omics_star_run):
        proc, _ = three_omics_star_run
        assert "topology=star" in proc.stdout

    def test_star_topology_creates_no_cross_omic_relations(self, three_omics_star_run):
        """--topology star: omics only ever connect through sample nodes, so
        none of the direct omic-omic relation names should ever appear."""
        proc, _ = three_omics_star_run
        for rel in _pairwise_relation_keys(THREE_OMIC_NAMES):
            assert rel not in proc.stdout, f"Unexpected relation '{rel}' in Edge dict under --topology star"

    def test_star_topology_keeps_omic_sample_and_self_relations(self, three_omics_star_run):
        proc, _ = three_omics_star_run
        for name in THREE_OMIC_NAMES:
            assert f"{name}_sample" in proc.stdout
            assert f"{name}_self" in proc.stdout

    def test_embeddings_still_saved_for_every_omic(self, three_omics_star_run):
        _, result_dir = three_omics_star_run
        for name in THREE_OMIC_NAMES:
            matrix = _read_embedding_matrix(result_dir, name)
            assert matrix.shape[0] == N_FEATURES_PER_OMIC


# ---------------------------------------------------------------------------
# --sample_mode union vs. intersection: omic3 is missing sample S9 entirely.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_mode_manifest(tmp_path_factory, omic3_missing_one_sample_path):
    return write_manifest(
        tmp_path_factory.mktemp("sample_mode_manifest") / "manifest.json",
        [omic_entry("omic1", OMIC1_PATH), omic_entry("omic2", OMIC2_PATH),
         omic_entry("omic3", omic3_missing_one_sample_path)],
    )


@pytest.fixture(scope="module")
def sample_mode_intersection_run(provisioned_python, tmp_path_factory, sample_mode_manifest):
    result_dir = tmp_path_factory.mktemp("sample_mode_intersection_result")
    proc = run_biomix_manifest(provisioned_python, METADATA_3GROUP, sample_mode_manifest, result_dir,
                                extra_args=["--sample_mode", "intersection"])
    return proc, result_dir


@pytest.fixture(scope="module")
def sample_mode_union_run(provisioned_python, tmp_path_factory, sample_mode_manifest):
    result_dir = tmp_path_factory.mktemp("sample_mode_union_result")
    proc = run_biomix_manifest(provisioned_python, METADATA_3GROUP, sample_mode_manifest, result_dir,
                                extra_args=["--sample_mode", "union"])
    return proc, result_dir


class TestSampleModeUnionVsIntersection:
    """omic3 is missing sample S9 entirely (see omic3_missing_one_sample_path)
    - the two sample_mode settings should disagree on whether S9 survives at
    all, which is the whole point of the flag."""

    def test_both_runs_complete_successfully(self, sample_mode_intersection_run, sample_mode_union_run):
        _assert_success(sample_mode_intersection_run[0])
        _assert_success(sample_mode_union_run[0])

    def test_intersection_drops_the_incomplete_sample(self, sample_mode_intersection_run):
        _, result_dir = sample_mode_intersection_run
        sample_matrix = _read_embedding_matrix(result_dir, "sample")
        assert sample_matrix.shape[0] == N_SAMPLES - 1

    def test_union_keeps_every_sample(self, sample_mode_union_run):
        _, result_dir = sample_mode_union_run
        sample_matrix = _read_embedding_matrix(result_dir, "sample")
        assert sample_matrix.shape[0] == N_SAMPLES

    def test_union_reports_the_missing_sample(self, sample_mode_union_run):
        proc, _ = sample_mode_union_run
        assert "[sample_mode=union] omic3: 1/9 samples have no data for this omic" in proc.stdout

    def test_intersection_reports_no_missing_sample_message(self, sample_mode_intersection_run):
        proc, _ = sample_mode_intersection_run
        assert "[sample_mode=union]" not in proc.stdout

    def test_every_omic_embedding_still_saved_under_union(self, sample_mode_union_run):
        _, result_dir = sample_mode_union_run
        for name in THREE_OMIC_NAMES:
            matrix = _read_embedding_matrix(result_dir, name)
            assert matrix.shape[0] == N_FEATURES_PER_OMIC


# ---------------------------------------------------------------------------
# Two omics sharing the same `kind` (same-cohort, two-platform "same omic
# twice" case) - they must stay independent node types, never merged.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def duplicate_kind_run(provisioned_python, tmp_path_factory, duplicate_kind_omic_path):
    result_dir = tmp_path_factory.mktemp("duplicate_kind_result")
    manifest_path = write_manifest(
        tmp_path_factory.mktemp("duplicate_kind_manifest") / "manifest.json",
        [
            omic_entry("omic1", OMIC1_PATH, kind="transcriptomics"),
            omic_entry("omic1b", duplicate_kind_omic_path, kind="transcriptomics"),
            omic_entry("omic2", OMIC2_PATH, kind="metabolomics"),
        ],
    )
    proc = run_biomix_manifest(provisioned_python, METADATA_3GROUP, manifest_path, result_dir)
    return proc, result_dir


class TestTwoOmicsSameKind:
    """'Same omic twice', same-cohort case: two manifest entries (omic1,
    omic1b) share `kind=transcriptomics` but have distinct `name`/`path` -
    they must stay two independent node types/feature spaces, never merged,
    exactly as omics_manifest.py's docstring documents."""

    def test_process_completes_successfully(self, duplicate_kind_run):
        proc, _ = duplicate_kind_run
        _assert_success(proc)

    def test_reports_all_three_omics_by_name_not_kind(self, duplicate_kind_run):
        proc, _ = duplicate_kind_run
        assert "Omics (3): ['omic1', 'omic1b', 'omic2']" in proc.stdout

    def test_same_kind_omics_get_independent_output_dirs(self, duplicate_kind_run):
        _, result_dir = duplicate_kind_run
        assert os.path.isdir(os.path.join(result_dir, "omic1"))
        assert os.path.isdir(os.path.join(result_dir, "omic1b"))
        omic1_matrix = _read_embedding_matrix(result_dir, "omic1")
        omic1b_matrix = _read_embedding_matrix(result_dir, "omic1b")
        assert omic1_matrix.shape[0] == N_FEATURES_PER_OMIC
        assert omic1b_matrix.shape[0] == N_FEATURES_PER_OMIC

    def test_same_kind_omics_produce_different_embeddings(self, duplicate_kind_run):
        """They're independently-generated feature sets, not the same file
        twice - their learned embeddings should not be identical, confirming
        `kind` never collapses two omics into one feature space."""
        _, result_dir = duplicate_kind_run
        omic1_matrix = _read_embedding_matrix(result_dir, "omic1")
        omic1b_matrix = _read_embedding_matrix(result_dir, "omic1b")
        assert not np.allclose(omic1_matrix.values, omic1b_matrix.values)

    def test_full_topology_still_wires_all_three_pairs_directly(self, duplicate_kind_run):
        proc, _ = duplicate_kind_run
        for rel in _pairwise_relation_keys(["omic1", "omic1b", "omic2"]):
            assert rel in proc.stdout


# ---------------------------------------------------------------------------
# Four omics: not a fixed ceiling - the same manifest-driven code path handles
# more omics with no code changes, at the O(k^2) pairwise-mask cost
# --topology full documents.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def four_omics_run(provisioned_python, tmp_path_factory, omic3_path, omic4_path):
    result_dir = tmp_path_factory.mktemp("four_omics_result")
    manifest_path = write_manifest(
        tmp_path_factory.mktemp("four_omics_manifest") / "manifest.json",
        [
            omic_entry("omic1", OMIC1_PATH), omic_entry("omic2", OMIC2_PATH),
            omic_entry("omic3", omic3_path), omic_entry("omic4", omic4_path),
        ],
    )
    proc = run_biomix_manifest(provisioned_python, METADATA_3GROUP, manifest_path, result_dir)
    return proc, result_dir


class TestFourOmicsScaleUp:

    def test_process_completes_successfully(self, four_omics_run):
        proc, _ = four_omics_run
        _assert_success(proc)

    def test_all_four_omic_output_dirs_created(self, four_omics_run):
        _, result_dir = four_omics_run
        for name in FOUR_OMIC_NAMES:
            assert os.path.isdir(os.path.join(result_dir, name))

    def test_all_six_pairwise_relations_present_under_full_topology(self, four_omics_run):
        proc, _ = four_omics_run
        expected = _pairwise_relation_keys(FOUR_OMIC_NAMES)
        assert len(expected) == 6  # C(4,2)
        for rel in expected:
            assert rel in proc.stdout, f"Expected relation '{rel}' in Edge dict"


# ---------------------------------------------------------------------------
# The legacy --omic1_path/--omic2_path flags are wrapped into an equivalent
# 2-entry manifest under the hood (omics_manifest.manifest_from_legacy_args) -
# this should be genuinely equivalent end-to-end, not just "also works".
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def legacy_two_omic_run(provisioned_python, tmp_path_factory):
    result_dir = tmp_path_factory.mktemp("legacy_equivalent_result")
    proc = run_biomix(provisioned_python, METADATA_3GROUP, result_dir)
    return proc, result_dir


@pytest.fixture(scope="module")
def manifest_two_omic_run(provisioned_python, tmp_path_factory):
    result_dir = tmp_path_factory.mktemp("manifest_equivalent_result")
    manifest_path = write_manifest(
        tmp_path_factory.mktemp("manifest_equivalent_manifest") / "manifest.json",
        [omic_entry("omic1", OMIC1_PATH), omic_entry("omic2", OMIC2_PATH)],
    )
    proc = run_biomix_manifest(provisioned_python, METADATA_3GROUP, manifest_path, result_dir)
    return proc, result_dir


class TestManifestEquivalentToLegacyTwoOmicFlags:

    def test_both_paths_complete_successfully(self, legacy_two_omic_run, manifest_two_omic_run):
        _assert_success(legacy_two_omic_run[0])
        _assert_success(manifest_two_omic_run[0])

    def test_both_paths_name_output_dirs_omic1_omic2(self, legacy_two_omic_run, manifest_two_omic_run):
        """Confirms the legacy flags are genuinely wrapped into the same
        'omic1'/'omic2' manifest names used everywhere else in this file -
        NOT the original PRAMIGO's hardcoded 'gene'/'metabo' dirs. (If this
        ever regresses, test_biomix_pipeline.py's own
        test_embeddings_saved_with_expected_shapes, which still checks for
        'gene'/'metabo', would need updating too - see also that test.)"""
        for _, result_dir in (legacy_two_omic_run, manifest_two_omic_run):
            assert os.path.isdir(os.path.join(result_dir, "omic1"))
            assert os.path.isdir(os.path.join(result_dir, "omic2"))

    def test_both_paths_produce_the_same_shaped_embeddings(self, legacy_two_omic_run, manifest_two_omic_run):
        for _, result_dir in (legacy_two_omic_run, manifest_two_omic_run):
            for name in ("omic1", "omic2"):
                matrix = _read_embedding_matrix(result_dir, name)
                assert matrix.shape[0] == N_FEATURES_PER_OMIC
            sample_matrix = _read_embedding_matrix(result_dir, "sample")
            assert sample_matrix.shape[0] == N_SAMPLES


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
