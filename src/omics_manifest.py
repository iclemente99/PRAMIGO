"""
Manifest-driven configuration for an arbitrary number of omic layers.

This is the piece of PRAMIGO_gen that replaces PRAMIGO's original "hand-duplicate
everything per omic index" pattern (--omic1_path, --omic2_path, --omic3_path, ...,
a bespoke reduction() call and mask per omic, num_types/num_relations pasted in by
hand, node_type==3 literals, ...). Instead of a fixed CLI surface, biomix_hgt.py
takes a single --omics_manifest pointing at a small YAML or JSON file listing
however many omics the run actually needs:

    topology: full              # or 'star' - see --topology / README
    sample_mode: intersection   # or 'union' - see --sample_mode / README
    omics:
      - name: transcriptomics
        path: ./data/rnaseq.tsv
        rate: 1.0                # optional, defaults to 1.0
        corr_cutoff: 0.7         # optional, defaults to 0.7
        kind: transcriptomics    # optional modality tag, defaults to `name`
      - name: methylomics
        path: ./data/methyl.tsv
        corr_cutoff: 0.6

`name` must be unique and is used everywhere downstream as the node-type / output
label for that omic (output subfolders, plot legends, edge-relation names, etc.),
so keep it short and filesystem-safe. `kind` is a separate, optional tag meant for
the "same modality, two files" case (e.g. two transcriptomics cohorts/platforms
that should be grouped/colored together in reports) without merging them into one
feature space - see README's "Same omic twice" section for why that's kept
distinct from a true multi-cohort merge.

For a smooth path from the original 2-omic CLI, --omic1_path/--omic2_path (plus
their --omic1_rate/--omic2_corr_cutoff/etc. siblings) are still accepted; if no
--omics_manifest is given, they're wrapped into an equivalent 2-entry manifest
automatically (see manifest_from_legacy_args below).
"""
import os
import json


DEFAULT_RATE = 1.0
DEFAULT_CORR_CUTOFF = 0.7
RESERVED_NAMES = {'sample', 'samples'}


class ManifestError(ValueError):
    """Raised for any structurally invalid --omics_manifest file."""
    pass


def _load_raw(path):
    with open(path, 'r') as fh:
        text = fh.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as e:
            raise ManifestError(
                f"--omics_manifest {path} is a YAML file but PyYAML is not installed "
                f"in this environment. Install it (`pip install pyyaml`) or provide "
                f"an equivalent .json manifest instead."
            ) from e
        return yaml.safe_load(text)
    if ext == '.json':
        return json.loads(text)
    # No/unrecognized extension: try YAML first (a superset of JSON syntax), and
    # fall back to plain JSON if PyYAML simply isn't installed.
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def load_manifest(path):
    """Loads and validates an omics manifest (YAML or JSON). Returns a dict:
        {'topology': 'full'|'star'|None,
         'sample_mode': 'intersection'|'union'|None,
         'omics': [{'name':..., 'path':..., 'rate':..., 'corr_cutoff':..., 'kind':...}, ...]}
    `topology`/`sample_mode` are None when the manifest doesn't set them, so the
    caller can fall back to a --topology/--sample_mode CLI flag or a hard default.
    Every omic entry is filled in with defaults for the optional fields.
    """
    if not os.path.exists(path):
        raise ManifestError(f"--omics_manifest file not found: {path}")
    raw = _load_raw(path)
    if not isinstance(raw, dict) or 'omics' not in raw:
        raise ManifestError(f"Manifest {path} must be a mapping with a top-level 'omics' list.")
    omics_raw = raw['omics']
    if not isinstance(omics_raw, list) or len(omics_raw) < 1:
        raise ManifestError(f"Manifest {path}: 'omics' must be a non-empty list.")

    seen_names = set()
    omics = []
    for i, entry in enumerate(omics_raw):
        if not isinstance(entry, dict) or 'name' not in entry or 'path' not in entry:
            raise ManifestError(f"Manifest {path}: omics[{i}] must have at least 'name' and 'path'.")
        name = str(entry['name']).strip()
        if not name:
            raise ManifestError(f"Manifest {path}: omics[{i}] has an empty 'name'.")
        if name.lower() in RESERVED_NAMES:
            raise ManifestError(
                f"Manifest {path}: omics[{i}] name '{name}' is reserved for the sample "
                f"node type and can't be used as an omic name."
            )
        if name in seen_names:
            raise ManifestError(f"Manifest {path}: duplicate omic name '{name}'.")
        seen_names.add(name)
        omics.append({
            'name': name,
            'path': entry['path'],
            'rate': float(entry.get('rate', DEFAULT_RATE)),
            'corr_cutoff': float(entry.get('corr_cutoff', DEFAULT_CORR_CUTOFF)),
            'kind': str(entry.get('kind', name)),
        })

    topology = raw.get('topology')
    if topology is not None and topology not in ('full', 'star'):
        raise ManifestError(f"Manifest {path}: topology must be 'full' or 'star', got {topology!r}.")
    sample_mode = raw.get('sample_mode')
    if sample_mode is not None and sample_mode not in ('intersection', 'union'):
        raise ManifestError(
            f"Manifest {path}: sample_mode must be 'intersection' or 'union', got {sample_mode!r}."
        )

    return {'topology': topology, 'sample_mode': sample_mode, 'omics': omics}


def manifest_from_legacy_args(args):
    """Builds the equivalent 2-omic manifest dict from the legacy
    --omic1_path/--omic2_path (+ rate/corr_cutoff) flags, for users not yet using
    --omics_manifest. This is exactly the input surface the original PRAMIGO
    biomix_hgt.py exposed."""
    if not args.omic1_path or not args.omic2_path:
        raise ManifestError(
            "Provide either --omics_manifest, or both --omic1_path and --omic2_path."
        )
    omics = [
        {'name': 'omic1', 'path': args.omic1_path, 'rate': args.omic1_rate,
         'corr_cutoff': args.omic1_corr_cutoff, 'kind': 'omic1'},
        {'name': 'omic2', 'path': args.omic2_path, 'rate': args.omic2_rate,
         'corr_cutoff': args.omic2_corr_cutoff, 'kind': 'omic2'},
    ]
    return {'topology': None, 'sample_mode': None, 'omics': omics}


def resolve_manifest(args):
    """Single entry point biomix_hgt.py calls: returns (omics, topology, sample_mode)
    after merging --omics_manifest (or the legacy --omic1_path/--omic2_path fallback)
    with the --topology/--sample_mode CLI flags. CLI flags win when explicitly passed;
    otherwise the manifest's own topology/sample_mode is used; otherwise the
    documented defaults ('full' / 'intersection', matching original PRAMIGO behavior)."""
    if args.omics_manifest:
        manifest = load_manifest(args.omics_manifest)
    else:
        manifest = manifest_from_legacy_args(args)

    if len(manifest['omics']) < 2:
        raise ManifestError(
            "PRAMIGO_gen needs at least 2 omics (plus the implicit 'sample' node type) "
            "to build a heterogeneous graph; got 1."
        )

    topology = args.topology or manifest['topology'] or 'full'
    sample_mode = args.sample_mode or manifest['sample_mode'] or 'intersection'
    return manifest['omics'], topology, sample_mode
