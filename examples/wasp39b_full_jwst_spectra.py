"""
WASP-39 b Transmission Spectrum Retrieval with ExoJAX + NumPyro
===============================================================

This example demonstrates how to retrieve the JWST transmission 
spectrum of NIRISS/SOSS+NIRSPEC/G395H+MIRI using *ExoJAX* and 
*NumPyro*'s Hamiltonian Monte-Carlo **NUTS** sampler for Bayesian 
inference.

Hajime Kawahara, Shotaro Tada

"""
# %%

import argparse
from dataclasses import dataclass
import json
import glob
import logging
import math
import os
import platform
import sys
import tempfile
import warnings

DEFAULT_MOLECULES = ("H2O", "CO", "CO2", "H2S", "SO2", "SiO")
SUPPORTED_MOLECULES = set(DEFAULT_MOLECULES)
DEFAULT_CHANNELS = ("niriss_order1", "niriss_order2", "nirspec_g395h", "miri_lrs")
SUPPORTED_CHANNELS = set(DEFAULT_CHANNELS)
DEFAULT_CIA_PAIRS = ("H2H2", "H2He")
SUPPORTED_CIA_PAIRS = set(DEFAULT_CIA_PAIRS)
CIA_RELATIVE_FILES = {
    "H2H2": os.path.join(".db_CIA", "H2-H2_2011.cia"),
    "H2He": os.path.join(".db_CIA", "H2-He_2011.cia"),
}
CKD_RELATIVE_PATHS = {
    "H2O": os.path.join("H2O", "1H2-16O", "POKAZATEL"),
    "CO": os.path.join("CO", "12C-16O", "Li2015"),
    "CO2": os.path.join("CO2", "12C-16O2", "UCL-4000"),
    "H2S": os.path.join("H2S", "1H2-32S", "AYT2"),
    "SO2": os.path.join("SO2", "32S-16O2", "ExoAmes"),
    "SiO": os.path.join("SiO", "28Si-16O", "SiOUVenIR"),
}
CKD_REQUIRED_DATASETS = (
    "mol_mass",
    "bin_centers",
    "samples",
    "weights",
    "t",
    "p",
    "kcoeff",
)

parser = argparse.ArgumentParser(
    description="WASP-39b wide-wavelength JWST transmission spectrum retrieval.",
    allow_abbrev=False,
)
parser.add_argument(
    "--plot-data-only",
    action="store_true",
    help="Stop after plotting the observed NIRISS, NIRSpec, and MIRI spectra.",
)
parser.add_argument(
    "--summarize-data",
    action="store_true",
    help="Print a summary of the observed spectra and stop before plotting.",
)
parser.add_argument(
    "--data-mode",
    choices=("nirspec", "wide"),
    default="nirspec",
    help=(
        "Observed data vector used by the retrieval. 'nirspec' keeps the current "
        "line-by-line prototype path; 'wide' requires the CKD forward model."
    ),
)
parser.add_argument(
    "--opacity-mode",
    choices=("premodit", "ckd"),
    default="premodit",
    help="Opacity mode used by the forward model.",
)
parser.add_argument(
    "--data-dir",
    default="wasp39_data",
    help="Directory containing the bundled WASP-39b observed spectra.",
)
parser.add_argument(
    "--opacity-root",
    default="path_to",
    help="Root directory containing HITEMP, ExoMol, and CIA line-list databases.",
)
parser.add_argument(
    "--ckd-root",
    default=".database",
    help="Root directory containing ExoMolOP CKD opacity tables.",
)
parser.add_argument(
    "--ckd-table-paths",
    default="",
    help=(
        "Optional comma-separated CKD table overrides of the form "
        "MOL=/path/to/table.h5. Use this when a molecule directory contains "
        "multiple local ExoMolOP tables."
    ),
)
parser.add_argument(
    "--allow-ckd-download",
    action="store_true",
    help="Allow missing ExoMolOP CKD tables to be downloaded during opacity loading.",
)
parser.add_argument(
    "--molecules",
    default=",".join(DEFAULT_MOLECULES),
    help="Comma-separated molecular opacity list.",
)
parser.add_argument(
    "--channels",
    default=",".join(DEFAULT_CHANNELS),
    help="Comma-separated observed-spectrum channels.",
)
parser.add_argument(
    "--cia-pairs",
    default=",".join(DEFAULT_CIA_PAIRS),
    help="Comma-separated CIA opacity pairs, or 'none' to disable CIA opacity.",
)
parser.add_argument(
    "--check-inputs",
    action="store_true",
    help="Print the expected input file and directory status, then stop.",
)
parser.add_argument(
    "--input-status-json",
    default="",
    help="Optional path to save --check-inputs status as JSON.",
)
parser.add_argument(
    "--check-forward",
    action="store_true",
    help="Run one fiducial forward-model evaluation, print shape checks, then stop.",
)
parser.add_argument(
    "--forward-check-json",
    default="",
    help="Optional path to save --check-forward diagnostics as JSON.",
)
parser.add_argument(
    "--quick",
    action="store_true",
    help="Use short SVI/HMC settings for a smoke test.",
)
parser.add_argument(
    "--max-observed",
    type=int,
    default=None,
    help=(
        "Use at most this many evenly spaced observed data points for diagnostics. "
        "Leave unset for the full selected data vector."
    ),
)
parser.add_argument(
    "--jax-platform",
    choices=("auto", "cpu", "gpu", "tpu"),
    default="auto",
    help="JAX platform override applied before importing JAX.",
)
parser.add_argument(
    "--rng-seed",
    type=int,
    default=0,
    help="Random seed for SVI, HMC, and posterior predictive draws.",
)
parser.add_argument(
    "--rv-min",
    type=float,
    default=-200.0,
    help="Lower radial-velocity prior bound in km/s.",
)
parser.add_argument(
    "--rv-max",
    type=float,
    default=0.0,
    help="Upper radial-velocity prior bound in km/s.",
)
parser.add_argument(
    "--svi-steps",
    type=int,
    default=1000,
    help="Number of SVI optimization steps before HMC.",
)
parser.add_argument(
    "--svi-lr",
    type=float,
    default=0.005,
    help="SVI Adam learning rate.",
)
parser.add_argument(
    "--num-warmup",
    type=int,
    default=1000,
    help="Number of HMC-NUTS warmup steps.",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of HMC-NUTS posterior samples.",
)
parser.add_argument(
    "--num-chains",
    type=int,
    default=1,
    help="Number of HMC-NUTS chains.",
)
parser.add_argument(
    "--chain-method",
    choices=("parallel", "sequential", "vectorized"),
    default="parallel",
    help="NumPyro MCMC chain method.",
)
parser.add_argument(
    "--max-tree-depth",
    type=int,
    default=5,
    help="Maximum NUTS tree depth.",
)
parser.add_argument(
    "--no-progress-bar",
    action="store_true",
    help="Disable NumPyro progress bars.",
)
parser.add_argument(
    "--skip-corner",
    action="store_true",
    help="Skip corner plot generation.",
)
parser.add_argument(
    "--svi-plot-samples",
    type=int,
    default=1000,
    help="Number of SVI guide samples used only for corner plot diagnostics.",
)
parser.add_argument(
    "--skip-data-plot",
    action="store_true",
    help="Skip observed-spectrum plot generation during retrieval.",
)
parser.add_argument(
    "--skip-diagnostic-plots",
    action="store_true",
    help="Skip post-HMC SVI-loss, spectrum-overlay, and corner diagnostics.",
)
parser.add_argument(
    "--output-dir",
    default="output_full_wasp39b",
    help="Directory for retrieval products.",
)
parser.add_argument(
    "--data-plot-path",
    default="wasp39b_transmission_spectrum.png",
    help="Path for the observed-spectrum plot.",
)
args = parser.parse_args()


def parse_molecule_list(molecule_text):
    """Parse and validate a comma-separated molecule list."""
    molecules = tuple(mol.strip() for mol in molecule_text.split(",") if mol.strip())
    if not molecules:
        parser.error("--molecules must include at least one molecule.")
    duplicates = sorted({mol for mol in molecules if molecules.count(mol) > 1})
    if duplicates:
        parser.error("Duplicate molecules in --molecules: " + ", ".join(duplicates))
    unsupported = [mol for mol in molecules if mol not in SUPPORTED_MOLECULES]
    if unsupported:
        parser.error(
            "Unsupported molecules in --molecules: "
            + ", ".join(unsupported)
            + ". Supported molecules are: "
            + ", ".join(DEFAULT_MOLECULES)
        )
    return molecules


def parse_channel_list(channel_text):
    """Parse and validate a comma-separated observed channel list."""
    channels = tuple(
        channel.strip() for channel in channel_text.split(",") if channel.strip()
    )
    if not channels:
        parser.error("--channels must include at least one channel.")
    duplicates = sorted({channel for channel in channels if channels.count(channel) > 1})
    if duplicates:
        parser.error("Duplicate channels in --channels: " + ", ".join(duplicates))
    unsupported = [channel for channel in channels if channel not in SUPPORTED_CHANNELS]
    if unsupported:
        parser.error(
            "Unsupported channels in --channels: "
            + ", ".join(unsupported)
            + ". Supported channels are: "
            + ", ".join(DEFAULT_CHANNELS)
        )
    return channels


def parse_cia_pair_list(cia_pair_text):
    """Parse and validate a comma-separated CIA pair list."""
    if cia_pair_text.strip().lower() == "none":
        return tuple()
    cia_pairs = tuple(pair.strip() for pair in cia_pair_text.split(",") if pair.strip())
    if not cia_pairs:
        parser.error("--cia-pairs must include at least one CIA pair.")
    duplicates = sorted({pair for pair in cia_pairs if cia_pairs.count(pair) > 1})
    if duplicates:
        parser.error("Duplicate CIA pairs in --cia-pairs: " + ", ".join(duplicates))
    unsupported = [pair for pair in cia_pairs if pair not in SUPPORTED_CIA_PAIRS]
    if unsupported:
        parser.error(
            "Unsupported CIA pairs in --cia-pairs: "
            + ", ".join(unsupported)
            + ". Supported CIA pairs are: "
            + ", ".join(DEFAULT_CIA_PAIRS)
        )
    return cia_pairs


def parse_ckd_table_path_map(table_path_text, molecules):
    """Parse optional molecule-specific CKD table path overrides."""
    if not table_path_text.strip():
        return {}

    table_paths = {}
    for entry in table_path_text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            parser.error("--ckd-table-paths entries must use MOL=/path/to/table.h5.")
        mol, path = (part.strip() for part in entry.split("=", 1))
        if mol not in SUPPORTED_MOLECULES:
            parser.error(
                "Unsupported molecule in --ckd-table-paths: "
                f"{mol}. Supported molecules are: "
                + ", ".join(DEFAULT_MOLECULES)
            )
        if mol not in molecules:
            parser.error(
                f"--ckd-table-paths specifies {mol}, but {mol} is not selected "
                "by --molecules."
            )
        if not path:
            parser.error(f"--ckd-table-paths entry for {mol} has an empty path.")
        if mol in table_paths:
            parser.error(f"--ckd-table-paths specifies {mol} more than once.")
        table_paths[mol] = path
    return table_paths


def validate_numeric_args(args):
    """Validate scalar run-control arguments before heavy imports."""
    positive_integer_args = {
        "--svi-steps": args.svi_steps,
        "--num-samples": args.num_samples,
        "--num-chains": args.num_chains,
        "--max-tree-depth": args.max_tree_depth,
        "--svi-plot-samples": args.svi_plot_samples,
    }
    for name, value in positive_integer_args.items():
        if value <= 0:
            parser.error(f"{name} must be a positive integer.")

    if args.svi_lr <= 0.0:
        parser.error("--svi-lr must be positive.")
    if args.num_warmup < 0:
        parser.error("--num-warmup must be zero or a positive integer.")
    if args.max_observed is not None and args.max_observed <= 0:
        parser.error("--max-observed must be a positive integer when set.")
    if args.rng_seed < 0:
        parser.error("--rng-seed must be zero or a positive integer.")
    if not math.isfinite(args.rv_min) or not math.isfinite(args.rv_max):
        parser.error("--rv-min and --rv-max must be finite.")
    if args.rv_min >= args.rv_max:
        parser.error("--rv-min must be smaller than --rv-max.")


def validate_artifact_args(args):
    """Validate artifact path options are used with the mode that writes them."""
    if args.input_status_json and not args.check_inputs:
        parser.error("--input-status-json requires --check-inputs.")
    if args.forward_check_json and not args.check_forward:
        parser.error("--forward-check-json requires --check-forward.")


def validate_mode_specific_args(args):
    """Reject CKD-only controls when the CKD opacity path is not selected."""
    if args.opacity_mode != "ckd" and args.ckd_table_paths.strip():
        parser.error("--ckd-table-paths requires --opacity-mode ckd.")
    if args.opacity_mode != "ckd" and args.allow_ckd_download:
        parser.error("--allow-ckd-download requires --opacity-mode ckd.")


selected_molecules = parse_molecule_list(args.molecules)
selected_channels = parse_channel_list(args.channels)
selected_cia_pairs = parse_cia_pair_list(args.cia_pairs)
selected_ckd_table_paths = parse_ckd_table_path_map(
    args.ckd_table_paths, selected_molecules
)
validate_numeric_args(args)
validate_artifact_args(args)
validate_mode_specific_args(args)


def format_selection(values):
    """Format selected CLI values for human-readable status output."""
    values = tuple(values)
    return ", ".join(values) if values else "(none)"


if args.quick:
    args.svi_steps = min(args.svi_steps, 20)
    args.num_warmup = min(args.num_warmup, 10)
    args.num_samples = min(args.num_samples, 10)
    args.max_tree_depth = min(args.max_tree_depth, 3)
    args.svi_plot_samples = min(args.svi_plot_samples, 100)

if (
    args.data_mode == "wide"
    and args.opacity_mode != "ckd"
    and not (args.plot_data_only or args.summarize_data or args.check_inputs)
):
    parser.error(
        "Wide-wavelength retrieval requires --opacity-mode ckd. The preMODIT "
        "prototype path is still NIRSpec-only."
    )

if (
    args.data_mode == "nirspec"
    and "nirspec_g395h" not in selected_channels
    and not (args.plot_data_only or args.summarize_data or args.check_inputs)
):
    parser.error("--data-mode nirspec requires --channels to include nirspec_g395h.")

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_input_path(path):
    """Resolve an input path from the current directory or this example directory."""
    path = os.fspath(path)
    if os.path.isabs(path) or os.path.exists(path):
        return path
    script_relative_path = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(script_relative_path):
        return script_relative_path
    return path


args.data_dir = resolve_input_path(args.data_dir)
args.opacity_root = resolve_input_path(args.opacity_root)
args.ckd_root = resolve_input_path(args.ckd_root)
selected_ckd_table_paths = {
    mol: resolve_input_path(path) for mol, path in selected_ckd_table_paths.items()
}

ciapath_list = {
    pair: os.path.join(args.opacity_root, CIA_RELATIVE_FILES[pair])
    for pair in DEFAULT_CIA_PAIRS
}
ciapath_list = {pair: ciapath_list[pair] for pair in selected_cia_pairs}


def ckd_h5_paths(path):
    """Return sorted CKD h5 table paths in a molecule directory."""
    return sorted(glob.glob(os.path.join(path, "*.h5")))


def ckd_nonempty_h5_paths(path):
    """Return sorted non-empty CKD h5 table paths in a molecule directory."""
    return [h5_path for h5_path in ckd_h5_paths(path) if os.path.getsize(h5_path) > 0]


def ckd_source_path(mol):
    """Return explicit CKD table path or the default molecule directory."""
    return selected_ckd_table_paths.get(
        mol, os.path.join(args.ckd_root, CKD_RELATIVE_PATHS[mol])
    )


def ckd_resolved_source_path(mol):
    """Return the CKD source path that will be handed to OpaCKD."""
    path = ckd_source_path(mol)
    if mol in selected_ckd_table_paths:
        return path
    if os.path.isdir(path):
        nonempty_h5_paths = ckd_nonempty_h5_paths(path)
        if len(nonempty_h5_paths) == 1:
            return nonempty_h5_paths[0]
    return path


def ckd_table_file_metadata(path):
    """Return lightweight file metadata for a resolved CKD table."""
    stat = os.stat(path)
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def ckd_table_schema_summary(path):
    """Return a lightweight ExoMolOP CKD h5 schema summary."""
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            datasets = {}
            missing = []
            for key in CKD_REQUIRED_DATASETS:
                if key not in handle:
                    missing.append(key)
                    continue
                dataset = handle[key]
                datasets[key] = {
                    "shape": [int(size) for size in dataset.shape],
                    "dtype": str(dataset.dtype),
                }
            if missing:
                return {
                    "status": "missing_datasets",
                    "missing_datasets": missing,
                    "datasets": datasets,
                }

            shape_issues = []
            value_issues = []
            shapes = {key: datasets[key]["shape"] for key in CKD_REQUIRED_DATASETS}
            kcoeff_shape = shapes["kcoeff"]
            if len(kcoeff_shape) != 4:
                shape_issues.append("kcoeff must be four-dimensional")
            else:
                expected_lengths = {
                    "p": kcoeff_shape[0],
                    "t": kcoeff_shape[1],
                    "bin_centers": kcoeff_shape[2],
                    "samples": kcoeff_shape[3],
                }
                for key, expected_length in expected_lengths.items():
                    if shapes[key] != [expected_length]:
                        shape_issues.append(
                            f"{key} shape {shapes[key]} does not match "
                            f"kcoeff axis length {expected_length}"
                        )
                if shapes["weights"] != shapes["samples"]:
                    shape_issues.append("weights shape must match samples shape")
            if "mol_name" in handle:
                dataset = handle["mol_name"]
                datasets["mol_name"] = {
                    "shape": [int(size) for size in dataset.shape],
                    "dtype": str(dataset.dtype),
                    "optional": True,
                }
                if len(datasets["mol_name"]["shape"]) != 1:
                    shape_issues.append(
                        "mol_name must be one-dimensional when present"
                    )
            if len(shapes["mol_mass"]) != 1 or shapes["mol_mass"][0] < 1:
                shape_issues.append("mol_mass must be a non-empty one-dimensional dataset")
            numeric_arrays = {
                key: np.asarray(handle[key][:], dtype=float)
                for key in ("mol_mass", "bin_centers", "samples", "weights", "t", "p")
            }
            numeric_summary = {}
            for key, values in numeric_arrays.items():
                numeric_summary[key] = {
                    "min": float(np.nanmin(values)) if values.size else None,
                    "max": float(np.nanmax(values)) if values.size else None,
                }
                if values.size == 0:
                    value_issues.append(f"{key} must be non-empty")
                    continue
                if not np.all(np.isfinite(values)):
                    value_issues.append(f"{key} must contain finite values")
            if numeric_arrays["mol_mass"].size and not np.all(
                numeric_arrays["mol_mass"] > 0.0
            ):
                value_issues.append("mol_mass must be positive")
            for key in ("bin_centers", "t", "p"):
                values = numeric_arrays[key]
                if values.size and not np.all(values > 0.0):
                    value_issues.append(f"{key} must be positive")
                if values.size > 1 and np.any(np.diff(values) <= 0.0):
                    value_issues.append(f"{key} must be strictly increasing")
            samples = numeric_arrays["samples"]
            if samples.size and not np.all((samples >= 0.0) & (samples <= 1.0)):
                value_issues.append("samples must lie within [0, 1]")
            if samples.size > 1 and np.any(np.diff(samples) <= 0.0):
                value_issues.append("samples must be strictly increasing")
            weights = numeric_arrays["weights"]
            if weights.size and not np.all(weights > 0.0):
                value_issues.append("weights must be positive")
            if weights.size and not np.isclose(np.sum(weights), 1.0):
                value_issues.append("weights must sum to one")
            issues = shape_issues + value_issues
            if not issues:
                status = "ok"
            elif shape_issues:
                status = "invalid_shape"
            else:
                status = "invalid_values"
            summary = {
                "status": status,
                "datasets": datasets,
                "numeric_summary": numeric_summary,
            }
            if issues:
                summary["issues"] = issues
            if shape_issues:
                summary["shape_issues"] = shape_issues
            if value_issues:
                summary["value_issues"] = value_issues
            return summary
    except OSError as exc:
        return {"status": "unreadable", "error": str(exc)}


def ckd_table_schema_problem_text(schema):
    """Return a compact human-readable explanation for a bad CKD h5 schema."""
    if schema.get("status") == "ok":
        return ""
    if schema.get("missing_datasets"):
        return "missing datasets: " + ", ".join(schema["missing_datasets"])
    if schema.get("issues"):
        return "; ".join(schema["issues"])
    if schema.get("error"):
        return schema["error"]
    return ""


def format_ckd_schema_status(schema):
    """Return a concise status string with schema issue detail when available."""
    detail = ckd_table_schema_problem_text(schema)
    if detail:
        return f"{schema['status']} ({detail})"
    return schema["status"]


def ckd_download_candidate_tables(directory):
    """Return default ExoMolOP/petitRADTRANS download candidate table paths."""
    from exojax.provider.url import petitRADTRANS_ktable_filenames

    exact_molecule_name = os.path.basename(os.path.dirname(directory))
    database = os.path.basename(directory)
    return [
        os.path.join(directory, filename)
        for filename in petitRADTRANS_ktable_filenames(exact_molecule_name, database)
    ]


def premodit_snapshot_candidates(mol):
    """Return supported preMODIT snapshot path candidates for a molecule."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    names = (f"opa_{mol}.zarr", f"opa{mol}.zarr")
    candidates = []
    for base_dir in ("", script_dir):
        for name in names:
            path = os.path.join(base_dir, name) if base_dir else name
            if path not in candidates:
                candidates.append(path)
    return candidates


def resolve_premodit_snapshot_path(mol):
    """Return the first existing preMODIT snapshot path for a molecule."""
    for path in premodit_snapshot_candidates(mol):
        if os.path.exists(path):
            return path
    return None


def configure_jax_platform():
    """Apply a JAX platform override before importing JAX."""
    if args.jax_platform == "auto":
        return
    if "JAX_PLATFORMS" in os.environ and os.environ["JAX_PLATFORMS"] != args.jax_platform:
        print(
            "JAX_PLATFORMS is already set to "
            f"{os.environ['JAX_PLATFORMS']}; --jax-platform {args.jax_platform} "
            "will not override it."
        )
        return
    os.environ["JAX_PLATFORMS"] = args.jax_platform
    if args.jax_platform == "cpu":
        # Some environments install a CUDA JAX plugin without visible GPUs. JAX
        # still discovers that plugin on import, so keep explicit CPU runs quiet.
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)


def configure_matplotlib_cache():
    """Use a writable Matplotlib cache when the default user config is blocked."""
    if os.environ.get("MPLCONFIGDIR"):
        return
    default_config_dir = os.path.join(os.path.expanduser("~"), ".config", "matplotlib")
    if os.path.isdir(default_config_dir):
        if os.access(default_config_dir, os.W_OK):
            return
    else:
        parent_dir = os.path.dirname(default_config_dir)
        if os.path.isdir(parent_dir) and os.access(parent_dir, os.W_OK):
            return

    fallback_dir = os.path.join(tempfile.gettempdir(), "exojax_matplotlib")
    try:
        os.makedirs(fallback_dir, exist_ok=True)
    except OSError:
        return
    os.environ["MPLCONFIGDIR"] = fallback_dir


def validate_local_inputs_before_heavy_imports():
    """Fail before importing JAX/ExoJAX when local required files are missing."""
    if args.plot_data_only or args.summarize_data or args.check_inputs:
        return

    missing = []
    for label, path in ciapath_list.items():
        if not os.path.exists(path):
            missing.append(f"CIA {label}: {path}")

    if args.opacity_mode == "ckd":
        for mol in selected_molecules:
            path = ckd_source_path(mol)
            if mol in selected_ckd_table_paths:
                if not os.path.isfile(path):
                    missing.append(f"ExoMolOP CKD {mol} h5 table: {path}")
                elif not path.endswith(".h5"):
                    missing.append(f"ExoMolOP CKD {mol} h5 table is not .h5: {path}")
                elif os.path.getsize(path) == 0:
                    missing.append(f"ExoMolOP CKD {mol} h5 table is empty: {path}")
                else:
                    schema = ckd_table_schema_summary(path)
                    if schema["status"] != "ok":
                        missing.append(
                            f"ExoMolOP CKD {mol} h5 schema is "
                            f"{format_ckd_schema_status(schema)}: {path}"
                        )
                continue
            if not os.path.isdir(path):
                if args.allow_ckd_download:
                    continue
                missing.append(f"ExoMolOP CKD {mol}: {path}")
                continue
            h5_paths = ckd_h5_paths(path)
            nonempty_h5_paths = [
                h5_path for h5_path in h5_paths if os.path.getsize(h5_path) > 0
            ]
            if not nonempty_h5_paths:
                if args.allow_ckd_download:
                    continue
                if h5_paths:
                    missing.append(f"ExoMolOP CKD {mol} h5 table is empty: {path}/*.h5")
                else:
                    missing.append(f"ExoMolOP CKD {mol} h5 table: {path}/*.h5")
            elif len(nonempty_h5_paths) > 1:
                missing.append(
                    "ExoMolOP CKD "
                    f"{mol} h5 table is ambiguous: {path}/*.h5 "
                    f"({len(nonempty_h5_paths)} non-empty files; "
                    "specify a directory with one table)"
                )
            else:
                schema = ckd_table_schema_summary(nonempty_h5_paths[0])
                if schema["status"] != "ok":
                    missing.append(
                        f"ExoMolOP CKD {mol} h5 schema is "
                        f"{format_ckd_schema_status(schema)}: {nonempty_h5_paths[0]}"
                    )

    if missing:
        message = (
            "Required local inputs are missing:\n  "
            + "\n  ".join(missing)
            + "\nRun with --check-inputs to inspect expected paths. "
            + "Use --allow-ckd-download to permit CKD downloads."
        )
        parser.exit(status=1, message=message + "\n")


validate_local_inputs_before_heavy_imports()
configure_jax_platform()

needs_plotting = args.plot_data_only or (
    not (args.summarize_data or args.check_inputs or args.check_forward)
    and (not args.skip_data_plot or not args.skip_diagnostic_plots)
)

if needs_plotting:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

if not (args.plot_data_only or args.summarize_data or args.check_inputs):
    import jax
    from jax import random
    import jax.numpy as jnp
    from contextlib import redirect_stdout

    import exojax
    if args.check_forward or args.skip_diagnostic_plots:
        corner = None
    else:
        try:
            import corner
        except ModuleNotFoundError:
            corner = None

    from exojax.rt import ArtTransPure
    from exojax.utils.constants import RJ, Rs, MJ
    from exojax.utils.astrofunc import gravity_jupiter
    from exojax.utils.grids import wav2nu

    from exojax.postproc.ckd import sample_ckd_bands_at_wavelengths
    from exojax.postproc.ckd import validate_ckd_band_coverage
    from exojax.postproc.ckd import validate_ckd_sampling_inputs
    from exojax.postproc.ckd import wavenumber_range_with_radial_velocity

    from exojax.database.cia.api import CdbCIA
    from exojax.opacity.opacont import OpaCIA
    from exojax.database import molinfo
    if args.opacity_mode == "ckd":
        from exojax.opacity import OpaCKD
    else:
        from astropy.io import fits
        from exojax.postproc.specop import SopRotation, SopInstProfile
        from exojax.utils.instfunc import resolution_to_gaussian_std
        from exojax.utils.grids import wavenumber_grid
        from exojax.database.hitemp.api import MdbHitemp
        from exojax.database.exomol.api import MdbExomol
        from exojax.opacity.premodit.api import OpaPremodit
        from exojax.opacity import saveopa

    if not args.check_forward:
        # --- Probabilistic Programming imports -------------------------------------
        from numpyro.infer import Predictive, MCMC, NUTS, SVI, Trace_ELBO
        import numpyro
        import numpyro.distributions as dist
        import numpyro.optim as optim
        from numpyro import handlers
        from numpyro.infer.autoguide import AutoMultivariateNormal
        from numpyro.infer.initialization import init_to_value
else:
    corner = None

# sphinx_gallery_thumbnail_path = '_static/transit.png'


@dataclass(frozen=True)
class ObservedSpectrum:
    """Observed transmission spectrum for one instrument channel."""

    name: str
    wavelength_nm: np.ndarray
    radius_ratio: np.ndarray
    radius_ratio_error_low: np.ndarray
    radius_ratio_error_high: np.ndarray
    plot_alpha: float

    @property
    def radius_ratio_error(self) -> np.ndarray:
        """Return a symmetric uncertainty approximation."""
        return 0.5 * (self.radius_ratio_error_low + self.radius_ratio_error_high)

    @property
    def yerr_for_plot(self):
        """Return scalar or asymmetric uncertainty in Matplotlib format."""
        if np.allclose(self.radius_ratio_error_low, self.radius_ratio_error_high):
            return self.radius_ratio_error
        return [self.radius_ratio_error_low, self.radius_ratio_error_high]


def load_observed_spectra(data_dir="wasp39_data"):
    """Load the NIRISS, NIRSpec, and MIRI WASP-39b transmission spectra."""
    import h5py

    data_dir = os.fspath(data_dir)

    wav_niriss_1, rp_niriss_1, err_low_niriss_1, err_high_niriss_1 = np.loadtxt(
        os.path.join(data_dir, "niriss_order1.txt"), unpack=True
    )
    wav_niriss_2, rp_niriss_2, err_low_niriss_2, err_high_niriss_2 = np.loadtxt(
        os.path.join(data_dir, "niriss_order2.txt"), unpack=True
    )

    wav_nirspec = np.load(os.path.join(data_dir, "wavelength.npy"))
    rp_nirspec = np.load(os.path.join(data_dir, "wasp39b_nirspec_g395h_rp_mean.npy"))
    err_nirspec = np.load(os.path.join(data_dir, "wasp39b_nirspec_g395h_rp_std.npy"))

    with h5py.File(os.path.join(data_dir, "miri.h5"), "r") as f:
        dppm = np.array(f["dppm"])
        dppm_err = np.array(f["dppm_error"])
        wav_miri_micron = np.array(f["wavelength"])

    ppmtor = 1e-6
    microntonm = 1e3
    wav_miri = wav_miri_micron * microntonm
    rp_miri = np.sqrt(dppm * ppmtor)
    err_miri = dppm_err * ppmtor / (2 * np.sqrt(dppm * ppmtor))

    return {
        "niriss_order1": ObservedSpectrum(
            "NIRISS Order 1",
            wav_niriss_1,
            rp_niriss_1,
            err_low_niriss_1,
            err_high_niriss_1,
            0.3,
        ),
        "niriss_order2": ObservedSpectrum(
            "NIRISS Order 2",
            wav_niriss_2,
            rp_niriss_2,
            err_low_niriss_2,
            err_high_niriss_2,
            0.3,
        ),
        "nirspec_g395h": ObservedSpectrum(
            "NIRSpec G395H",
            wav_nirspec,
            rp_nirspec,
            err_nirspec,
            err_nirspec,
            0.3,
        ),
        "miri_lrs": ObservedSpectrum(
            "MIRI",
            wav_miri,
            rp_miri,
            err_miri,
            err_miri,
            0.9,
        ),
    }


def expected_input_paths(
    data_dir,
    opacity_root,
    ckd_root,
    molecules,
    channels,
    cia_pairs,
    ckd_table_paths,
):
    """Return expected data and opacity paths used by this example."""
    data_file_map = {
        "niriss_order1": {"niriss_order1": os.path.join(data_dir, "niriss_order1.txt")},
        "niriss_order2": {"niriss_order2": os.path.join(data_dir, "niriss_order2.txt")},
        "nirspec_g395h": {
            "nirspec_wavelength": os.path.join(data_dir, "wavelength.npy"),
            "nirspec_rp_mean": os.path.join(
                data_dir, "wasp39b_nirspec_g395h_rp_mean.npy"
            ),
            "nirspec_rp_std": os.path.join(
                data_dir, "wasp39b_nirspec_g395h_rp_std.npy"
            ),
            "nirspec_resolution": os.path.join(
                data_dir, "jwst_nirspec_g395h_disp.fits"
            ),
        },
        "miri_lrs": {"miri": os.path.join(data_dir, "miri.h5")},
    }
    data_files = {}
    for channel in channels:
        data_files.update(data_file_map[channel])
    cia_file_map = {
        pair: os.path.join(opacity_root, relative_path)
        for pair, relative_path in CIA_RELATIVE_FILES.items()
    }
    cia_files = {pair: cia_file_map[pair] for pair in cia_pairs}
    database_dirs = {
        "hitemp": os.path.join(opacity_root, ".db_HITEMP"),
        "exomol": os.path.join(opacity_root, ".db_ExoMol"),
        "ckd": ckd_root,
    }
    ckd_dir_map = {
        mol: os.path.join(ckd_root, relative_path)
        for mol, relative_path in CKD_RELATIVE_PATHS.items()
    }
    ckd_dirs = {mol: ckd_dir_map[mol] for mol in molecules}
    ckd_table_map = {
        mol: ckd_table_paths.get(mol, os.path.join(ckd_dirs[mol], "*.h5"))
        for mol in molecules
    }
    return data_files, cia_files, database_dirs, ckd_dirs, ckd_table_map


def print_input_status(
    data_dir,
    opacity_root,
    ckd_root,
    molecules,
    channels,
    cia_pairs,
    ckd_table_paths,
):
    """Print whether expected input files and directories exist."""
    data_files, cia_files, database_dirs, ckd_dirs, ckd_table_map = expected_input_paths(
        data_dir,
        opacity_root,
        ckd_root,
        molecules,
        channels,
        cia_pairs,
        ckd_table_paths,
    )

    print("Observed data files:")
    for key, path in data_files.items():
        print(f"  {key}: {path} ({'ok' if os.path.exists(path) else 'missing'})")

    print("CIA files:")
    for key, path in cia_files.items():
        print(f"  {key}: {path} ({'ok' if os.path.exists(path) else 'missing'})")

    print("Opacity roots:")
    for key, path in database_dirs.items():
        print(f"  {key}: {path} ({'ok' if os.path.isdir(path) else 'missing'})")

    print("ExoMolOP CKD sources:")
    for key, path in ckd_dirs.items():
        table_path = ckd_table_map[key]
        directory_exists = os.path.isdir(path)
        schema_path = None
        if key in ckd_table_paths:
            print(f"  {key} directory: {path} (not used; explicit table override)")
            h5_status = "ok" if os.path.isfile(table_path) else "missing"
            if h5_status == "ok" and not table_path.endswith(".h5"):
                h5_status = "not .h5"
            elif h5_status == "ok" and os.path.getsize(table_path) == 0:
                h5_status = "empty"
            elif h5_status == "ok":
                schema_path = table_path
            h5_status += "; explicit"
        elif directory_exists:
            print(f"  {key} directory: {path} ({'ok' if directory_exists else 'missing'})")
            h5_paths = ckd_h5_paths(path)
            nonempty_h5_paths = [
                h5_path for h5_path in h5_paths if os.path.getsize(h5_path) > 0
            ]
            if not h5_paths:
                h5_status = "missing"
            elif not nonempty_h5_paths:
                h5_status = "empty"
            elif len(nonempty_h5_paths) == 1:
                h5_status = "ok"
                schema_path = nonempty_h5_paths[0]
            else:
                h5_status = (
                    f"multiple ({len(nonempty_h5_paths)} non-empty); "
                    "specify a directory with one table"
                )
        else:
            print(f"  {key} directory: {path} ({'ok' if directory_exists else 'missing'})")
            h5_status = "not checked"
        print(f"  {key} h5 table: {table_path} ({h5_status})")
        if key not in ckd_table_paths and h5_status in (
            "missing",
            "empty",
            "not checked",
        ):
            print(f"  {key} download candidates:")
            for candidate in ckd_download_candidate_tables(path):
                print(f"    {candidate}")
        if schema_path is not None:
            schema = ckd_table_schema_summary(schema_path)
            print(f"  {key} h5 schema: {format_ckd_schema_status(schema)}")


def input_status_payload(
    data_dir,
    opacity_root,
    ckd_root,
    molecules,
    channels,
    cia_pairs,
    ckd_table_paths,
):
    """Return machine-readable input status for preflight artifacts."""
    data_files, cia_files, database_dirs, ckd_dirs, ckd_table_map = expected_input_paths(
        data_dir,
        opacity_root,
        ckd_root,
        molecules,
        channels,
        cia_pairs,
        ckd_table_paths,
    )

    def file_entries(paths):
        return {
            key: {"path": path, "status": "ok" if os.path.exists(path) else "missing"}
            for key, path in paths.items()
        }

    def directory_entries(paths):
        return {
            key: {"path": path, "status": "ok" if os.path.isdir(path) else "missing"}
            for key, path in paths.items()
        }

    problems = []
    observed_data_files = file_entries(data_files)
    cia_file_entries = file_entries(cia_files)
    opacity_root_entries = directory_entries(database_dirs)
    premodit_snapshots = {
        mol: {"candidates": premodit_snapshot_candidates(mol)}
        for mol in molecules
    }
    for mol, entry in premodit_snapshots.items():
        resolved_path = resolve_premodit_snapshot_path(mol)
        entry["path"] = resolved_path or premodit_snapshot_candidates(mol)[0]
        entry["status"] = "ok" if resolved_path else "missing"
        if resolved_path:
            entry["resolved_path"] = resolved_path
    for key, entry in observed_data_files.items():
        if entry["status"] != "ok":
            problems.append(f"observed_data_files.{key}: {entry['status']}")
    for key, entry in cia_file_entries.items():
        if entry["status"] != "ok":
            problems.append(f"cia_files.{key}: {entry['status']}")
    if args.opacity_mode != "ckd" and opa_load:
        for key, entry in premodit_snapshots.items():
            if entry["status"] != "ok":
                problems.append(f"premodit_snapshots.{key}: {entry['status']}")

    ckd_sources = {}
    for mol, directory in ckd_dirs.items():
        table_path = ckd_table_map[mol]
        explicit = mol in ckd_table_paths
        directory_exists = os.path.isdir(directory)
        source = {
            "directory": directory,
            "directory_status": "ok" if directory_exists else "missing",
            "table": table_path,
            "explicit_table": explicit,
            "download_required": False,
        }
        if not explicit:
            source["download_candidate_tables"] = ckd_download_candidate_tables(
                directory
            )
        if explicit:
            table_status = "ok" if os.path.isfile(table_path) else "missing"
            if table_status == "ok" and not table_path.endswith(".h5"):
                table_status = "not_h5"
            if table_status == "ok" and os.path.getsize(table_path) == 0:
                table_status = "empty"
            source["directory_status"] = "not_used_explicit_table"
            source["table_status"] = table_status
            if table_status == "ok":
                source["resolved_table"] = table_path
                source["resolved_table_metadata"] = ckd_table_file_metadata(table_path)
                source["table_schema"] = ckd_table_schema_summary(table_path)
        elif directory_exists:
            h5_paths = ckd_h5_paths(directory)
            nonempty_h5_paths = [
                h5_path for h5_path in h5_paths if os.path.getsize(h5_path) > 0
            ]
            source["h5_count"] = len(h5_paths)
            source["nonempty_h5_count"] = len(nonempty_h5_paths)
            if not h5_paths:
                source["table_status"] = "missing"
                if args.allow_ckd_download:
                    source["download_required"] = True
                    source["download_target"] = directory
            elif not nonempty_h5_paths:
                source["table_status"] = "empty"
                source["h5_candidates"] = h5_paths
                if args.allow_ckd_download:
                    source["download_required"] = True
                    source["download_target"] = directory
            elif len(nonempty_h5_paths) == 1:
                metadata = ckd_table_file_metadata(nonempty_h5_paths[0])
                source["table_status"] = "ok"
                source["resolved_table"] = nonempty_h5_paths[0]
                source["resolved_table_metadata"] = metadata
                source["table_schema"] = ckd_table_schema_summary(nonempty_h5_paths[0])
                empty_h5_paths = sorted(set(h5_paths) - set(nonempty_h5_paths))
                if empty_h5_paths:
                    source["ignored_empty_h5_candidates"] = empty_h5_paths
            else:
                source["table_status"] = "multiple"
                source["h5_candidates"] = nonempty_h5_paths
        else:
            source["table_status"] = "not_checked"
            if args.allow_ckd_download and not explicit:
                source["download_required"] = True
                source["download_target"] = directory
        if args.opacity_mode == "ckd":
            table_schema = source.get("table_schema")
            if explicit:
                if source["table_status"] != "ok":
                    problems.append(
                        f"ckd_sources.{mol}.table: {source['table_status']}"
                    )
                elif table_schema is not None and table_schema["status"] != "ok":
                    problems.append(
                        f"ckd_sources.{mol}.schema: {table_schema['status']}"
                    )
            elif source["table_status"] == "multiple":
                problems.append(f"ckd_sources.{mol}.table: {source['table_status']}")
            elif table_schema is not None and table_schema["status"] != "ok":
                problems.append(f"ckd_sources.{mol}.schema: {table_schema['status']}")
            elif not args.allow_ckd_download:
                if source["directory_status"] != "ok":
                    problems.append(
                        f"ckd_sources.{mol}.directory: "
                        f"{source['directory_status']}"
                    )
                elif source["table_status"] != "ok":
                    problems.append(
                        f"ckd_sources.{mol}.table: {source['table_status']}"
                    )
        ckd_sources[mol] = source

    return {
        "ready_for_local_run": not problems,
        "problems": problems,
        "selections": {
            "data_mode": args.data_mode,
            "opacity_mode": args.opacity_mode,
            "molecules": list(molecules),
            "channels": list(channels),
            "cia_pairs": list(cia_pairs),
            "allow_ckd_download": bool(args.allow_ckd_download),
        },
        "observed_data_files": observed_data_files,
        "cia_files": cia_file_entries,
        "opacity_roots": opacity_root_entries,
        "premodit_snapshots": premodit_snapshots,
        "ckd_sources": ckd_sources,
    }


def save_input_status_json(path, payload):
    """Save --check-inputs status JSON."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Input status JSON saved to {path}")


def save_json(path, payload, label):
    """Save a JSON artifact."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"{label} saved to {path}")


def plot_observed_spectra(observed_spectra, save_path):
    """Plot all observed WASP-39b transmission spectra."""
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)

    for spectrum in observed_spectra.values():
        plt.errorbar(
            spectrum.wavelength_nm,
            spectrum.radius_ratio,
            yerr=spectrum.yerr_for_plot,
            fmt=".",
            label=spectrum.name,
            alpha=spectrum.plot_alpha,
        )

    plt.xscale("log")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Rp/Rs")
    plt.legend()
    plt.title("WASP-39b Transmission Spectrum")
    plt.savefig(save_path, dpi=200)
    plt.close()


def concatenate_observed_spectra(observed_spectra):
    """Return all observed spectra as wavelength-sorted arrays."""
    wavelength_nm = np.concatenate(
        [spectrum.wavelength_nm for spectrum in observed_spectra.values()]
    )
    radius_ratio = np.concatenate(
        [spectrum.radius_ratio for spectrum in observed_spectra.values()]
    )
    radius_ratio_error = np.concatenate(
        [spectrum.radius_ratio_error for spectrum in observed_spectra.values()]
    )
    channel = np.concatenate(
        [
            np.full(spectrum.wavelength_nm.shape, i, dtype=int)
            for i, spectrum in enumerate(observed_spectra.values())
        ]
    )

    sort_index = np.argsort(wavelength_nm)
    return (
        wavelength_nm[sort_index],
        radius_ratio[sort_index],
        radius_ratio_error[sort_index],
        channel[sort_index],
    )


def select_observed_channels(observed_spectra, channels):
    """Return observed spectra restricted to requested channels."""
    return {channel: observed_spectra[channel] for channel in channels}


def summarize_observed_spectra(observed_spectra, wavelength_nm_all):
    """Print a compact summary of the observed spectra."""
    print("Observed WASP-39b spectra:")
    for key, spectrum in observed_spectra.items():
        print(
            f"  {key}: n={spectrum.wavelength_nm.size}, "
            f"wavelength=[{np.min(spectrum.wavelength_nm):.3f}, "
            f"{np.max(spectrum.wavelength_nm):.3f}] nm"
        )
    print(
        f"  combined: n={wavelength_nm_all.size}, "
        f"wavelength=[{np.min(wavelength_nm_all):.3f}, "
        f"{np.max(wavelength_nm_all):.3f}] nm"
    )


def observed_channel_summary(observed_spectra):
    """Return per-channel observed-data metadata for saved run configuration."""
    return {
        key: {
            "name": spectrum.name,
            "index": int(i),
            "n_observed": int(spectrum.wavelength_nm.size),
            "wavelength_nm_min": float(np.min(spectrum.wavelength_nm)),
            "wavelength_nm_max": float(np.max(spectrum.wavelength_nm)),
        }
        for i, (key, spectrum) in enumerate(observed_spectra.items())
    }


def retrieval_channel_summary(channels, channel_index, wavelength_nm):
    """Return per-channel metadata for the selected retrieval vector."""
    summary = {}
    for index, channel in enumerate(channels):
        mask = channel_index == index
        n_selected = int(np.sum(mask))
        channel_summary = {"index": int(index), "n_selected": n_selected}
        if n_selected:
            channel_wavelength = wavelength_nm[mask]
            channel_summary.update(
                {
                    "wavelength_nm_min": float(np.min(channel_wavelength)),
                    "wavelength_nm_max": float(np.max(channel_wavelength)),
                }
            )
        else:
            channel_summary.update(
                {"wavelength_nm_min": None, "wavelength_nm_max": None}
            )
        summary[channel] = channel_summary
    return summary


def ckd_band_summary():
    """Return metadata for loaded CKD bands."""
    if ckd_nu_bands is None:
        return None
    summary = {
        "n_bands": int(np.asarray(ckd_nu_bands).size),
        "n_g": int(np.asarray(ckd_weights).size),
        "nu_band_min_cm-1": float(np.min(np.asarray(ckd_nu_bands))),
        "nu_band_max_cm-1": float(np.max(np.asarray(ckd_nu_bands))),
    }
    if ckd_reference.band_edges is not None:
        band_edges = np.asarray(ckd_reference.band_edges)
        summary.update(
            {
                "nu_edge_min_cm-1": float(np.min(band_edges)),
                "nu_edge_max_cm-1": float(np.max(band_edges)),
            }
        )
    return summary


def ckd_table_summary():
    """Return molecule-level metadata for loaded CKD tables."""
    if args.opacity_mode != "ckd":
        return {}
    summary = {}
    for mol, opa in opa_mols.items():
        info = opa.ckd_info
        temperatures = np.asarray(info.T_grid)
        pressures = np.asarray(info.P_grid)
        ggrid = np.asarray(info.ggrid)
        weights = np.asarray(info.weights)
        nu_bands = np.asarray(info.nu_bands)
        band_edges = np.asarray(info.band_edges)
        summary[mol] = {
            "source": ckdpath_list_ExomolOP[mol],
            "molmass": float(opa.molmass),
            "n_temperature": int(temperatures.size),
            "temperature_min_K": float(np.min(temperatures)),
            "temperature_max_K": float(np.max(temperatures)),
            "n_pressure": int(pressures.size),
            "pressure_min_bar": float(np.min(pressures)),
            "pressure_max_bar": float(np.max(pressures)),
            "n_g": int(ggrid.size),
            "g_min": float(np.min(ggrid)),
            "g_max": float(np.max(ggrid)),
            "weight_sum": float(np.sum(weights)),
            "n_bands": int(nu_bands.size),
            "nu_band_min_cm-1": float(np.min(nu_bands)),
            "nu_band_max_cm-1": float(np.max(nu_bands)),
            "nu_edge_min_cm-1": float(np.min(band_edges)),
            "nu_edge_max_cm-1": float(np.max(band_edges)),
        }
    return summary


def select_retrieval_data(data_mode, nirspec_spectrum, all_wavelength, all_rp, all_err):
    """Return the observed data vector requested by the retrieval mode."""
    if data_mode == "nirspec":
        return (
            nirspec_spectrum.wavelength_nm,
            nirspec_spectrum.radius_ratio,
            nirspec_spectrum.radius_ratio_error,
        )
    if data_mode == "wide":
        return all_wavelength, all_rp, all_err
    raise ValueError(f"Unknown data_mode: {data_mode}")


def validate_observed_data_vector(label, wavelength, radius_ratio, radius_ratio_error):
    """Validate the observed data vector before forward/HMC execution."""
    wavelength = np.asarray(wavelength)
    radius_ratio = np.asarray(radius_ratio)
    radius_ratio_error = np.asarray(radius_ratio_error)
    if (
        wavelength.ndim != 1
        or radius_ratio.ndim != 1
        or radius_ratio_error.ndim != 1
    ):
        raise ValueError(f"{label} arrays must be one-dimensional.")
    if not (
        wavelength.shape == radius_ratio.shape == radius_ratio_error.shape
    ):
        raise ValueError(f"{label} wavelength, radius ratio, and error shapes differ.")
    if wavelength.size == 0:
        raise ValueError(f"{label} must contain at least one data point.")
    if not (
        np.all(np.isfinite(wavelength))
        and np.all(np.isfinite(radius_ratio))
        and np.all(np.isfinite(radius_ratio_error))
    ):
        raise ValueError(f"{label} arrays must contain finite values.")
    if not np.all(wavelength > 0.0):
        raise ValueError(f"{label} wavelengths must be positive.")
    if not np.all(radius_ratio > 0.0):
        raise ValueError(f"{label} radius ratios must be positive.")
    if not np.all(radius_ratio_error > 0.0):
        raise ValueError(f"{label} uncertainties must be positive.")


def cap_observed_data(wavelength, radius_ratio, radius_ratio_error, channel_index, limit):
    """Return an evenly spaced subset of the selected observed data vector."""
    if limit is None or limit >= wavelength.size:
        return wavelength, radius_ratio, radius_ratio_error, channel_index
    indices = np.unique(np.linspace(0, wavelength.size - 1, limit, dtype=int))
    return (
        wavelength[indices],
        radius_ratio[indices],
        radius_ratio_error[indices],
        channel_index[indices],
    )


def observed_wav2nu(wavelength, unit):
    """Convert observed wavelengths to wavenumbers without expected order noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Both input wavelength and output wavenumber are in ascending order.",
            category=UserWarning,
        )
        return wav2nu(wavelength, unit)


# %%
# Setup and configuration
# -------------------------------------------------------------------

# Planet–star system parameters and orbital period (days)
period_day = 4.05528
Mp_mean, Mp_std = 0.281, 0.032  # [M_J]
Rstar_mean, Rstar_std = 0.939, 0.022  # [R_Sun]

# Output directory
DIR_SAVE = args.output_dir

# Opacity loading flag: set to True to load precomputed opacities
opa_load = True
# Opacity saving flag: set to True to save computed opacities
opa_save = False


# %%
# Load observed transmission spectrum
# -----------------------------------

if args.check_inputs:
    status_payload = input_status_payload(
        args.data_dir,
        args.opacity_root,
        args.ckd_root,
        selected_molecules,
        selected_channels,
        selected_cia_pairs,
        selected_ckd_table_paths,
    )
    print_input_status(
        args.data_dir,
        args.opacity_root,
        args.ckd_root,
        selected_molecules,
        selected_channels,
        selected_cia_pairs,
        selected_ckd_table_paths,
    )
    if args.input_status_json:
        save_input_status_json(args.input_status_json, status_payload)
    sys.exit(0)

observed_spectra = load_observed_spectra(args.data_dir)
observed_spectra = select_observed_channels(observed_spectra, selected_channels)
wav_obs_all, rp_mean_all, rp_std_all, channel_index_all = concatenate_observed_spectra(
    observed_spectra
)
validate_observed_data_vector(
    "selected observed data", wav_obs_all, rp_mean_all, rp_std_all
)

if args.summarize_data:
    summarize_observed_spectra(observed_spectra, wav_obs_all)
    print("Selected molecules: " + format_selection(selected_molecules))
    print("Selected channels: " + format_selection(selected_channels))
    print("Selected CIA pairs: " + format_selection(selected_cia_pairs))
    sys.exit(0)
if args.plot_data_only:
    plot_observed_spectra(observed_spectra, args.data_plot_path)
    sys.exit(0)

niriss_order1 = observed_spectra.get("niriss_order1")
niriss_order2 = observed_spectra.get("niriss_order2")
nirspec_g395h = observed_spectra.get("nirspec_g395h")
miri_lrs = observed_spectra.get("miri_lrs")

wav_obs_fit, rp_mean_fit, rp_std_fit = select_retrieval_data(
    args.data_mode, nirspec_g395h, wav_obs_all, rp_mean_all, rp_std_all
)
if args.data_mode == "wide":
    channel_index_fit = channel_index_all
else:
    channel_index_fit = np.full(
        wav_obs_fit.shape, selected_channels.index("nirspec_g395h"), dtype=int
    )

wav_obs_fit, rp_mean_fit, rp_std_fit, channel_index_fit = cap_observed_data(
    wav_obs_fit,
    rp_mean_fit,
    rp_std_fit,
    channel_index_fit,
    args.max_observed,
)
validate_observed_data_vector(
    "retrieval observed data", wav_obs_fit, rp_mean_fit, rp_std_fit
)

wav_obs_nirspec = None
rp_mean_nirspec = None
rp_std_nirspec = None
if nirspec_g395h is not None:
    if args.data_mode == "nirspec":
        wav_obs_nirspec = wav_obs_fit
        rp_mean_nirspec = rp_mean_fit
        rp_std_nirspec = rp_std_fit
    else:
        wav_obs_nirspec = nirspec_g395h.wavelength_nm
        rp_mean_nirspec = nirspec_g395h.radius_ratio
        rp_std_nirspec = nirspec_g395h.radius_ratio_error


# Convert from wavelength to wavenumber for modelling.
if not args.plot_data_only:
    inst_fit_nus = observed_wav2nu(wav_obs_fit, "nm")
    if args.opacity_mode != "ckd":
        inst_nirspec_nus = observed_wav2nu(wav_obs_nirspec, "nm")
        ckd_nurange = None
    else:
        ckd_nurange = wavenumber_range_with_radial_velocity(
            inst_fit_nus, args.rv_min, args.rv_max
        )


# Plot the input data before building the heavier opacity objects.
if not args.check_forward and not args.skip_data_plot:
    plot_observed_spectra(observed_spectra, args.data_plot_path)


# %%
# Instrumental resolution
# -------------------------
#
# Read the NIRSpec/G395H resolving-power curve and interpolate it so the
# forward model can convert to a Gaussian instrumental broadening kernel.


def load_resolution_curve():
    """Load and cache the NIRSpec/G395H resolution curve from the FITS table."""
    with fits.open(os.path.join(args.data_dir, "jwst_nirspec_g395h_disp.fits")) as hdul:
        data = np.asarray([list(row) for row in hdul[1].data])
    return data


if args.opacity_mode != "ckd":
    _res_curve = load_resolution_curve()


    def res_G395H(wavelength_nm: float) -> float:
        """Return the resolving power *R* of JWST NIRSpec/G395H at *wavelength_nm*."""
        return np.interp(wavelength_nm / 1000.0, _res_curve[:, 0], _res_curve[:, 2])


    Rinst = res_G395H(np.mean(wav_obs_nirspec))


# %%
# Wavenumber grid and spectral operators for NIRSPEC
# ------------------------------------------------------
#
# Build a high-resolution wavenumber grid for forward modelling and construct
# spectral operators to mimic rotation and the NIRSpec line-spread function.

if args.opacity_mode != "ckd":
    N_nirspec = 30_000  # spectral points; lower for faster demo
    nu_grid_nirspec, wav_grid_nirspec, res_high_nirspec = wavenumber_grid(
        np.min(wav_obs_nirspec) - 15,
        np.max(wav_obs_nirspec) + 15,
        N=N_nirspec,
        unit="nm",
        xsmode="premodit",
    )
    print(f"wavenumber grid: R={res_high_nirspec:.0f}")

    beta_inst_nirspec = resolution_to_gaussian_std(Rinst)
    sop_rot_nirspec = SopRotation(nu_grid_nirspec, vsini_max=100.0)
    sop_inst_nirspec = SopInstProfile(nu_grid_nirspec, vrmax=300.0)

# %%
# Atmospheric radiative‑transfer object
# -------------------------------------

diffmode = 0
nlayer = 120  # number of layers in the atmosphere
pressure_top, pressure_btm = 1e-11, 1e1  # [bar]
art = ArtTransPure(
    pressure_top=pressure_top,
    pressure_btm=pressure_btm,
    nlayer=nlayer,
    nu_grid=None if args.opacity_mode == "ckd" else nu_grid_nirspec,
    warn_no_nu_grid=args.opacity_mode != "ckd",
)
Tlow, Thigh = 500.0, 2000.0
art.change_temperature_range(Tlow, Thigh)

# %%
# Opacity sources
# ---------------
#
# Load collision-induced absorption (CIA) and line opacities. The script
# prefers saved preMODIT snapshots (``opa_*.zarr``); if missing and
# ``opa_load=False``, it will build them from HITEMP/ExoMol databases and save
# them for reuse.

def build_cia_opacities(nu_grid):
    """Build CIA opacity calculators on the requested wavenumber grid."""
    return {
        name: OpaCIA(CdbCIA(path, nurange=nu_grid), nu_grid=nu_grid)
        for name, path in ciapath_list.items()
    }

# Line absorption (HITEMP + ExoMol)
db_HITEMP = os.path.join(args.opacity_root, ".db_HITEMP")
db_ExoMol = os.path.join(args.opacity_root, ".db_ExoMol")

molpath_list_HITEMP = {
    "H2O": os.path.join(db_HITEMP, "H2O"),
    "CO": os.path.join(db_HITEMP, "CO"),
    "CO2": os.path.join(db_HITEMP, "CO2"),
    # "CH4": os.path.join(db_HITEMP, "CH4"),
}

molpath_list_Exomol = {
    # "NH3": os.path.join(db_ExoMol, "NH3", "14N-1H3", "CoYuTe"),
    "H2S": os.path.join(db_ExoMol, "H2S", "1H2-32S", "AYT2"),
    # "HCN": os.path.join(db_ExoMol, "HCN", "1H-12C-14N", "Harris"),
    # "C2H2": os.path.join(db_ExoMol, "C2H2", "12C2-1H2", "aCeTY"),
    "SO2": os.path.join(db_ExoMol, "SO2", "32S-16O2", "ExoAmes"),
    "SiO": os.path.join(db_ExoMol, "SiO", "28Si-16O", "SiOUVenIR"),
}

ckdpath_map_ExomolOP = {mol: ckd_resolved_source_path(mol) for mol in DEFAULT_MOLECULES}
molpath_list_HITEMP = {
    mol: path for mol, path in molpath_list_HITEMP.items() if mol in selected_molecules
}
molpath_list_Exomol = {
    mol: path for mol, path in molpath_list_Exomol.items() if mol in selected_molecules
}
ckdpath_list_ExomolOP = {
    mol: ckdpath_map_ExomolOP[mol] for mol in selected_molecules
}

ndiv = 6  # preMODIT stitch blocks


def validate_required_inputs():
    """Raise a clear error when local files required for retrieval are missing."""
    missing = []
    for label, path in ciapath_list.items():
        if not os.path.exists(path):
            missing.append(f"CIA {label}: {path}")

    if args.opacity_mode != "ckd" and opa_load:
        for mol in selected_molecules:
            path = resolve_premodit_snapshot_path(mol)
            if path is None:
                candidates = ", ".join(premodit_snapshot_candidates(mol))
                missing.append(f"preMODIT snapshot {mol}: {candidates}")

    if missing:
        message = (
            "Required local inputs are missing:\n  "
            + "\n  ".join(missing)
            + "\nRun with --check-inputs to inspect expected paths."
        )
        raise FileNotFoundError(message)


def build_premodit_from_snapshot(snapshot, molmass, mol):
    """Create preMODIT opacity and persist it for reuse."""
    opa = OpaPremodit.from_snapshot(
        snapshot,
        nu_grid_nirspec,
        nstitch=ndiv,
        diffmode=diffmode,
        auto_trange=[Tlow, Thigh],
        dit_grid_resolution=1,
        allow_32bit=True,
        cutwing=1 / (2 * ndiv),
    )
    saveopa(opa, "opa_" + mol + ".zarr", format="zarr", aux={"molmass": molmass})
    return opa

def load_ckd_from_exomolop(mol, path, nurange):
    """Load an ExoMolOP CKD table for one molecule."""
    opa = OpaCKD.from_external("exomolop", path, nurange=nurange)
    return opa, opa.molmass

def load_or_build_opacity(mol, path, mdb_factory):
    """Load saved opacity or build from database snapshot."""
    if opa_load:
        snapshot_path = resolve_premodit_snapshot_path(mol)
        if snapshot_path is None:
            candidates = ", ".join(premodit_snapshot_candidates(mol))
            raise FileNotFoundError(f"preMODIT snapshot {mol}: {candidates}")
        opa = OpaPremodit.from_saved_opa(snapshot_path, strict=False)
        return opa, opa.aux["molmass"]

    mdb = mdb_factory(path)
    molmass = mdb.molmass
    opa = build_premodit_from_snapshot(mdb.to_snapshot(), molmass, mol)
    del mdb
    return opa, molmass


def load_molecular_opacities():
    """Load or create all molecular opacities for HITEMP and ExoMol."""
    opa_mols_local = {}
    molmass_list = []

    if args.opacity_mode == "ckd":
        print("Loading ExoMolOP CKD tables …")
        print(
            "  CKD nurange with RV prior: "
            f"{ckd_nurange[0]:.3f}-{ckd_nurange[1]:.3f} cm-1"
        )
        for mol, path in ckdpath_list_ExomolOP.items():
            print(f"  * {mol} (ExoMolOP CKD): {path}")
            opa, molmass = load_ckd_from_exomolop(mol, path, nurange=ckd_nurange)
            opa_mols_local[mol] = opa
            molmass_list.append(molmass)
        return opa_mols_local, jnp.array(molmass_list)

    print("Loading HITEMP/ExoMol databases …")
    for mol, path in molpath_list_HITEMP.items():
        print(f"  * {mol} (HITEMP)")
        mdb_factory = lambda p: MdbHitemp(p, nu_grid_nirspec, gpu_transfer=False, isotope=1)
        opa, molmass = load_or_build_opacity(mol, path, mdb_factory)
        opa_mols_local[mol] = opa
        molmass_list.append(molmass)

    for mol, path in molpath_list_Exomol.items():
        print(f"  * {mol} (ExoMol)")
        mdb_factory = lambda p: MdbExomol(p, nu_grid_nirspec, gpu_transfer=False)
        opa, molmass = load_or_build_opacity(mol, path, mdb_factory)
        opa_mols_local[mol] = opa
        molmass_list.append(molmass)

    return opa_mols_local, jnp.array(molmass_list)


def validate_ckd_table_compatibility(opa_mols_local):
    """Validate that loaded molecule CKD tables share the same CKD grid."""
    reference_mol, reference_opa = next(iter(opa_mols_local.items()))
    reference_info = reference_opa.ckd_info
    reference_arrays = {
        "band centers": np.asarray(reference_opa.nu_bands),
        "band edges": np.asarray(reference_opa.band_edges),
        "g-grid": np.asarray(reference_info.ggrid),
        "quadrature weights": np.asarray(reference_info.weights),
    }
    for mol, opa in opa_mols_local.items():
        info = opa.ckd_info
        arrays = {
            "band centers": np.asarray(opa.nu_bands),
            "band edges": np.asarray(opa.band_edges),
            "g-grid": np.asarray(info.ggrid),
            "quadrature weights": np.asarray(info.weights),
        }
        for label, reference_array in reference_arrays.items():
            array = arrays[label]
            if array.shape != reference_array.shape or not np.allclose(
                array, reference_array
            ):
                raise ValueError(
                    "CKD table "
                    f"{label} mismatch for {mol}; expected compatibility with "
                    f"{reference_mol}. "
                    f"{reference_mol} source={ckdpath_list_ExomolOP[reference_mol]}, "
                    f"shape={reference_array.shape}; "
                    f"{mol} source={ckdpath_list_ExomolOP[mol]}, shape={array.shape}."
                )
    return reference_opa


validate_required_inputs()
opa_mols, molmass_arr = load_molecular_opacities()
if args.opacity_mode == "ckd":
    ckd_reference = validate_ckd_table_compatibility(opa_mols)
    ckd_nu_bands = ckd_reference.nu_bands
    ckd_weights = ckd_reference.ckd_info.weights
    validate_ckd_band_coverage(ckd_nu_bands, ckd_nurange, ckd_reference.band_edges)
    validate_ckd_sampling_inputs(
        ckd_nu_bands, np.ones_like(np.asarray(ckd_nu_bands)), wav_obs_fit
    )
    opa_cias = build_cia_opacities(ckd_nu_bands)
else:
    ckd_nu_bands = None
    ckd_weights = None
    opa_cias = build_cia_opacities(nu_grid_nirspec)

# %% 
# Spectral model function
# -----------------------

def spectral_model(radius_btm, Mp, Rstar, RV, vmr_arr, T0, logP_cloud):
    
    vmr_tot = jnp.clip(jnp.sum(vmr_arr, axis=0), 0.0, 1.0)
    vmrH2 = (1.0 - vmr_tot) * 6.0 / 7.0
    vmrHe = (1.0 - vmr_tot) * 1.0 / 7.0

    mmw = (
        molinfo.molmass_isotope("H2") * vmrH2
        + molinfo.molmass_isotope("He", db_HIT=False) * vmrHe
        + jnp.dot(molmass_arr, vmr_arr)
    )

    # --- Temperature structure -------------------------------------------
    Tarr = T0 * jnp.ones_like(art.pressure)  # constant T profile


    # Fixed cloud width in log10(P) space (narrow deck).
    width_cloud = 1.0 / 25.0

    # Set the Gaussian amplitude so that the *integrated* cloud optical depth
    # over the atmosphere is ~50, independent of the number of layers.
    dtau_c = (
        50.0
        * ((jnp.log10(pressure_btm) - jnp.log10(pressure_top)) / nlayer)
        / width_cloud
    )
    pressure_arr = jnp.log10(art.pressure)
    cloud_profile = (pressure_arr[:, None] - logP_cloud) / width_cloud
    # Per-layer optical-depth increment: normalized Gaussian in log10(P).
    dtau_cloud = (
        dtau_c / jnp.sqrt(jnp.pi) * jnp.exp(-jnp.clip(cloud_profile**2, -50, 50))
    )
    # --- Gravity profile --------------------------------------------------
    gravity_btm = gravity_jupiter(radius_btm / RJ, Mp / MJ)
    gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)

    if args.opacity_mode == "ckd":
        nband = ckd_nu_bands.size
        ng = ckd_weights.size
        dtau_ckd = jnp.broadcast_to(dtau_cloud[:, None], (pressure_arr.size, ng, nband))

        for molA, molB in [("H2", "H2"), ("H2", "He")]:
            cia_pair = molA + molB
            if cia_pair not in opa_cias:
                continue
            logacia_matrix = opa_cias[cia_pair].logacia_matrix(Tarr)
            vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
            dtau_cia = art.opacity_profile_cia(
                logacia_matrix, Tarr, vmrX, vmrY, mmw, gravity
            )
            dtau_ckd += dtau_cia[:, None, :]

        for i, mol in enumerate(opa_mols):
            xstensor_ckd = opa_mols[mol].xstensor_ckd(Tarr, art.pressure)
            dtau_ckd += art.opacity_profile_xs_ckd(
                xstensor_ckd, vmr_arr[i], mmw, gravity
            )

        rp2_bands = art.run_ckd(
            dtau_ckd, Tarr, mmw, radius_btm, gravity_btm, ckd_weights
        )
        rp2_sample = sample_ckd_bands_at_wavelengths(
            ckd_nu_bands, rp2_bands, wav_obs_fit, radial_velocity=RV, unit="nm"
        )
        return jnp.sqrt(rp2_sample) * (radius_btm / Rstar)

    # Broadcast to all wavelengths to make the cloud gray.
    dtau_nirspec = jnp.broadcast_to(dtau_cloud, (pressure_arr.size, nu_grid_nirspec.size))

    # --- Opacity summation -------------------------------------------------
    # CIA
    for molA, molB in [("H2", "H2"), ("H2", "He")]:
        cia_pair = molA + molB
        if cia_pair not in opa_cias:
            continue
        logacia_matrix = opa_cias[cia_pair].logacia_matrix(Tarr)
        vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
        dtau_nirspec += art.opacity_profile_cia(
            logacia_matrix, Tarr, vmrX, vmrY, mmw, gravity
        )

    # Line opacity
    for i, mol in enumerate(opa_mols):
        xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
        dtau_nirspec += art.opacity_profile_xs(xsmatrix, vmr_arr[i], mmw, gravity)

    # --- Radiative‑transfer ------------------------------------------------
    rp2 = art.run(
        dtau_nirspec, Tarr, mmw, radius_btm, gravity_btm
    )  # (radius/radius_btm)^2 spectrum

    # --- Broadening kernels ------------------------------------------------
    Frot_inst = sop_inst_nirspec.ipgauss(rp2, beta_inst_nirspec)
    Rp2_sample = sop_inst_nirspec.sampling(Frot_inst, RV, inst_nirspec_nus)

    mu = jnp.sqrt(Rp2_sample) * (radius_btm / Rstar)  # (radius/Rstar) spectrum
    return mu


# %%
# Probabilistic model
# -------------------
#
# The NumPyro model couples planetary/stellar parameters, molecular mixing
# ratios, a grey cloud deck, and a simple isothermal temperature structure. It
# produces a model transmission spectrum convolved with rotation and the
# instrumental profile, then compares it to the observed ``R_p/R_s`` data.


def model_c(rp_mean, rp_std):
    """NumPyro model: forward spectral model + priors."""

    # --- Atmospheric composition -----------------------------------------
    vmr_arr = []
    for mol in opa_mols:
        logVMR = numpyro.sample(f"logVMR_{mol}", dist.Uniform(-15, 0))
        vmr_arr.append(art.constant_mmr_profile(jnp.power(10.0, logVMR)))
    vmr_arr = jnp.array(vmr_arr)

    # --- Temperature structure -------------------------------------------
    T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))

    # --- Grey cloud deck ---------------------------------------------------
    # We model a wavelength-independent (gray) cloud deck as a Gaussian in
    # log10-pressure. The cloud center logP_cloud is a free parameter.
    logP_cloud = numpyro.sample("logP_cloud", dist.Uniform(-11, 1))
    

    # --- Planet / star parameters -----------------------------------------
    Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0)) * MJ
    Rstar = (
        numpyro.sample("Rs", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0)) * Rs
    )
    radius_btm = numpyro.sample("Radius_btm", dist.Uniform(1.0, 1.5)) * RJ
    RV = numpyro.sample("RV", dist.Uniform(args.rv_min, args.rv_max))

    mu = spectral_model(radius_btm, Mp, Rstar, RV, vmr_arr, T0, logP_cloud)
    mu_obs_order = mu if args.opacity_mode == "ckd" else mu[::-1]
    # --- Likelihood -------------------------------------------------------
    numpyro.deterministic("rp_mu", mu_obs_order)
    numpyro.sample("rp", dist.Normal(mu_obs_order, rp_std), obs=rp_mean)


def run_forward_check():
    """Evaluate the forward model once and print data/model shape diagnostics."""
    fiducial_vmr = []
    for _mol in opa_mols:
        fiducial_vmr.append(art.constant_mmr_profile(1.0e-5))
    fiducial_vmr = jnp.array(fiducial_vmr)
    fiducial_rv = 0.5 * (args.rv_min + args.rv_max)

    fiducial_mu = spectral_model(
        radius_btm=1.27 * RJ,
        Mp=Mp_mean * MJ,
        Rstar=Rstar_mean * Rs,
        RV=fiducial_rv,
        vmr_arr=fiducial_vmr,
        T0=1200.0,
        logP_cloud=-3.0,
    )
    fiducial_mu = fiducial_mu if args.opacity_mode == "ckd" else fiducial_mu[::-1]
    fiducial_mu_np = np.asarray(jax.device_get(fiducial_mu))
    finite_model = bool(np.all(np.isfinite(fiducial_mu_np)))
    shape_matches = fiducial_mu_np.shape == rp_mean_fit.shape
    forward_payload = {
        "data_mode": args.data_mode,
        "opacity_mode": args.opacity_mode,
        "molecules": list(opa_mols.keys()),
        "channels": list(selected_channels),
        "cia_pairs": list(selected_cia_pairs),
        "input_status": input_status_payload(
            args.data_dir,
            args.opacity_root,
            args.ckd_root,
            selected_molecules,
            selected_channels,
            selected_cia_pairs,
            selected_ckd_table_paths,
        ),
        "observed_shape": list(rp_mean_fit.shape),
        "model_shape": list(fiducial_mu_np.shape),
        "shape_matches": shape_matches,
        "finite_model": finite_model,
        "model_rprs_min": float(np.min(fiducial_mu_np)),
        "model_rprs_max": float(np.max(fiducial_mu_np)),
        "fiducial_rv_kms": float(fiducial_rv),
        "n_observed": int(rp_mean_fit.size),
        "wavelength_nm_min": float(np.min(wav_obs_fit)),
        "wavelength_nm_max": float(np.max(wav_obs_fit)),
        "retrieval_channel_summary": retrieval_channel_summary(
            selected_channels, channel_index_fit, wav_obs_fit
        ),
    }
    if args.opacity_mode == "ckd":
        forward_payload["ckd_nurange_cm-1"] = [
            float(ckd_nurange[0]),
            float(ckd_nurange[1]),
        ]
        forward_payload["ckd_band_summary"] = ckd_band_summary()
        forward_payload["ckd_sources"] = ckdpath_list_ExomolOP
        forward_payload["ckd_table_summary"] = ckd_table_summary()

    print("Forward-model check:")
    print(f"  data_mode: {args.data_mode}")
    print(f"  opacity_mode: {args.opacity_mode}")
    print("  molecules: " + format_selection(opa_mols.keys()))
    print("  channels: " + format_selection(selected_channels))
    print("  CIA pairs: " + format_selection(selected_cia_pairs))
    print(f"  fiducial RV: {fiducial_rv:.3f} km/s")
    if args.opacity_mode == "ckd":
        band_summary = ckd_band_summary()
        print(
            "  CKD bands: "
            f"n={band_summary['n_bands']}, "
            f"g={band_summary['n_g']}, "
            f"centers=[{band_summary['nu_band_min_cm-1']:.3f}, "
            f"{band_summary['nu_band_max_cm-1']:.3f}] cm-1"
        )
        if "nu_edge_min_cm-1" in band_summary:
            print(
                "  CKD band edges: "
                f"[{band_summary['nu_edge_min_cm-1']:.3f}, "
                f"{band_summary['nu_edge_max_cm-1']:.3f}] cm-1"
            )
    print(f"  observed shape: {rp_mean_fit.shape}")
    print(f"  model shape: {fiducial_mu_np.shape}")
    print(f"  finite model: {finite_model}")
    print(
        "  model Rp/Rs range: "
        f"[{np.min(fiducial_mu_np):.6f}, {np.max(fiducial_mu_np):.6f}]"
    )
    if args.forward_check_json:
        save_json(args.forward_check_json, forward_payload, "Forward-check JSON")
    if not shape_matches:
        raise ValueError(
            "Forward-model shape does not match the selected observed data vector."
        )
    if not finite_model:
        raise ValueError("Forward model returned non-finite values.")


if args.check_forward:
    run_forward_check()
    sys.exit(0)


# %%
# Stochastic Variational Inference (SVI) warm-up for HMC-NUTS (Optional)
# ----------------------------------------------------------------------
#
# Run stochastic variational inference with a custom guide that keeps Mp and
# Rs on their priors while fitting an AutoMultivariateNormal to the remaining
# latent variables. The SVI median seeds HMC and its Fisher information is
# reused as a mass matrix estimate.


def build_guide():
    """Construct an AutoMVN guide over the latent sample sites."""
    model_hidden = handlers.block(model_c, hide=["rp_mu"])
    return AutoMultivariateNormal(model_hidden)


def save_run_config(output_dir):
    """Persist the retrieval configuration for reproducibility."""
    config = vars(args).copy()
    config["molecules"] = list(selected_molecules)
    config["channels"] = list(selected_channels)
    config["channel_summary"] = observed_channel_summary(observed_spectra)
    config["retrieval_channel_summary"] = retrieval_channel_summary(
        selected_channels, channel_index_fit, wav_obs_fit
    )
    config["cia_pairs"] = list(selected_cia_pairs)
    config["data_dir"] = args.data_dir
    config["opacity_root"] = args.opacity_root
    config["ckd_root"] = args.ckd_root
    config["ckd_table_paths"] = selected_ckd_table_paths
    config["ckd_sources"] = ckdpath_list_ExomolOP if args.opacity_mode == "ckd" else {}
    config["ckd_table_summary"] = ckd_table_summary()
    config["rv_min_kms"] = args.rv_min
    config["rv_max_kms"] = args.rv_max
    config["skip_diagnostic_plots"] = bool(args.skip_diagnostic_plots)
    if ckd_nurange is not None:
        config["ckd_nurange_cm-1"] = [float(ckd_nurange[0]), float(ckd_nurange[1])]
    config["ckd_band_summary"] = ckd_band_summary()
    config["n_observed"] = int(rp_mean_fit.size)
    config["wavelength_nm_min"] = float(np.min(wav_obs_fit))
    config["wavelength_nm_max"] = float(np.max(wav_obs_fit))
    config["opacity_molecules_loaded"] = list(opa_mols.keys())
    config["exojax_version"] = getattr(exojax, "__version__", "unknown")
    config["python_version"] = platform.python_version()
    config["jax_default_backend"] = jax.default_backend()
    config["jax_devices"] = [str(device) for device in jax.devices()]

    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Run configuration saved to {path}")


def save_svi_outputs(params, losses, init_values, output_dir):
    """Persist SVI artifacts for reuse or inspection."""
    params_cpu = {k: np.asarray(jax.device_get(v)) for k, v in params.items()}
    losses_cpu = np.asarray(jax.device_get(losses))
    init_cpu = {k: np.asarray(jax.device_get(v)) for k, v in init_values.items()}

    np.savez(os.path.join(output_dir, "svi_params.npz"), **params_cpu)
    np.save(os.path.join(output_dir, "svi_losses.npy"), losses_cpu)
    np.savez(os.path.join(output_dir, "svi_init_values.npz"), **init_cpu)

    print(f"SVI params saved to {output_dir}/svi_params.npz")
    print(f"SVI losses saved to {output_dir}/svi_losses.npy")
    print(f"SVI init values saved to {output_dir}/svi_init_values.npz")


def save_observed_data(output_dir):
    """Persist the selected observed data vector used by this retrieval."""
    path = os.path.join(output_dir, "observed_data.npz")
    np.savez(
        path,
        wavelength_nm=np.asarray(wav_obs_fit),
        radius_ratio=np.asarray(rp_mean_fit),
        radius_ratio_error=np.asarray(rp_std_fit),
        channel_index=np.asarray(channel_index_fit),
        channel_names=np.asarray(selected_channels),
    )
    print(f"Observed data vector saved to {path}")


def save_posterior_samples(sample_dict, output_dir):
    """Persist posterior samples as host NumPy arrays."""
    path = os.path.join(output_dir, "posterior_sample.npz")
    sample_cpu = {k: np.asarray(jax.device_get(v)) for k, v in sample_dict.items()}
    np.savez(path, **sample_cpu)
    print(f"Posterior samples saved to {path}")


def save_mcmc_summary(mcmc, output_dir):
    """Print and persist the MCMC summary when enough samples are available."""
    path = os.path.join(output_dir, "mcmc_summary.txt")
    if args.num_samples < 4:
        message = (
            "MCMC summary skipped because --num-samples < 4. "
            "Posterior samples were still saved for smoke-test inspection."
        )
        print(message)
        with open(path, "w") as f:
            f.write(message + "\n")
        return

    mcmc.print_summary()
    with open(path, "w") as f:
        with redirect_stdout(f):
            mcmc.print_summary()


def save_predictive_spectrum(prediction_dict, output_dir):
    """Persist posterior predictive spectra as host NumPy arrays."""
    predictive_payload = {
        "wavelength_nm": np.asarray(wav_obs_fit),
        "channel_index": np.asarray(channel_index_fit),
        "channel_names": np.asarray(selected_channels),
    }
    if "rp_mu" in prediction_dict:
        rp_mu = np.asarray(jax.device_get(prediction_dict["rp_mu"]))
        path_mu = os.path.join(output_dir, "rp_mu_pred.npy")
        np.save(path_mu, rp_mu)
        predictive_payload["rp_mu"] = rp_mu
        print(f"Posterior model spectra saved to {path_mu}")

    if "rp" in prediction_dict:
        rp = np.asarray(jax.device_get(prediction_dict["rp"]))
        path_rp = os.path.join(output_dir, "rp_pred.npy")
        np.save(path_rp, rp)
        predictive_payload["rp"] = rp
        print(f"Posterior predictive noisy spectra saved to {path_rp}")

    path_npz = os.path.join(output_dir, "posterior_predictive.npz")
    np.savez(path_npz, **predictive_payload)
    print(f"Posterior predictive bundle saved to {path_npz}")


def save_run_status(output_dir, posterior_samples, prediction_dict, losses):
    """Persist a compact status artifact for completed HMC smoke/retrieval runs."""
    artifact_names = [
        "run_config.json",
        "observed_data.npz",
        "svi_params.npz",
        "svi_losses.npy",
        "svi_init_values.npz",
        "posterior_sample.npz",
        "posterior_predictive.npz",
        "rp_mu_pred.npy",
        "rp_pred.npy",
        "mcmc_summary.txt",
    ]
    skipped_artifacts = []
    if args.skip_diagnostic_plots:
        skipped_artifacts.extend(["svi_loss.png", "spectrum_overlay.png"])
    else:
        artifact_names.extend(["svi_loss.png", "spectrum_overlay.png"])
    artifacts = {
        name: os.path.exists(os.path.join(output_dir, name)) for name in artifact_names
    }
    posterior_shapes = {
        key: list(np.asarray(jax.device_get(value)).shape)
        for key, value in posterior_samples.items()
    }
    predictive_shapes = {
        key: list(np.asarray(jax.device_get(value)).shape)
        for key, value in prediction_dict.items()
    }
    posterior_finite = {
        key: bool(np.all(np.isfinite(np.asarray(jax.device_get(value)))))
        for key, value in posterior_samples.items()
    }
    predictive_finite = {
        key: bool(np.all(np.isfinite(np.asarray(jax.device_get(value)))))
        for key, value in prediction_dict.items()
    }
    loss_values = np.asarray(jax.device_get(losses))
    finite_checks = {
        "posterior_all_finite": all(posterior_finite.values()),
        "predictive_all_finite": all(predictive_finite.values()),
        "svi_losses_all_finite": bool(np.all(np.isfinite(loss_values))),
    }
    n_observed = int(rp_mean_fit.size)
    posterior_rp_mu_shape = posterior_shapes.get("rp_mu")
    predictive_rp_mu_shape = predictive_shapes.get("rp_mu")
    predictive_rp_shape = predictive_shapes.get("rp")
    shape_checks = {
        "posterior_rp_mu_matches_observed": (
            bool(posterior_rp_mu_shape) and posterior_rp_mu_shape[-1] == n_observed
        ),
        "predictive_rp_mu_matches_observed": (
            bool(predictive_rp_mu_shape) and predictive_rp_mu_shape[-1] == n_observed
        ),
        "predictive_rp_matches_observed": (
            bool(predictive_rp_shape) and predictive_rp_shape[-1] == n_observed
        ),
    }
    status = {
        "status": "completed",
        "data_mode": args.data_mode,
        "opacity_mode": args.opacity_mode,
        "molecules": list(selected_molecules),
        "channels": list(selected_channels),
        "cia_pairs": list(selected_cia_pairs),
        "input_status": input_status_payload(
            args.data_dir,
            args.opacity_root,
            args.ckd_root,
            selected_molecules,
            selected_channels,
            selected_cia_pairs,
            selected_ckd_table_paths,
        ),
        "retrieval_channel_summary": retrieval_channel_summary(
            selected_channels, channel_index_fit, wav_obs_fit
        ),
        "ckd_sources": ckdpath_list_ExomolOP if args.opacity_mode == "ckd" else {},
        "ckd_band_summary": ckd_band_summary(),
        "ckd_table_summary": ckd_table_summary(),
        "n_observed": n_observed,
        "wavelength_nm_min": float(np.min(wav_obs_fit)),
        "wavelength_nm_max": float(np.max(wav_obs_fit)),
        "rv_min_kms": float(args.rv_min),
        "rv_max_kms": float(args.rv_max),
        "num_warmup": int(args.num_warmup),
        "num_samples": int(args.num_samples),
        "num_chains": int(args.num_chains),
        "svi_steps": int(args.svi_steps),
        "svi_lr": float(args.svi_lr),
        "chain_method": args.chain_method,
        "max_tree_depth": int(args.max_tree_depth),
        "rng_seed": int(args.rng_seed),
        "skip_diagnostic_plots": bool(args.skip_diagnostic_plots),
        "final_svi_loss": float(np.asarray(jax.device_get(losses))[-1]),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "posterior_shapes": posterior_shapes,
        "predictive_shapes": predictive_shapes,
        "posterior_finite": posterior_finite,
        "predictive_finite": predictive_finite,
        "finite_checks": finite_checks,
        "shape_checks": shape_checks,
        "expected_artifacts": artifact_names,
        "skipped_artifacts": skipped_artifacts,
        "artifacts": artifacts,
        "all_expected_artifacts_present": all(artifacts.values()),
        "ready_for_inspection": (
            all(artifacts.values())
            and all(shape_checks.values())
            and all(finite_checks.values())
        ),
    }
    if ckd_nurange is not None:
        status["ckd_nurange_cm-1"] = [float(ckd_nurange[0]), float(ckd_nurange[1])]
    save_json(os.path.join(output_dir, "run_status.json"), status, "Run status JSON")


def run_svi(rng_key, rp_mean, rp_std, num_steps=1000, lr=0.005):
    """Execute SVI, return params, losses, init strategy, median, and guide."""
    guide = build_guide()
    optimizer = optim.Adam(lr)
    svi = SVI(model_c, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(
        rng_key,
        num_steps,
        rp_mean=rp_mean,
        rp_std=rp_std,
        progress_bar=not args.no_progress_bar,
    )

    params = svi_result.params
    losses = svi_result.losses

    # Median in the constrained space.
    svi_median = guide.median(params)
    # Keep Mp and Rs anchored to their prior means for HMC initialisation.
    svi_median.update({"Mp": Mp_mean, "Rs": Rstar_mean})
    init_strategy = init_to_value(values=svi_median)

    save_svi_outputs(params, losses, svi_median, DIR_SAVE)
    return params, losses, init_strategy, svi_median, guide


os.makedirs(DIR_SAVE, exist_ok=True)
save_run_config(DIR_SAVE)
save_observed_data(DIR_SAVE)
rng_key = random.PRNGKey(args.rng_seed)

print("Stochastic Variational Inference (SVI) to find initial values for HMC-NUTS …")
print(
    "Run settings: "
    f"svi_steps={args.svi_steps}, svi_lr={args.svi_lr}, "
    f"num_warmup={args.num_warmup}, num_samples={args.num_samples}, "
    f"num_chains={args.num_chains}, chain_method={args.chain_method}, "
    f"max_tree_depth={args.max_tree_depth}, rng_seed={args.rng_seed}, "
    f"rv_prior=[{args.rv_min}, {args.rv_max}] km/s, "
    f"svi_plot_samples={args.svi_plot_samples}"
)

rng_key, rng_key_ = random.split(rng_key)

_svi_params, losses, init_strategy, svi_median, svi_guide = run_svi(
    rng_key_,
    rp_mean=rp_mean_fit,
    rp_std=rp_std_fit,
    num_steps=args.svi_steps,
    lr=args.svi_lr,
)
print(f"Final SVI loss: {float(losses[-1]):.2f}")
print("HMC initial values:", init_strategy)

# %%
# HMC-NUTS sampling
# -----------------

print("Launching HMC-NUTS …")

kernel = NUTS(
    model_c,
    max_tree_depth=args.max_tree_depth,
    init_strategy=init_strategy,
)
rng_key, rng_key_ = random.split(rng_key)

mcmc = MCMC(
    kernel,
    num_warmup=args.num_warmup,
    num_samples=args.num_samples,
    num_chains=args.num_chains,
    chain_method=args.chain_method,
    progress_bar=not args.no_progress_bar,
)
mcmc.run(rng_key_, rp_mean=rp_mean_fit, rp_std=rp_std_fit)

# Print summary to console *and* save to file
save_mcmc_summary(mcmc, DIR_SAVE)

# Save posterior samples and predictive spectra
posterior_sample = mcmc.get_samples()
save_posterior_samples(posterior_sample, DIR_SAVE)

print("Generating predictive spectrum …")

pred = Predictive(model_c, posterior_sample, return_sites=["rp", "rp_mu"])
predictions = pred(rng_key_, rp_mean=None, rp_std=rp_std_fit)

save_predictive_spectrum(predictions, DIR_SAVE)


# %%
# Plotting
# --------
#
# Generate quick-look diagnostics: SVI loss curve, HMC predictive spectrum,
# observed/SVI/HMC overlay, and corner plots for a subset of parameters.
if args.skip_diagnostic_plots:
    print("Skipping post-HMC diagnostic plots.")
else:
    print("Plotting SVI and HMC diagnostics …")


def plot_svi_loss(loss_values, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(loss_values))
    ax.plot(x, np.asarray(loss_values), lw=1.5)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("Loss")
    ax.set_title("SVI loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_overlay(wavelength_nm, rp_obs, rp_err, rp_hmc, rp_svi, save_path):
    rp_hmc_np = np.asarray(rp_hmc)
    mean = rp_hmc_np.mean(axis=0)
    std = rp_hmc_np.std(axis=0)
    rp_svi_np = np.asarray(rp_svi)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        wavelength_nm,
        rp_obs,
        yerr=rp_err,
        fmt=".",
        ms=1,
        color="k",
        ecolor="0.3",
        elinewidth=0.5,
        alpha=0.5,
        label="Observed",
    )
    ax.fill_between(
        wavelength_nm,
        mean - std,
        mean + std,
        color="C0",
        alpha=0.25,
        label=r"HMC ±1$\sigma$",
    )
    ax.plot(wavelength_nm, mean, color="C0", lw=1.4, label="HMC mean")
    ax.plot(wavelength_nm, rp_svi_np, color="C3", lw=1.4, label="SVI median model")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"$R_p/R_s$")
    ax.set_title("Observed vs SVI vs HMC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _corner_data(sample_dict, variables):
    cols = []
    labels = []
    available = [v for v in variables if v in sample_dict]
    for var in available:
        arr = np.asarray(sample_dict[var])
        arr = arr.reshape(arr.shape[0], -1)
        for j in range(arr.shape[1]):
            cols.append(arr[:, j])
            labels.append(var if arr.shape[1] == 1 else f"{var}[{j}]")
    if not cols:
        return None, None
    return np.column_stack(cols), labels


def _corner_dimension_count(sample_dict, variables):
    """Return number of scalar dimensions that would be plotted by corner."""
    total = 0
    for var in variables:
        if var not in sample_dict:
            continue
        arr = np.asarray(sample_dict[var])
        total += int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
    return total


def _posterior_sample_count(sample_dict):
    """Return the number of flattened HMC posterior samples."""
    if not sample_dict:
        return 0
    first = next(iter(sample_dict.values()))
    return int(np.asarray(first).shape[0])


def plot_corner(hmc_samples=None, svi_samples=None, variables=None, save_path=None):
    """Corner plot helper: supports HMC only, SVI only, or HMC+SVI overlay."""
    if corner is None:
        print("corner is not installed; skipping corner plot.")
        return

    datasets = []
    labels = None

    if hmc_samples is not None:
        hmc_data, labels = _corner_data(hmc_samples, variables)
        if hmc_data is not None:
            datasets.append((hmc_data, "C0", {}))

    if svi_samples is not None:
        svi_data, labels_svi = _corner_data(svi_samples, variables)
        if labels is None:
            labels = labels_svi
        if svi_data is not None:
            datasets.append((svi_data, "C3", {"hist_kwargs": {"linestyle": "--"}}))

    if not datasets or labels is None:
        print("No data for corner plot; skipping.")
        return

    fig = None
    for data, color, extra_kwargs in datasets:
        fig = corner.corner(
            data,
            labels=labels,
            color=color,
            bins=40,
            smooth=1.0,
            fig=fig,
            show_titles=True,
            **extra_kwargs,
        )

    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# Generate deterministic rp_mu from SVI median parameters
if not args.skip_diagnostic_plots:
    rng_key, rng_plot = random.split(rng_key)
    svi_pred = Predictive(
        model_c, params=svi_median, num_samples=1, return_sites=["rp_mu"]
    )
    svi_mu = svi_pred(rng_plot, rp_mean=rp_mean_fit, rp_std=rp_std_fit)["rp_mu"][0]

    loss_plot_path = os.path.join(DIR_SAVE, "svi_loss.png")
    plot_svi_loss(losses, loss_plot_path)

    overlay_plot_path = os.path.join(DIR_SAVE, "spectrum_overlay.png")
    plot_overlay(
        wav_obs_fit,
        rp_mean_fit,
        rp_std_fit,
        predictions["rp_mu"],
        svi_mu,
        overlay_plot_path,
    )

    corner_vars = ["Radius_btm", "T0", "logP_cloud", "RV"]
    corner_vars += [f"logVMR_{mol}" for mol in list(opa_mols.keys())]
    if not args.skip_corner:
        corner_dim = _corner_dimension_count(posterior_sample, corner_vars)
        hmc_corner_samples = _posterior_sample_count(posterior_sample)
        svi_corner_samples = int(args.svi_plot_samples)
        if corner is None:
            print("corner is not installed; skipping corner plots.")
        elif hmc_corner_samples <= corner_dim or svi_corner_samples <= corner_dim:
            print(
                "Skipping corner plots because HMC and SVI guide sample counts must "
                f"exceed the corner dimension ({corner_dim}); "
                f"hmc_samples={hmc_corner_samples}, svi_samples={svi_corner_samples}."
            )
        else:
            # Draw samples from the SVI guide for visualization only when needed.
            rng_key, rng_svi = random.split(rng_key)
            svi_samples = svi_guide.sample_posterior(
                rng_svi,
                _svi_params,
                rp_mean=rp_mean_fit,
                rp_std=rp_std_fit,
                sample_shape=(args.svi_plot_samples,),
            )

            corner_plot_path = os.path.join(DIR_SAVE, "corner_plot_svi.png")
            plot_corner(
                svi_samples=svi_samples,
                variables=corner_vars,
                save_path=corner_plot_path,
            )

            hmc_corner_plot_path = os.path.join(DIR_SAVE, "corner_plot.png")
            plot_corner(
                hmc_samples=posterior_sample,
                variables=corner_vars,
                save_path=hmc_corner_plot_path,
            )

            hmc_svi_corner_overlay_path = os.path.join(
                DIR_SAVE, "corner_plot_hmc_svi_overlay.png"
            )
            plot_corner(
                hmc_samples=posterior_sample,
                svi_samples=svi_samples,
                variables=corner_vars,
                save_path=hmc_svi_corner_overlay_path,
            )

save_run_status(DIR_SAVE, posterior_sample, predictions, losses)
