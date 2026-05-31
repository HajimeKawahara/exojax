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
import math
import os
import platform
import sys

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
    help="Comma-separated CIA opacity pairs.",
)
parser.add_argument(
    "--check-inputs",
    action="store_true",
    help="Print the expected input file and directory status, then stop.",
)
parser.add_argument(
    "--check-forward",
    action="store_true",
    help="Run one fiducial forward-model evaluation, print shape checks, then stop.",
)
parser.add_argument(
    "--quick",
    action="store_true",
    help="Use short SVI/HMC settings for a smoke test.",
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
    cia_pairs = tuple(pair.strip() for pair in cia_pair_text.split(",") if pair.strip())
    if not cia_pairs:
        parser.error("--cia-pairs must include at least one CIA pair.")
    unsupported = [pair for pair in cia_pairs if pair not in SUPPORTED_CIA_PAIRS]
    if unsupported:
        parser.error(
            "Unsupported CIA pairs in --cia-pairs: "
            + ", ".join(unsupported)
            + ". Supported CIA pairs are: "
            + ", ".join(DEFAULT_CIA_PAIRS)
        )
    return cia_pairs


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
    if args.rng_seed < 0:
        parser.error("--rng-seed must be zero or a positive integer.")
    if not math.isfinite(args.rv_min) or not math.isfinite(args.rv_max):
        parser.error("--rv-min and --rv-max must be finite.")
    if args.rv_min >= args.rv_max:
        parser.error("--rv-min must be smaller than --rv-max.")


selected_molecules = parse_molecule_list(args.molecules)
selected_channels = parse_channel_list(args.channels)
selected_cia_pairs = parse_cia_pair_list(args.cia_pairs)
validate_numeric_args(args)

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
    raise ValueError(
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
import h5py

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

ciapath_list = {
    pair: os.path.join(args.opacity_root, CIA_RELATIVE_FILES[pair])
    for pair in DEFAULT_CIA_PAIRS
}
ciapath_list = {pair: ciapath_list[pair] for pair in selected_cia_pairs}


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


def validate_local_inputs_before_heavy_imports():
    """Fail before importing JAX/ExoJAX when local required files are missing."""
    if args.plot_data_only or args.summarize_data or args.check_inputs:
        return

    missing = []
    for label, path in ciapath_list.items():
        if not os.path.exists(path):
            missing.append(f"CIA {label}: {path}")

    if args.opacity_mode == "ckd" and not args.allow_ckd_download:
        for mol in selected_molecules:
            path = os.path.join(args.ckd_root, CKD_RELATIVE_PATHS[mol])
            if not os.path.isdir(path):
                missing.append(f"ExoMolOP CKD {mol}: {path}")
                continue
            if not glob.glob(os.path.join(path, "*.h5")):
                missing.append(f"ExoMolOP CKD {mol} h5 table: {path}/*.h5")

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

needs_plotting = args.plot_data_only or not (
    args.summarize_data or args.check_inputs or args.check_forward
)

if needs_plotting:
    import matplotlib.pyplot as plt

if not (args.plot_data_only or args.summarize_data or args.check_inputs):
    import jax
    from jax import random
    import jax.numpy as jnp
    from contextlib import redirect_stdout

    import exojax
    if args.check_forward:
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
        from numpyro.infer.autoguide import AutoMultivariateNormal, AutoGuideList
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
    data_dir, opacity_root, ckd_root, molecules, channels, cia_pairs
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
    return data_files, cia_files, database_dirs, ckd_dirs


def print_input_status(data_dir, opacity_root, ckd_root, molecules, channels, cia_pairs):
    """Print whether expected input files and directories exist."""
    data_files, cia_files, database_dirs, ckd_dirs = expected_input_paths(
        data_dir, opacity_root, ckd_root, molecules, channels, cia_pairs
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

    print("ExoMolOP CKD directories:")
    for key, path in ckd_dirs.items():
        directory_exists = os.path.isdir(path)
        print(f"  {key}: {path} ({'ok' if directory_exists else 'missing'})")
        h5_status = (
            "ok"
            if directory_exists and glob.glob(os.path.join(path, "*.h5"))
            else "missing"
            if directory_exists
            else "not checked"
        )
        print(f"  {key} h5 table: {os.path.join(path, '*.h5')} ({h5_status})")


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
    print_input_status(
        args.data_dir,
        args.opacity_root,
        args.ckd_root,
        selected_molecules,
        selected_channels,
        selected_cia_pairs,
    )
    sys.exit(0)

observed_spectra = load_observed_spectra(args.data_dir)
observed_spectra = select_observed_channels(observed_spectra, selected_channels)
wav_obs_all, rp_mean_all, rp_std_all, channel_index_all = concatenate_observed_spectra(
    observed_spectra
)

if args.summarize_data:
    summarize_observed_spectra(observed_spectra, wav_obs_all)
    print("Selected molecules: " + ", ".join(selected_molecules))
    print("Selected channels: " + ", ".join(selected_channels))
    print("Selected CIA pairs: " + ", ".join(selected_cia_pairs))
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

wav_obs_nirspec = None
rp_mean_nirspec = None
rp_std_nirspec = None
if nirspec_g395h is not None:
    wav_obs_nirspec = nirspec_g395h.wavelength_nm
    rp_mean_nirspec = nirspec_g395h.radius_ratio
    rp_std_nirspec = nirspec_g395h.radius_ratio_error


# Convert from wavelength to wavenumber for modelling.
if not args.plot_data_only:
    inst_fit_nus = wav2nu(wav_obs_fit, "nm")
    if args.opacity_mode != "ckd":
        inst_nirspec_nus = wav2nu(wav_obs_nirspec, "nm")
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

ckdpath_map_ExomolOP = {
    mol: os.path.join(args.ckd_root, relative_path)
    for mol, relative_path in CKD_RELATIVE_PATHS.items()
}
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
            path = "opa_" + mol + ".zarr"
            if not os.path.exists(path):
                missing.append(f"preMODIT snapshot {mol}: {path}")

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
        opa = OpaPremodit.from_saved_opa("opa_" + mol + ".zarr", strict=False)
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
            print(f"  * {mol} (ExoMolOP CKD)")
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


validate_required_inputs()
opa_mols, molmass_arr = load_molecular_opacities()
if args.opacity_mode == "ckd":
    ckd_reference = next(iter(opa_mols.values()))
    ckd_nu_bands = ckd_reference.nu_bands
    ckd_weights = ckd_reference.ckd_info.weights
    for mol, opa in opa_mols.items():
        if len(opa.nu_bands) != len(ckd_nu_bands):
            raise ValueError(f"CKD band count mismatch for {mol}.")
        if not np.allclose(np.asarray(opa.nu_bands), np.asarray(ckd_nu_bands)):
            raise ValueError(f"CKD band centers mismatch for {mol}.")
    validate_ckd_band_coverage(ckd_nu_bands, ckd_nurange)
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

    fiducial_mu = spectral_model(
        radius_btm=1.27 * RJ,
        Mp=Mp_mean * MJ,
        Rstar=Rstar_mean * Rs,
        RV=-50.0,
        vmr_arr=fiducial_vmr,
        T0=1200.0,
        logP_cloud=-3.0,
    )
    fiducial_mu = fiducial_mu if args.opacity_mode == "ckd" else fiducial_mu[::-1]
    fiducial_mu_np = np.asarray(jax.device_get(fiducial_mu))

    print("Forward-model check:")
    print(f"  data_mode: {args.data_mode}")
    print(f"  opacity_mode: {args.opacity_mode}")
    print("  molecules: " + ", ".join(opa_mols.keys()))
    print("  channels: " + ", ".join(selected_channels))
    print("  CIA pairs: " + ", ".join(selected_cia_pairs))
    print(f"  observed shape: {rp_mean_fit.shape}")
    print(f"  model shape: {fiducial_mu_np.shape}")
    print(f"  finite model: {np.all(np.isfinite(fiducial_mu_np))}")
    print(
        "  model Rp/Rs range: "
        f"[{np.min(fiducial_mu_np):.6f}, {np.max(fiducial_mu_np):.6f}]"
    )
    if fiducial_mu_np.shape != rp_mean_fit.shape:
        raise ValueError(
            "Forward-model shape does not match the selected observed data vector."
        )
    if not np.all(np.isfinite(fiducial_mu_np)):
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


def prior_guide(rp_mean, rp_std):
    """Guide for Mp and Rs so they follow their priors during SVI."""
    Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0.0))
    Rs = numpyro.sample("Rs", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0.0))
    return {"Mp": Mp, "Rs": Rs}


def build_guide():
    """Construct guide with separated priors for Mp/Rs and AutoMVN for the rest."""
    guide = AutoGuideList(model_c)
    guide.append(prior_guide)
    # Hide rp_mu so the Auto guide only sees latent sample sites
    model_hidden = handlers.block(model_c, hide=["Mp", "Rs", "rp_mu"])
    guide.append(AutoMultivariateNormal(model_hidden))
    return guide


def save_run_config(output_dir):
    """Persist the retrieval configuration for reproducibility."""
    config = vars(args).copy()
    config["molecules"] = list(selected_molecules)
    config["channels"] = list(selected_channels)
    config["channel_summary"] = observed_channel_summary(observed_spectra)
    config["cia_pairs"] = list(selected_cia_pairs)
    config["data_dir"] = args.data_dir
    config["opacity_root"] = args.opacity_root
    config["ckd_root"] = args.ckd_root
    config["rv_min_kms"] = args.rv_min
    config["rv_max_kms"] = args.rv_max
    if ckd_nurange is not None:
        config["ckd_nurange_cm-1"] = [float(ckd_nurange[0]), float(ckd_nurange[1])]
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
    )

    params = svi_result.params
    losses = svi_result.losses

    # Median of the AutoMVN part in the constrained space.
    svi_median = guide[-1].median(params)
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
mcmc.print_summary()
with open(os.path.join(DIR_SAVE, "mcmc_summary.txt"), "w") as f:
    with redirect_stdout(f):
        mcmc.print_summary()

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
rng_key, rng_plot = random.split(rng_key)
svi_pred = Predictive(model_c, params=svi_median, num_samples=1, return_sites=["rp_mu"])
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
    # Draw samples from the SVI guide for visualization only when needed.
    rng_key, rng_svi = random.split(rng_key)
    svi_samples = svi_guide[-1].sample_posterior(
        rng_svi,
        _svi_params,
        rp_mean=rp_mean_fit,
        rp_std=rp_std_fit,
        sample_shape=(args.svi_plot_samples,),
    )

    corner_plot_path = os.path.join(DIR_SAVE, "corner_plot_svi.png")
    plot_corner(svi_samples=svi_samples, variables=corner_vars, save_path=corner_plot_path)

    hmc_corner_plot_path = os.path.join(DIR_SAVE, "corner_plot.png")
    plot_corner(
        hmc_samples=posterior_sample, variables=corner_vars, save_path=hmc_corner_plot_path
    )

    hmc_svi_corner_overlay_path = os.path.join(DIR_SAVE, "corner_plot_hmc_svi_overlay.png")
    plot_corner(
        hmc_samples=posterior_sample,
        svi_samples=svi_samples,
        variables=corner_vars,
        save_path=hmc_svi_corner_overlay_path,
    )
