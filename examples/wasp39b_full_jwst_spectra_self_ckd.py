"""
[transmission/CKD] WASP-39 b self-CKD vs ExoMolOP comparison
================================================================

This example is a focused forward-model comparison for the WASP-39 b JWST
transmission data.  It builds an in-house CKD table from an ExoJAX line-opacity
calculator and compares the resulting transmission spectrum with the ExoMolOP
R=1000 CKD table at the same observed wavelength points.

The default local line list is the bundled H2O ExoMol SAMPLE database, which
covers only 4301-4399 cm-1.  The script therefore trims the observed WASP-39 b
data to the wavelength points covered by the self-CKD grid.  Use
``--self-nu-min``/``--self-nu-max`` and ``--self-mdb-path`` with a full database
to run a wider comparison.

Recommended smoke run from the repository root::

    NUMBA_DISABLE_JIT=1 python examples/wasp39b_full_jwst_spectra_self_ckd.py \
      --molecule H2O --ckd-resolution 1000 --ng 16 --nu-grid-points 1200 \
      --t-grid-size 6 --p-grid-size 6 --max-observed 32

The ``NUMBA_DISABLE_JIT=1`` setting avoids a RADIS/numba cache issue seen in
some editable or egg-based environments.

For the WASP-39 b H2O product comparison, first build R=1000 and R=3000 patch
tables with ``--build-self-patches``.  Then run the R=1000 manifest as
``--self-patch-manifest`` and pass the R=3000 manifest through
``--product-r3000-self-patch-manifest``.  The script writes the usual
``comparison_data.npz`` plus full-range and 3000-3200 nm product figures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np


DEFAULT_MOLECULE = "H2O"
DEFAULT_RV_FIXED_KMS = -83.18
CKD_RELATIVE_PATHS = {
    "H2O": os.path.join("H2O", "1H2-16O", "POKAZATEL"),
    "CO2": os.path.join("CO2", "12C-16O2", "UCL-4000"),
}
SELF_MDB_PATHS = {
    "H2O": os.path.join("H2O", "1H2-16O", "SAMPLE"),
}
CIA_RELATIVE_FILES = {
    "H2H2": os.path.join(".db_CIA", "H2-H2_2011.cia"),
}


@dataclass(frozen=True)
class ObservedSpectrum:
    name: str
    wavelength_nm: np.ndarray
    radius_ratio: np.ndarray
    radius_ratio_error_low: np.ndarray
    radius_ratio_error_high: np.ndarray

    @property
    def radius_ratio_error(self) -> np.ndarray:
        return 0.5 * (self.radius_ratio_error_low + self.radius_ratio_error_high)


def parser_with_defaults() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare WASP-39b ExoMolOP CKD and self-generated CKD.",
        allow_abbrev=False,
    )
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE, choices=sorted(CKD_RELATIVE_PATHS))
    parser.add_argument("--data-dir", default="examples/wasp39_data")
    parser.add_argument("--ckd-root", default=".database")
    parser.add_argument("--opacity-root", default="examples/path_to")
    parser.add_argument("--self-mdb-path", default="")
    parser.add_argument("--self-table", default="")
    parser.add_argument(
        "--self-patch-manifest",
        default="",
        help="Use patch CKD tables from this manifest for the self CKD forward model.",
    )
    parser.add_argument(
        "--product-r3000-self-patch-manifest",
        default="",
        help="Also forward this self CKD manifest and write ExoMolOP/R1000/R3000 product plots.",
    )
    parser.add_argument("--output-dir", default="output_wasp39b_self_ckd_compare")
    parser.add_argument("--overwrite-self-table", action="store_true")
    parser.add_argument(
        "--build-self-patches",
        action="store_true",
        help="Build self CKD tables patch-by-patch and write a manifest, then exit.",
    )
    parser.add_argument(
        "--patch-width",
        type=float,
        default=None,
        help="Patch width in cm-1 for --build-self-patches.",
    )
    parser.add_argument(
        "--patch-nu-grid-points",
        type=int,
        default=None,
        help="Wavenumber grid points per patch. Defaults to scaling --nu-grid-points by patch width.",
    )
    parser.add_argument(
        "--patch-manifest",
        default="",
        help="Manifest filename for --build-self-patches. Defaults inside --output-dir.",
    )
    parser.add_argument("--ckd-resolution", type=float, default=1000.0)
    parser.add_argument("--ckd-band-width", type=float, default=None)
    parser.add_argument("--band-spacing", choices=("log", "linear"), default="log")
    parser.add_argument("--ng", type=int, default=16)
    parser.add_argument("--nu-grid-points", type=int, default=1200)
    parser.add_argument("--self-nu-min", type=float, default=4301.0)
    parser.add_argument("--self-nu-max", type=float, default=4399.0)
    parser.add_argument("--t-grid-size", type=int, default=6)
    parser.add_argument("--p-grid-size", type=int, default=6)
    parser.add_argument("--nlayer", type=int, default=120)
    parser.add_argument("--pressure-top", type=float, default=1.0e-11)
    parser.add_argument("--pressure-btm", type=float, default=1.0e1)
    parser.add_argument("--temperature", type=float, default=1200.0)
    parser.add_argument("--log-vmr", type=float, default=-5.0)
    parser.add_argument(
        "--background-vmr",
        default="",
        help="Comma list of background VMRs as MOL=log10VMR. These affect MMW but not opacity.",
    )
    parser.add_argument("--logp-cloud", type=float, default=-3.0)
    parser.add_argument("--cloud-tau", type=float, default=50.0)
    parser.add_argument("--radius-btm-rj", type=float, default=1.27)
    parser.add_argument("--mass-mj", type=float, default=0.281)
    parser.add_argument("--rstar-rs", type=float, default=0.939)
    parser.add_argument("--rv-fixed", type=float, default=DEFAULT_RV_FIXED_KMS)
    parser.add_argument("--cia-pairs", default="H2H2", help="Comma list or 'none'.")
    parser.add_argument("--max-observed", type=int, default=None)
    parser.add_argument("--product-zoom-min-nm", type=float, default=3000.0)
    parser.add_argument("--product-zoom-max-nm", type=float, default=3200.0)
    parser.add_argument("--jax-platform", choices=("auto", "cpu", "gpu", "tpu"), default="cpu")
    parser.add_argument("--skip-plot", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.ckd_resolution <= 0.0:
        parser.error("--ckd-resolution must be positive.")
    if args.ckd_band_width is not None and args.ckd_band_width <= 0.0:
        parser.error("--ckd-band-width must be positive.")
    for name in ("ng", "nu_grid_points", "t_grid_size", "p_grid_size", "nlayer"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.self_nu_min <= 0.0 or args.self_nu_max <= 0.0:
        parser.error("--self-nu-min and --self-nu-max must be positive.")
    if args.self_nu_min >= args.self_nu_max:
        parser.error("--self-nu-min must be smaller than --self-nu-max.")
    if args.build_self_patches:
        if args.patch_width is None or args.patch_width <= 0.0:
            parser.error("--build-self-patches requires positive --patch-width.")
        if args.patch_nu_grid_points is not None and args.patch_nu_grid_points <= 0:
            parser.error("--patch-nu-grid-points must be positive when set.")
        if args.self_table:
            parser.error("--self-table cannot be used with --build-self-patches.")
        if args.self_patch_manifest:
            parser.error("--self-patch-manifest cannot be used with --build-self-patches.")
        if args.product_r3000_self_patch_manifest:
            parser.error("--product-r3000-self-patch-manifest cannot be used with --build-self-patches.")
    if args.product_r3000_self_patch_manifest and not math.isclose(args.ckd_resolution, 1000.0):
        parser.error("--product-r3000-self-patch-manifest expects the primary self run to be R=1000.")
    if args.self_patch_manifest and args.self_table:
        parser.error("--self-table cannot be used with --self-patch-manifest.")
    if args.pressure_top <= 0.0 or args.pressure_btm <= 0.0:
        parser.error("--pressure-top and --pressure-btm must be positive.")
    if args.pressure_top >= args.pressure_btm:
        parser.error("--pressure-top must be smaller than --pressure-btm.")
    if args.max_observed is not None and args.max_observed <= 0:
        parser.error("--max-observed must be positive when set.")
    if args.product_zoom_min_nm >= args.product_zoom_max_nm:
        parser.error("--product-zoom-min-nm must be smaller than --product-zoom-max-nm.")
    try:
        parse_background_vmr(args.background_vmr)
    except ValueError as exc:
        parser.error(str(exc))


def parse_background_vmr(text: str) -> dict[str, float]:
    entries = {}
    if not text.strip():
        return entries
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError("--background-vmr entries must be MOL=log10VMR.")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("--background-vmr entries must include a molecule name.")
        try:
            entries[name] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid log10VMR for --background-vmr {name}.") from exc
    return entries


def configure_runtime(args: argparse.Namespace) -> None:
    if args.jax_platform != "auto":
        platform_name = "cuda" if args.jax_platform == "gpu" else args.jax_platform
        os.environ.setdefault("JAX_PLATFORMS", platform_name)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "exojax_matplotlib"))


def resolve_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    repo_relative = Path(__file__).resolve().parents[1] / candidate
    return str(repo_relative if repo_relative.exists() else candidate)


def load_observed_spectra(data_dir: str) -> dict[str, ObservedSpectrum]:
    import h5py

    data = Path(data_dir)
    wav_niriss_1, rp_niriss_1, err_low_niriss_1, err_high_niriss_1 = np.loadtxt(
        data / "niriss_order1.txt", unpack=True
    )
    wav_niriss_2, rp_niriss_2, err_low_niriss_2, err_high_niriss_2 = np.loadtxt(
        data / "niriss_order2.txt", unpack=True
    )
    wav_nirspec = np.load(data / "wavelength.npy")
    rp_nirspec = np.load(data / "wasp39b_nirspec_g395h_rp_mean.npy")
    err_nirspec = np.load(data / "wasp39b_nirspec_g395h_rp_std.npy")

    with h5py.File(data / "miri.h5", "r") as handle:
        dppm = np.array(handle["dppm"])
        dppm_err = np.array(handle["dppm_error"])
        wav_miri_micron = np.array(handle["wavelength"])

    rp_miri = np.sqrt(dppm * 1.0e-6)
    err_miri = dppm_err * 1.0e-6 / (2.0 * rp_miri)
    return {
        "niriss_order1": ObservedSpectrum(
            "NIRISS Order 1", wav_niriss_1, rp_niriss_1, err_low_niriss_1, err_high_niriss_1
        ),
        "niriss_order2": ObservedSpectrum(
            "NIRISS Order 2", wav_niriss_2, rp_niriss_2, err_low_niriss_2, err_high_niriss_2
        ),
        "nirspec_g395h": ObservedSpectrum(
            "NIRSpec G395H", wav_nirspec, rp_nirspec, err_nirspec, err_nirspec
        ),
        "miri_lrs": ObservedSpectrum(
            "MIRI LRS", wav_miri_micron * 1000.0, rp_miri, err_miri, err_miri
        ),
    }


def concatenate_observed_spectra(
    observed: dict[str, ObservedSpectrum],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = list(observed)
    wavelength = np.concatenate([observed[name].wavelength_nm for name in names])
    radius_ratio = np.concatenate([observed[name].radius_ratio for name in names])
    radius_ratio_error = np.concatenate([observed[name].radius_ratio_error for name in names])
    channel = np.concatenate(
        [np.full(observed[name].wavelength_nm.shape, i, dtype=int) for i, name in enumerate(names)]
    )
    order = np.argsort(wavelength)
    return wavelength[order], radius_ratio[order], radius_ratio_error[order], channel[order]


def select_covered_observations(
    wavelength_nm: np.ndarray,
    radius_ratio: np.ndarray,
    radius_ratio_error: np.ndarray,
    channel: np.ndarray,
    nu_min: float,
    nu_max: float,
    limit: int | None,
    radial_velocity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from exojax.utils.constants import c

    nu_obs = 1.0e7 / wavelength_nm
    shifted_nu_obs = nu_obs * (1.0 + radial_velocity / c)
    mask = (shifted_nu_obs >= nu_min) & (shifted_nu_obs <= nu_max)
    if not np.any(mask):
        raise ValueError(
            "No observed WASP-39b points are covered by the requested self CKD range "
            f"{nu_min:.3f}-{nu_max:.3f} cm-1 after radial-velocity shift."
        )
    wavelength_nm = wavelength_nm[mask]
    radius_ratio = radius_ratio[mask]
    radius_ratio_error = radius_ratio_error[mask]
    channel = channel[mask]
    if limit is not None and limit < wavelength_nm.size:
        idx = np.unique(np.linspace(0, wavelength_nm.size - 1, limit, dtype=int))
        wavelength_nm = wavelength_nm[idx]
        radius_ratio = radius_ratio[idx]
        radius_ratio_error = radius_ratio_error[idx]
        channel = channel[idx]
    return wavelength_nm, radius_ratio, radius_ratio_error, channel


def ckd_h5_path(args: argparse.Namespace) -> str:
    directory = Path(args.ckd_root) / CKD_RELATIVE_PATHS[args.molecule]
    h5_paths = sorted(path for path in directory.glob("*.h5") if path.stat().st_size > 0)
    if len(h5_paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one non-empty ExoMolOP h5 table in {directory}; "
            f"found {len(h5_paths)}."
        )
    return str(h5_paths[0])


def self_band_width(args: argparse.Namespace) -> float:
    if args.ckd_band_width is not None:
        return args.ckd_band_width
    return math.sqrt(args.self_nu_min * args.self_nu_max) / args.ckd_resolution


def make_wavenumber_grid(args: argparse.Namespace):
    from exojax.utils.grids import wavenumber_grid

    wav_left = 1.0e7 / args.self_nu_max
    wav_right = 1.0e7 / args.self_nu_min
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Both input wavelength", category=UserWarning)
        return wavenumber_grid(
            wav_left,
            wav_right,
            args.nu_grid_points,
            unit="nm",
            xsmode="premodit",
        )


def build_self_ckd_patches(args: argparse.Namespace) -> None:
    from exojax.opacity.ckd.precompute import precompute_ckd_tables_by_patches

    out_dir = Path(args.output_dir)
    t_grid = np.linspace(500.0, 2000.0, args.t_grid_size)
    p_grid = np.logspace(np.log10(args.pressure_top), np.log10(args.pressure_btm), args.p_grid_size)
    manifest_name = Path(args.patch_manifest).name if args.patch_manifest else "self_ckd_patch_manifest.json"
    points_per_patch = args.patch_nu_grid_points
    if points_per_patch is None:
        points_per_patch = max(
            16,
            int(math.ceil(args.nu_grid_points * args.patch_width / (args.self_nu_max - args.self_nu_min))),
        )

    def make_patch_nu_grid(nu_min, nu_max, n_grid):
        patch_args = argparse.Namespace(**vars(args))
        patch_args.self_nu_min = float(nu_min)
        patch_args.self_nu_max = float(nu_max)
        patch_args.nu_grid_points = int(n_grid)
        nu_grid, _wav_grid, _grid_resolution = make_wavenumber_grid(patch_args)
        return nu_grid

    def make_patch_base_opa(nu_grid, nu_min, nu_max):
        patch_args = argparse.Namespace(**vars(args))
        patch_args.self_nu_min = float(nu_min)
        patch_args.self_nu_max = float(nu_max)
        base_opa, _molmass, _mdb_path, _n_lines = build_self_base_opa(patch_args, nu_grid)
        return base_opa

    precompute_ckd_tables_by_patches(
        make_patch_base_opa,
        make_patch_nu_grid,
        args.self_nu_min,
        args.self_nu_max,
        args.patch_width,
        t_grid,
        p_grid,
        out_dir,
        Ng=args.ng,
        ckd_resolution=args.ckd_resolution,
        band_spacing=args.band_spacing,
        nu_grid_points_per_patch=points_per_patch,
        overwrite=args.overwrite_self_table,
        manifest_name=manifest_name,
        table_prefix=f"self_ckd_{args.molecule}",
    )
    print(f"Self CKD patch build complete: {out_dir / manifest_name}")


def build_self_base_opa(args: argparse.Namespace, nu_grid):
    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaPremodit

    mdb_path = args.self_mdb_path or SELF_MDB_PATHS.get(args.molecule)
    if not mdb_path:
        raise ValueError(
            f"No default self line-list path is configured for {args.molecule}. "
            "Provide --self-mdb-path."
        )
    mdb_path = resolve_path(mdb_path)
    mdb = MdbExomol(
        mdb_path,
        nurange=nu_grid,
        gpu_transfer=False,
        local_databases=".",
        broadf_download=False,
    )
    opa = OpaPremodit(
        mdb,
        nu_grid,
        auto_trange=[500.0, 2000.0],
        allow_32bit=True,
    )
    return opa, float(mdb.molmass), mdb_path, int(len(mdb.nu_lines))


def load_or_build_self_ckd(args: argparse.Namespace, base_opa):
    from exojax.opacity import OpaCKD

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = Path(
        args.self_table
        or out_dir / f"self_ckd_{args.molecule}_R{int(args.ckd_resolution)}.npz"
    )
    band_width = self_band_width(args)
    ckd = OpaCKD(base_opa, Ng=args.ng, band_width=band_width, band_spacing=args.band_spacing)
    if table_path.exists() and not args.overwrite_self_table:
        ckd.load_tables(str(table_path), base_opa=base_opa)
        return ckd, str(table_path), False

    t_grid = np.linspace(500.0, 2000.0, args.t_grid_size)
    p_grid = np.logspace(np.log10(args.pressure_top), np.log10(args.pressure_btm), args.p_grid_size)
    ckd.precompute_tables(t_grid, p_grid, to_path=str(table_path), overwrite=True)
    return ckd, str(table_path), True


def load_self_patch_manifest(path: str) -> dict:
    manifest_path = Path(resolve_path(path))
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    schema_version = manifest.get("schema_version")
    if schema_version == "wasp39b_self_ckd_patch_manifest.v1":
        manifest["tables"] = [
            {
                "index": patch["index"],
                "nu_min_cm-1": patch["nu_min_cm-1"],
                "nu_max_cm-1": patch["nu_max_cm-1"],
                "path": patch["table"],
                "n_bands": patch["n_bands"],
                "band_width_cm-1": patch["band_width_cm-1"],
            }
            for patch in manifest["patches"]
        ]
        manifest["nu_min_cm-1"] = manifest["self_nu_min_cm-1"]
        manifest["nu_max_cm-1"] = manifest["self_nu_max_cm-1"]
        manifest["Ng"] = manifest["ng"]
        manifest["ckd_resolution"] = manifest["ckd_resolution_requested"]
        manifest["_self_molmass"] = manifest["patches"][0].get("molmass")
        manifest["_self_mdb_path"] = manifest.get("self_mdb_path", "")
    elif schema_version != "ckd_patch_manifest.v1":
        raise ValueError(f"Unsupported CKD patch manifest schema: {manifest_path}")
    tables = manifest.get("tables", [])
    if not tables:
        raise ValueError(f"CKD patch manifest contains no tables: {manifest_path}")
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_manifest_dir"] = str(manifest_path.parent)
    return manifest


def patch_table_path(manifest: dict, table: dict) -> str:
    raw_path = Path(table["path"]).expanduser()
    if raw_path.is_absolute() or raw_path.exists():
        return str(raw_path)
    manifest_sibling = Path(manifest["_manifest_dir"]) / raw_path.name
    if manifest_sibling.exists():
        return str(manifest_sibling)
    manifest_relative = Path(manifest["_manifest_dir"]) / raw_path
    if manifest_relative.exists():
        return str(manifest_relative)
    return str(raw_path)


def ckd_edge_range(opa_ckd) -> tuple[float, float]:
    band_edges = np.asarray(opa_ckd.band_edges, dtype=float)
    return float(np.min(band_edges)), float(np.max(band_edges))


def compute_transmission_from_self_patches(
    args: argparse.Namespace,
    art,
    manifest: dict,
    molmass: float,
    wav_obs_nm: np.ndarray,
):
    import jax
    from exojax.opacity import OpaCKD
    from exojax.postproc.ckd import validate_ckd_band_coverage
    from exojax.utils.constants import c

    shifted_nu_obs = (1.0e7 / wav_obs_nm) * (1.0 + args.rv_fixed / c)
    self_mu = np.full(wav_obs_nm.shape, np.nan)
    patch_reports = []

    for table in sorted(manifest["tables"], key=lambda item: item["nu_min_cm-1"]):
        table_path = patch_table_path(manifest, table)
        ckd = OpaCKD.load_only().load_tables(table_path)
        ckd.molmass = molmass
        edge_min, edge_max = ckd_edge_range(ckd)
        mask = np.isnan(self_mu) & (shifted_nu_obs >= edge_min) & (shifted_nu_obs <= edge_max)
        n_observed = int(np.count_nonzero(mask))
        patch_reports.append(
            {
                "index": int(table["index"]),
                "path": table_path,
                "nu_min_cm-1": float(table["nu_min_cm-1"]),
                "nu_max_cm-1": float(table["nu_max_cm-1"]),
                "edge_min_cm-1": edge_min,
                "edge_max_cm-1": edge_max,
                "n_bands": int(np.asarray(ckd.nu_bands).size),
                "n_observed": n_observed,
            }
        )
        if n_observed == 0:
            del ckd
            continue
        validate_ckd_band_coverage(ckd.nu_bands, (edge_min, edge_max), ckd.band_edges)
        self_mu[mask] = compute_transmission(
            args,
            art,
            ckd,
            molmass,
            build_cia_opacities(args, ckd.nu_bands),
            wav_obs_nm[mask],
        )
        del ckd
        jax.clear_caches()

    if np.any(~np.isfinite(self_mu)):
        missing = np.where(~np.isfinite(self_mu))[0]
        missing_nu = shifted_nu_obs[missing]
        raise ValueError(
            "Self CKD patch tables do not cover all selected observed points. "
            f"Missing {missing.size} points, shifted nu range="
            f"{np.min(missing_nu):.3f}-{np.max(missing_nu):.3f} cm-1."
        )

    return self_mu, patch_reports


def build_cia_opacities(args: argparse.Namespace, nu_bands):
    if args.cia_pairs.strip().lower() == "none":
        return {}

    from exojax.database.cia.api import CdbCIA
    from exojax.opacity.opacont import OpaCIA

    opacities = {}
    for pair in [part.strip() for part in args.cia_pairs.split(",") if part.strip()]:
        if pair not in CIA_RELATIVE_FILES:
            raise ValueError(f"Unsupported CIA pair for this example: {pair}")
        path = Path(args.opacity_root) / CIA_RELATIVE_FILES[pair]
        if not path.exists():
            raise FileNotFoundError(f"CIA file not found: {path}")
        opacities[pair] = OpaCIA(CdbCIA(str(path), nurange=nu_bands), nu_grid=nu_bands)
    return opacities


def compute_transmission(
    args: argparse.Namespace,
    art,
    opa_ckd,
    molmass: float,
    cia_opacities,
    wav_obs_nm: np.ndarray,
):
    import jax
    import jax.numpy as jnp

    from exojax.database import molinfo
    from exojax.postproc.ckd import sample_ckd_bands_at_wavelengths
    from exojax.utils.astrofunc import gravity_jupiter
    from exojax.utils.constants import MJ, RJ, Rs

    temperature = args.temperature * jnp.ones_like(art.pressure)
    vmr_mol = art.constant_profile(10.0 ** args.log_vmr)
    background_vmr = {
        name: art.constant_profile(10.0**log_vmr)
        for name, log_vmr in parse_background_vmr(args.background_vmr).items()
    }
    vmr_background_total = sum(background_vmr.values(), jnp.zeros_like(vmr_mol))
    vmr_total = jnp.clip(vmr_mol + vmr_background_total, 0.0, 1.0)
    vmr_h2 = (1.0 - vmr_total) * 6.0 / 7.0
    vmr_he = (1.0 - vmr_total) * 1.0 / 7.0
    mmw = (
        molinfo.molmass_isotope("H2") * vmr_h2
        + molinfo.molmass_isotope("He", db_HIT=False) * vmr_he
        + molmass * vmr_mol
    )
    for name, vmr in background_vmr.items():
        mmw = mmw + molinfo.molmass_isotope(name) * vmr

    radius_btm = args.radius_btm_rj * RJ
    mass = args.mass_mj * MJ
    rstar = args.rstar_rs * Rs
    gravity_btm = gravity_jupiter(args.radius_btm_rj, args.mass_mj)
    gravity = art.gravity_profile(temperature, mmw, radius_btm, gravity_btm)

    nband = opa_ckd.nu_bands.size
    ng = opa_ckd.ckd_info.weights.size
    pressure_log = jnp.log10(art.pressure)
    width_cloud = 1.0 / 25.0
    delta_log_pressure = -jnp.log10(art.pressure_decrease_rate)
    dtau_c = args.cloud_tau * delta_log_pressure / width_cloud
    cloud_profile = (pressure_log[:, None] - args.logp_cloud) / width_cloud
    dtau_cloud = dtau_c / jnp.sqrt(jnp.pi) * jnp.exp(-jnp.clip(cloud_profile**2, -50, 50))
    dtau_ckd = jnp.broadcast_to(dtau_cloud[:, None], (pressure_log.size, ng, nband))

    for pair, cia in cia_opacities.items():
        logacia_matrix = cia.logacia_matrix(temperature)
        if pair == "H2H2":
            vmr_x, vmr_y = vmr_h2, vmr_h2
        else:
            continue
        dtau_cia = art.opacity_profile_cia(logacia_matrix, temperature, vmr_x, vmr_y, mmw, gravity)
        dtau_ckd = dtau_ckd + dtau_cia[:, None, :]

    xs_ckd = opa_ckd.xstensor_ckd(temperature, art.pressure)
    dtau_ckd = dtau_ckd + art.opacity_profile_xs_ckd(xs_ckd, vmr_mol, mmw, gravity)
    rp2_bands = art.run_ckd(
        dtau_ckd,
        temperature,
        mmw,
        radius_btm,
        gravity_btm,
        opa_ckd.ckd_info.weights,
    )
    rp2_sample = sample_ckd_bands_at_wavelengths(
        opa_ckd.nu_bands,
        rp2_bands,
        wav_obs_nm,
        radial_velocity=args.rv_fixed,
        unit="nm",
    )
    return np.asarray(jax.device_get(jnp.sqrt(rp2_sample) * (radius_btm / rstar)))


def comparison_summary(args, wav, obs, err, self_mu, external_mu, self_ckd, external_ckd, paths):
    diff = self_mu - external_mu
    diff_ppm = diff * 1.0e6
    self_n_bands = paths.get("self_n_bands")
    if self_n_bands is None:
        self_n_bands = int(np.asarray(self_ckd.nu_bands).size)
    return {
        "molecule": args.molecule,
        "n_observed": int(wav.size),
        "wavelength_nm_min": float(np.min(wav)),
        "wavelength_nm_max": float(np.max(wav)),
        "self_nu_min_cm-1": float(args.self_nu_min),
        "self_nu_max_cm-1": float(args.self_nu_max),
        "ckd_resolution_requested": float(args.ckd_resolution),
        "self_band_width_cm-1": float(self_band_width(args)),
        "ng": int(args.ng),
        "nlayer": int(args.nlayer),
        "pressure_top_bar": float(args.pressure_top),
        "pressure_btm_bar": float(args.pressure_btm),
        "background_vmr_log10": parse_background_vmr(args.background_vmr),
        "self_n_bands": int(self_n_bands),
        "external_n_bands": int(np.asarray(external_ckd.nu_bands).size),
        "self_table": paths["self_table"],
        "external_table": paths["external_table"],
        "self_table_created": bool(paths["self_table_created"]),
        "self_mdb_path": paths["self_mdb_path"],
        "self_mdb_n_lines": int(paths["self_mdb_n_lines"]),
        "rprs_self_min": float(np.min(self_mu)),
        "rprs_self_max": float(np.max(self_mu)),
        "rprs_exomolop_min": float(np.min(external_mu)),
        "rprs_exomolop_max": float(np.max(external_mu)),
        "delta_rprs_mean_ppm": float(np.mean(diff_ppm)),
        "delta_rprs_rms_ppm": float(np.sqrt(np.mean(diff_ppm**2))),
        "delta_rprs_max_abs_ppm": float(np.max(np.abs(diff_ppm))),
        "observed_rprs_mean": float(np.mean(obs)),
        "observed_error_mean": float(np.mean(err)),
        "finite": bool(
            np.all(np.isfinite(self_mu))
            and np.all(np.isfinite(external_mu))
            and np.all(np.isfinite(diff))
        ),
    }


def save_outputs(args, wav, obs, err, channel, self_mu, external_mu, summary):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "comparison_data.npz",
        wavelength_nm=wav,
        observed_rprs=obs,
        observed_error=err,
        channel_index=channel,
        self_rprs=self_mu,
        exomolop_rprs=external_mu,
        delta_rprs=self_mu - external_mu,
    )
    with open(out_dir / "comparison_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.skip_plot:
        return

    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax0.errorbar(wav, obs, yerr=err, fmt=".", color="0.5", alpha=0.5, label="Observed")
    ax0.plot(wav, external_mu, "o", ms=4, label="ExoMolOP CKD R1000")
    ax0.plot(wav, self_mu, "s", ms=4, label=f"Self CKD R{args.ckd_resolution:.0f}")
    ax0.set_ylabel("Rp/Rs")
    ax0.legend()
    ax1.axhline(0.0, color="0.3", lw=1)
    ax1.plot(wav, (self_mu - external_mu) * 1.0e6, "o", ms=4)
    ax1.set_xlabel("Wavelength [nm]")
    ax1.set_ylabel("Self - ExoMolOP [ppm in Rp/Rs]")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_self_vs_exomolop.png", dpi=200)
    plt.close(fig)


def product_stats(wav, obs, external_mu, self_r1000_mu, self_r3000_mu, mask):
    stats = {"n": int(np.count_nonzero(mask))}
    if stats["n"] == 0:
        return stats
    stats["wavelength_nm_min"] = float(np.min(wav[mask]))
    stats["wavelength_nm_max"] = float(np.max(wav[mask]))
    for label, values in [
        ("observed", obs),
        ("exomolop", external_mu),
        ("self_R1000", self_r1000_mu),
        ("self_R3000", self_r3000_mu),
    ]:
        selected = values[mask]
        stats[label] = {
            "mean": float(np.mean(selected)),
            "p95_minus_p05_ppm": float(
                (np.percentile(selected, 95) - np.percentile(selected, 5)) * 1.0e6
            ),
            "peak_to_peak_ppm": float((np.max(selected) - np.min(selected)) * 1.0e6),
        }
    for label, values in [
        ("self_R1000_minus_exomolop", self_r1000_mu - external_mu),
        ("self_R3000_minus_exomolop", self_r3000_mu - external_mu),
        ("self_R3000_minus_self_R1000", self_r3000_mu - self_r1000_mu),
    ]:
        selected_ppm = values[mask] * 1.0e6
        stats[label] = {
            "mean_ppm": float(np.mean(selected_ppm)),
            "rms_ppm": float(np.sqrt(np.mean(selected_ppm**2))),
            "max_abs_ppm": float(np.max(np.abs(selected_ppm))),
        }
    return stats


def save_product_plot(
    path,
    wav,
    obs,
    err,
    external_mu,
    self_r1000_mu,
    self_r3000_mu,
    mask,
    title,
    logx,
):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    wav_plot = wav[mask]
    ax0.errorbar(
        wav_plot,
        obs[mask],
        yerr=err[mask],
        fmt=".",
        ms=3,
        color="0.45",
        ecolor="0.75",
        elinewidth=0.6,
        alpha=0.45,
        label="Observed",
    )
    ax0.plot(
        wav_plot,
        external_mu[mask],
        ".",
        ms=4.0,
        color="C0",
        alpha=0.5,
        label="ExoMolOP CKD R1000",
    )
    ax0.plot(
        wav_plot,
        self_r1000_mu[mask],
        ".",
        ms=4.0,
        color="C1",
        alpha=0.5,
        label="Self CKD R1000",
    )
    ax0.plot(
        wav_plot,
        self_r3000_mu[mask],
        ".",
        ms=4.0,
        color="C3",
        alpha=0.5,
        label="Self CKD R3000",
    )
    ax0.set_ylabel("Rp/Rs")
    ax0.legend(loc="upper right", frameon=True)
    ax0.set_title(title)

    ax1.axhline(0.0, color="0.25", lw=1.0)
    ax1.plot(
        wav_plot,
        (self_r1000_mu[mask] - external_mu[mask]) * 1.0e6,
        ".",
        ms=4.0,
        color="C1",
        alpha=0.5,
        label="Self R1000 - ExoMolOP",
    )
    ax1.plot(
        wav_plot,
        (self_r3000_mu[mask] - external_mu[mask]) * 1.0e6,
        ".",
        ms=4.0,
        color="C3",
        alpha=0.5,
        label="Self R3000 - ExoMolOP",
    )
    ax1.set_xlabel("Wavelength [nm]")
    ax1.set_ylabel("Delta Rp/Rs [ppm]")
    ax1.legend(loc="lower right", frameon=True)
    if logx:
        ax0.set_xscale("log")
        ax1.set_xscale("log")
    else:
        ax1.set_xlim(float(np.min(wav_plot)), float(np.max(wav_plot)))
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def save_product_outputs(args, wav, obs, err, external_mu, self_r1000_mu, self_r3000_mu):
    out_dir = Path(args.output_dir)
    full_mask = np.ones_like(wav, dtype=bool)
    zoom_mask = (wav >= args.product_zoom_min_nm) & (wav <= args.product_zoom_max_nm)
    if not np.any(zoom_mask):
        raise ValueError(
            "No selected observed points fall in the product zoom range "
            f"{args.product_zoom_min_nm}-{args.product_zoom_max_nm} nm."
        )

    full_base = out_dir / "comparison_exomolop_self_R1000_R3000"
    zoom_base = (
        out_dir
        / f"comparison_exomolop_self_R1000_R3000_{args.product_zoom_min_nm:.0f}_"
        f"{args.product_zoom_max_nm:.0f}nm"
    )
    save_product_plot(
        full_base,
        wav,
        obs,
        err,
        external_mu,
        self_r1000_mu,
        self_r3000_mu,
        full_mask,
        "WASP-39b H2O CKD comparison",
        logx=True,
    )
    save_product_plot(
        zoom_base,
        wav,
        obs,
        err,
        external_mu,
        self_r1000_mu,
        self_r3000_mu,
        zoom_mask,
        (
            "WASP-39b H2O CKD comparison, "
            f"{args.product_zoom_min_nm:.0f}-{args.product_zoom_max_nm:.0f} nm"
        ),
        logx=False,
    )

    stats = {
        "full": product_stats(wav, obs, external_mu, self_r1000_mu, self_r3000_mu, full_mask),
        "zoom": product_stats(wav, obs, external_mu, self_r1000_mu, self_r3000_mu, zoom_mask),
    }
    np.savez(
        out_dir / "comparison_exomolop_self_R1000_R3000_data.npz",
        wavelength_nm=wav,
        observed_rprs=obs,
        observed_error=err,
        exomolop_rprs=external_mu,
        self_r1000_rprs=self_r1000_mu,
        self_r3000_rprs=self_r3000_mu,
        delta_self_r1000_exomolop=self_r1000_mu - external_mu,
        delta_self_r3000_exomolop=self_r3000_mu - external_mu,
    )
    with open(out_dir / "comparison_exomolop_self_R1000_R3000_stats.json", "w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"  product plot: {full_base.with_suffix('.png')}")
    print(f"  product zoom: {zoom_base.with_suffix('.png')}")


def main() -> None:
    parser = parser_with_defaults()
    args = parser.parse_args()
    validate_args(args, parser)
    args.data_dir = resolve_path(args.data_dir)
    args.ckd_root = resolve_path(args.ckd_root)
    args.opacity_root = resolve_path(args.opacity_root)
    configure_runtime(args)

    if args.build_self_patches:
        build_self_ckd_patches(args)
        return

    from exojax.opacity import OpaCKD
    from exojax.postproc.ckd import validate_ckd_band_coverage
    from exojax.rt import ArtTransPure

    patch_manifest = None
    if args.self_patch_manifest:
        patch_manifest = load_self_patch_manifest(args.self_patch_manifest)
        args.self_nu_min = float(patch_manifest["nu_min_cm-1"])
        args.self_nu_max = float(patch_manifest["nu_max_cm-1"])

    observed = load_observed_spectra(args.data_dir)
    wav_all, obs_all, err_all, channel_all = concatenate_observed_spectra(observed)
    wav, obs, err, channel = select_covered_observations(
        wav_all,
        obs_all,
        err_all,
        channel_all,
        args.self_nu_min,
        args.self_nu_max,
        args.max_observed,
        args.rv_fixed,
    )

    art = ArtTransPure(
        pressure_top=args.pressure_top,
        pressure_btm=args.pressure_btm,
        nlayer=args.nlayer,
        nu_grid=None,
        warn_no_nu_grid=False,
    )
    art.change_temperature_range(500.0, 2000.0)

    external_table = ckd_h5_path(args)
    external_ckd = OpaCKD.from_external(
        "exomolop",
        external_table,
        nurange=(args.self_nu_min, args.self_nu_max),
    )
    validate_ckd_band_coverage(
        external_ckd.nu_bands,
        (args.self_nu_min, args.self_nu_max),
        external_ckd.band_edges,
    )

    nu_grid = None
    grid_resolution = None
    self_patch_reports = []
    if patch_manifest is None:
        nu_grid, _wav_grid, grid_resolution = make_wavenumber_grid(args)
        base_opa, self_molmass, self_mdb_path, self_mdb_n_lines = build_self_base_opa(
            args, nu_grid
        )
        self_ckd, self_table, self_table_created = load_or_build_self_ckd(args, base_opa)
        self_ckd.molmass = self_molmass
        validate_ckd_band_coverage(
            self_ckd.nu_bands,
            (args.self_nu_min, args.self_nu_max),
            self_ckd.band_edges,
        )
        cia_self = build_cia_opacities(args, self_ckd.nu_bands)
        self_mu = compute_transmission(args, art, self_ckd, self_molmass, cia_self, wav)
        self_n_bands = int(np.asarray(self_ckd.nu_bands).size)
    else:
        self_molmass = float(patch_manifest.get("_self_molmass") or external_ckd.molmass)
        self_table = str(Path(resolve_path(args.self_patch_manifest)))
        self_table_created = False
        self_mdb_path = patch_manifest.get("_self_mdb_path", "")
        self_mdb_n_lines = 0
        self_ckd = None
        self_mu, self_patch_reports = compute_transmission_from_self_patches(
            args, art, patch_manifest, self_molmass, wav
        )
        self_n_bands = int(sum(table["n_bands"] for table in patch_manifest["tables"]))

    cia_external = build_cia_opacities(args, external_ckd.nu_bands)
    external_mu = compute_transmission(
        args,
        art,
        external_ckd,
        float(external_ckd.molmass),
        cia_external,
        wav,
    )

    paths = {
        "self_table": self_table,
        "self_table_created": self_table_created,
        "external_table": external_table,
        "self_mdb_path": self_mdb_path,
        "self_mdb_n_lines": self_mdb_n_lines,
        "self_n_bands": self_n_bands,
    }
    summary = comparison_summary(
        args, wav, obs, err, self_mu, external_mu, self_ckd, external_ckd, paths
    )
    if nu_grid is not None:
        summary["self_nu_grid_points"] = int(nu_grid.size)
        summary["self_nu_grid_resolution"] = float(grid_resolution)
    if patch_manifest is not None:
        summary["self_patch_manifest"] = patch_manifest["_manifest_path"]
        summary["self_patch_table_count"] = int(len(patch_manifest["tables"]))
        summary["self_patch_used_count"] = int(
            sum(report["n_observed"] > 0 for report in self_patch_reports)
        )
        summary["self_patch_reports"] = self_patch_reports
    summary["jax_platforms_env"] = os.environ.get("JAX_PLATFORMS", "")
    save_outputs(args, wav, obs, err, channel, self_mu, external_mu, summary)

    product_r3000_mu = None
    if args.product_r3000_self_patch_manifest:
        product_manifest = load_self_patch_manifest(args.product_r3000_self_patch_manifest)
        product_molmass = float(product_manifest.get("_self_molmass") or external_ckd.molmass)
        product_r3000_mu, product_reports = compute_transmission_from_self_patches(
            args,
            art,
            product_manifest,
            product_molmass,
            wav,
        )
        save_product_outputs(args, wav, obs, err, external_mu, self_mu, product_r3000_mu)
        product_summary = {
            "r3000_self_patch_manifest": product_manifest["_manifest_path"],
            "r3000_self_patch_table_count": int(len(product_manifest["tables"])),
            "r3000_self_patch_used_count": int(
                sum(report["n_observed"] > 0 for report in product_reports)
            ),
            "r3000_self_n_bands": int(
                sum(table["n_bands"] for table in product_manifest["tables"])
            ),
        }
        with open(Path(args.output_dir) / "comparison_product_summary.json", "w") as handle:
            json.dump(product_summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print("Self CKD vs ExoMolOP CKD comparison complete.")
    print(f"  molecule: {args.molecule}")
    print(f"  observed points: {wav.size}")
    print(f"  wavelength range: {np.min(wav):.3f}-{np.max(wav):.3f} nm")
    print(f"  self CKD table: {self_table} ({'created' if self_table_created else 'reused'})")
    print(f"  ExoMolOP table: {external_table}")
    if patch_manifest is not None:
        print(
            "  self patch tables: "
            f"{summary['self_patch_used_count']}/{summary['self_patch_table_count']} used"
        )
    print(f"  self bands: {self_n_bands}")
    print(f"  ExoMolOP bands: {np.asarray(external_ckd.nu_bands).size}")
    if product_r3000_mu is not None:
        print("  product comparison: ExoMolOP R1000, self R1000, self R3000")
    print(f"  RMS delta Rp/Rs: {summary['delta_rprs_rms_ppm']:.3f} ppm")
    print(f"  Max |delta Rp/Rs|: {summary['delta_rprs_max_abs_ppm']:.3f} ppm")
    print(f"  summary: {Path(args.output_dir) / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
