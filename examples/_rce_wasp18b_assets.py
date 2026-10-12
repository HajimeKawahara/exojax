"""Load explicitly identified, SHA256-pinned assets for the WASP-18b example.

The manifest uses ``schema_version: exojax.rce-wasp18b/1`` and a ``files``
mapping from asset names to ``path``, ``sha256``, and ``source``. Paths are
relative to the manifest unless absolute. Every file, including Diffgrid NPZ
metadata sidecars, must be pinned. Loading never downloads or builds a table.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SCHEMA_VERSION = "exojax.rce-wasp18b/1"
SURFACE_FLUX_UNIT = "erg s-1 cm-2 (cm-1)-1"
PHOENIX_FLUX_UNIT = "erg s-1 cm-2 cm-1"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetManifest:
    """Host configuration and verified local files; ``config`` is the full JSON."""

    path: Path
    config: dict
    paths: dict[str, Path]
    sha256: dict[str, str]


def load_manifest(path):
    """Verify all files before handing any asset to a scientific reader."""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION!r}")
    files = config.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest files must be a nonempty mapping")
    paths, hashes = {}, {"__manifest__": _sha256(path)}
    for key, item in files.items():
        if key == "__manifest__":
            raise ValueError("__manifest__ is a reserved asset name")
        if not isinstance(item, dict):
            raise ValueError(f"asset {key!r} must define path, sha256, and source")
        if not isinstance(item.get("source"), str) or not item["source"].strip():
            raise ValueError(f"asset {key!r} requires an explicit source")
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError(f"asset {key!r} requires a local path")
        expected = item.get("sha256", "")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"asset {key!r} requires a SHA256 hex digest")
        asset = Path(item["path"]).expanduser()
        asset = (path.parent / asset).resolve()
        if not asset.is_file():
            raise FileNotFoundError(f"Missing asset {key!r}: {asset}")
        actual = _sha256(asset)
        if actual != expected.lower():
            raise ValueError(f"SHA256 mismatch for asset {key!r}: {asset}")
        paths[key], hashes[key] = asset, actual
    return AssetManifest(path, config, paths, hashes)


def _asset(manifest, key):
    if key not in manifest.paths:
        raise ValueError(f"Asset {key!r} is not declared in manifest files")
    return manifest.paths[key]


def load_opacity(manifest, specification):
    """Restore one CKD NPZ, ExoMolOP HDF5, or Diffgrid NPZ table.

    ``specification`` contains ``asset`` and ``format`` (``ckd-npz``,
    ``exomolop``, or ``diffgrid``). Diffgrid also requires ``metadata_asset``
    naming its sibling ``*_metadata.json`` file. Existing library validation
    and strict Diffgrid version/dtype checks remain active.
    """
    from exojax.opacity import OpaCKD, OpaDiffgrid

    path = _asset(manifest, specification["asset"])
    format_name = specification["format"]
    if format_name == "ckd-npz":
        if path.suffix != ".npz":
            raise ValueError("ckd-npz requires a .npz file")
        return OpaCKD.load_only().load_tables(str(path), io_format="npz")
    if format_name == "exomolop":
        if path.suffix != ".h5":
            raise ValueError("exomolop requires a pinned .h5 file")
        import h5py
        import jax.numpy as jnp

        opa = OpaCKD.from_external("exomolop", str(path))
        with h5py.File(path, "r") as table:
            for field, expected in (("kcoeff", "cm^2/molecule"), ("p", "bar")):
                unit = table[field].attrs.get("units")
                if isinstance(unit, bytes):
                    unit = unit.decode("utf-8")
                if unit != expected:
                    raise ValueError(f"ExoMolOP {field} units must be {expected!r}")
            if "bin_edges" not in table:
                raise ValueError("P7 ExoMolOP requires native bin_edges")
            edges = np.asarray(table["bin_edges"], dtype=float)
        centers = np.asarray(opa.nu_bands)
        if (edges.ndim != 1 or edges.size != centers.size + 1
                or not np.all(np.isfinite(edges)) or np.any(edges <= 0)
                or np.any(np.diff(edges) <= 0)
                or np.any(centers < edges[:-1]) or np.any(centers > edges[1:])):
            raise ValueError("ExoMolOP bin_edges must increase and bound every band center")
        # The generic provider infers edges from centers; energy integrals must
        # instead retain the native intervals used to construct these k tables.
        opa.band_edges = jnp.asarray(np.column_stack((edges[:-1], edges[1:])))
        opa.band_width = float(np.median(np.diff(edges)))
        opa.ckd_info = replace(opa.ckd_info, band_edges=opa.band_edges)
        return opa
    if format_name == "diffgrid":
        if path.suffix != ".npz":
            raise ValueError("diffgrid requires a .npz file and pinned metadata_asset")
        sidecar = _asset(manifest, specification.get("metadata_asset"))
        expected = path.with_name(path.stem + "_metadata.json")
        if sidecar != expected:
            raise ValueError(f"Diffgrid metadata_asset must identify {expected}")
        return OpaDiffgrid.from_saved_opa(str(path))
    raise ValueError(f"Unsupported opacity format {format_name!r}")


def load_ckd_opacities(manifest):
    """Return species-to-CKD objects from ``opacity.ckd`` in the manifest."""
    result = {}
    for species, spec in manifest.config["opacity"]["ckd"].items():
        spec = dict(spec)
        if spec["format"] not in ("npz", "exomolop"):
            raise ValueError("opacity.ckd format must be npz or exomolop")
        if spec["format"] == "npz":
            spec["format"] = "ckd-npz"
        result[species] = load_opacity(manifest, spec)
    return result


def load_lbl_opacities(manifest):
    """Return species-to-Diffgrid objects from ``opacity.lbl`` in the manifest."""
    result = {}
    for species, spec in manifest.config["opacity"]["lbl"].items():
        if spec["format"] != "diffgrid":
            raise ValueError("opacity.lbl format must be diffgrid")
        result[species] = load_opacity(manifest, spec)
    return result


@dataclass(frozen=True)
class StellarSpectrum:
    """Stellar surface flux per wavenumber, in erg/s/cm2/(cm-1)."""

    nu_cm1: np.ndarray
    surface_flux_per_cm1: np.ndarray
    metadata: dict

    def validate_coverage(self, nu_cm1):
        """Reject requests outside the tabulated range before interpolation."""
        nu = np.asarray(nu_cm1, dtype=float)
        if not nu.size or not np.all(np.isfinite(nu)) or np.any(nu <= 0):
            raise ValueError("Requested stellar wavenumbers must be finite and positive")
        if np.min(nu) < self.nu_cm1[0] or np.max(nu) > self.nu_cm1[-1]:
            raise ValueError("Stellar spectrum does not cover the requested wavenumbers")

    def at(self, nu_cm1):
        """Interpolate on the host with explicit coverage checks."""
        self.validate_coverage(nu_cm1)
        return np.interp(nu_cm1, self.nu_cm1, self.surface_flux_per_cm1)


def load_stellar_spectrum(manifest, required_nu=None):
    """Load a surface spectrum without inferring units or a stellar model.

    ``stellar`` requires ``format``, ``version``, and ``flux_unit``. NPZ files
    contain ``nu_cm1`` and ``surface_flux_per_cm1`` and use SURFACE_FLUX_UNIT.
    PHOENIX FITS uses ``wavelength_asset``, ``flux_asset``, an explicit
    ``wavelength_unit``, and PHOENIX_FLUX_UNIT (flux per cm of wavelength).
    Its conversion is F_wavenumber = F_wavelength * wavelength_cm**2.
    """
    spec = manifest.config["stellar"]
    if not isinstance(spec.get("version"), str) or not spec["version"].strip():
        raise ValueError("stellar.version must identify the source version")
    metadata = dict(spec.get("metadata", {}))
    metadata["version"] = spec["version"]
    if spec["format"] == "npz":
        if spec.get("flux_unit") != SURFACE_FLUX_UNIT:
            raise ValueError(f"stellar flux_unit must be {SURFACE_FLUX_UNIT!r}")
        with np.load(_asset(manifest, spec["asset"]), allow_pickle=False) as data:
            nu = np.asarray(data["nu_cm1"], dtype=float)
            flux = np.asarray(data["surface_flux_per_cm1"], dtype=float)
    elif spec["format"] == "phoenix-fits":
        from astropy import units as u
        from astropy.io import fits

        if spec.get("flux_unit") != PHOENIX_FLUX_UNIT:
            raise ValueError(f"stellar flux_unit must be {PHOENIX_FLUX_UNIT!r}")
        if "wavelength_unit" not in spec:
            raise ValueError("PHOENIX wavelength_unit is required")
        wave_unit = u.Unit(spec["wavelength_unit"])
        if not wave_unit.is_equivalent(u.cm):
            raise ValueError("PHOENIX wavelength_unit must be a length unit")
        wavelength, wave_header = fits.getdata(
            _asset(manifest, spec["wavelength_asset"]), header=True
        )
        flux, header = fits.getdata(_asset(manifest, spec["flux_asset"]), header=True)
        for actual, expected, name, keyword in (
            (wave_header.get("BUNIT"), wave_unit, "wavelength", "BUNIT"),
            (wave_header.get("UNIT"), wave_unit, "wavelength", "UNIT"),
            (header.get("BUNIT"), u.erg / u.s / u.cm**3, "flux", "BUNIT"),
        ):
            if actual is not None and u.Unit(actual) != expected:
                raise ValueError(f"PHOENIX {name} {keyword} disagrees with manifest unit")
        wave_cm = (np.asarray(wavelength, dtype=float) * wave_unit).to_value(u.cm)
        if not np.all(np.isfinite(wave_cm)) or np.any(wave_cm <= 0):
            raise ValueError("PHOENIX wavelengths must be finite and positive")
        nu = 1.0 / wave_cm
        flux = np.asarray(flux, dtype=float) * wave_cm**2
        metadata["header"] = dict(header)
        for source, target in (("PHXTEFF", "teff_K"), ("PHXLOGG", "logg"),
                               ("PHXVER", "phoenix_version")):
            if source in header:
                metadata[target] = header[source]
    else:
        raise ValueError(f"Unsupported stellar format {spec['format']!r}")
    if nu.ndim != 1 or nu.size < 2 or flux.shape != nu.shape:
        raise ValueError("Stellar coordinates and flux must be matching 1D arrays")
    if not np.all(np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError("Stellar wavenumbers must be finite and positive")
    if not np.all(np.isfinite(flux)) or np.any(flux < 0) or not np.any(flux > 0):
        raise ValueError("Stellar surface flux must be finite, nonnegative, and nonzero")
    order = np.argsort(nu)
    nu, flux = nu[order], flux[order]
    if np.any(np.diff(nu) <= 0):
        raise ValueError("Stellar wavenumbers must be unique")
    metadata["flux_unit"] = SURFACE_FLUX_UNIT
    result = StellarSpectrum(nu, flux, metadata)
    if required_nu is not None:
        result.validate_coverage(required_nu)
    return result
