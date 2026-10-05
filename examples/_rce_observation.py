"""Observation helper for the ExoSpecDB WASP-18b emission candidate.

Load and validate files on the host once, then close over ``observation.W`` in
a JAX forward model. Supply surface Fnu evaluated at ``input_wavelength_um``;
``pixel_flux_ratio(W, planet_fnu, star_fnu, radius_ratio**2)`` returns Fp/Fstar.
The response is a sum of calibrated pixel flux densities, without additional
throughput or bin-width weights. An LSF/full response is unavailable in this
product; testing this sampling approximation belongs to the forward model.

The supplied errors already include residual scaling and red-noise beta.
They remain conditional on fixed timing/geometry and omit spectral covariance.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from astropy import units as u
from astropy.table import Table
import jax.numpy as jnp
import numpy as np


_SCHEMA = "exospecdb.wasp18b-soss-emission-candidate/0.1.0"
_COLUMNS = (
    "wavelength_um", "wavelength_lower_um", "wavelength_upper_um", "n_pixels",
    "fp_fs", "uncertainty", "accepted", "mask_reason", "red_noise_beta",
    "n_used", "fp_fs_ppm", "uncertainty_ppm",
)


@dataclass(frozen=True)
class EmissionObservation:
    """Validated host data; ``table`` retains all original rows and reasons.

    ``W_all`` includes rejected/empty bins, while ``W`` contains accepted bins.
    File hashes record the three verified outputs and the summary manifest.
    They establish product consistency, not independent provenance authenticity.
    """

    table: Table
    input_wavelength_um: np.ndarray
    static_pixel_mask: np.ndarray
    W_all: np.ndarray
    recipe: dict
    summary: dict
    sha256: dict

    @property
    def row_indices(self):
        """Original zero-based indices of the accepted rows."""
        return np.flatnonzero(self.table["accepted"])

    @property
    def rejected_row_indices(self):
        return np.flatnonzero(~self.table["accepted"])

    @property
    def W(self):
        return self.W_all[self.row_indices]

    @property
    def fp_fs(self):
        return np.asarray(self.table["fp_fs"])[self.row_indices]

    @property
    def uncertainty(self):
        return np.asarray(self.table["uncertainty"])[self.row_indices]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_emission_candidate(spectrum_path):
    """Read an ECSV and its sibling recipe, diagnostics and summary manifest.

    Inconsistent metadata or an invalid accepted row raises ``ValueError``;
    rejected rows are retained unchanged, including their NaNs and mask reasons.
    This helper implements only the explicit 0.1.0 candidate schema.
    """
    spectrum_path = Path(spectrum_path)
    if spectrum_path.name != "emission_candidate.ecsv":
        raise ValueError("expected emission_candidate.ecsv and its companion files")
    directory = spectrum_path.parent
    summary = json.loads((directory / "summary.json").read_text())
    hashes = {"summary.json": _sha256(directory / "summary.json")}
    for name in (spectrum_path.name, "recipe.json", "diagnostics.npz"):
        record = summary.get("outputs", {}).get(name, {})
        path = directory / name
        hashes[name] = _sha256(path)
        if hashes[name] != record.get("sha256"):
            raise ValueError(f"SHA256 mismatch for {name}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"size mismatch for {name}")
    recipe = json.loads((directory / "recipe.json").read_text())
    if any(item.get("schema_version") != _SCHEMA for item in (recipe, summary)):
        raise ValueError("unsupported emission candidate schema_version")
    if not u.Unit(summary["input_flux_unit"]).is_equivalent(u.Jy):
        raise ValueError("input_flux_unit must describe Fnu")
    table = Table.read(spectrum_path, format="ascii.ecsv")
    if set(_COLUMNS) - set(table.colnames):
        raise ValueError("missing required emission candidate columns")
    for key, expected in (("status", "candidate"), ("spectral_order", 1)):
        if table.meta.get(key) != expected or summary.get(key) != expected:
            raise ValueError(f"inconsistent {key}")
    if (table.meta.get("event_type") != "secondary_eclipse"
            or recipe.get("spectral_order") != 1
            or table.meta.get("target") != recipe.get("target")
            or summary.get("target") != recipe.get("target")):
        raise ValueError("inconsistent target, event_type or spectral_order")
    for name in _COLUMNS:
        column = table[name]
        expected_unit = u.um if name.endswith("_um") else u.dimensionless_unscaled
        if column.unit is not None and column.unit != expected_unit:
            raise ValueError(f"unexpected unit for {name}: {column.unit}")
        if name != "mask_reason" and np.any(np.ma.getmaskarray(column)):
            raise ValueError(f"masked entries in {name}; use accepted and mask_reason")
    if table["accepted"].dtype.kind != "b":
        raise ValueError("accepted must be boolean")
    for name in ("n_pixels", "n_used"):
        if table[name].dtype.kind not in "iu" or np.any(table[name] < 0):
            raise ValueError(f"{name} must contain nonnegative integers")
    # Astropy reads quoted empty strings as masked strings in this ECSV schema.
    reasons = np.ma.asarray(table["mask_reason"]).filled("").astype(str)
    accepted = np.asarray(table["accepted"])
    if not np.any(accepted) or np.any(accepted != (reasons == "")):
        raise ValueError("accepted and mask_reason disagree, or no bins accepted")

    with np.load(directory / "diagnostics.npz", allow_pickle=False) as diagnostics:
        wave = diagnostics["input_wavelength_um"].copy()
        static_mask = diagnostics["static_pixel_mask"].copy()
    if (wave.ndim != 1 or static_mask.shape != wave.shape
            or static_mask.dtype.kind != "b"
            or not np.all(np.isfinite(wave) & (wave > 0))):
        raise ValueError("invalid input_wavelength_um or static_pixel_mask")
    left = np.asarray(table["wavelength_lower_um"])
    right = np.asarray(table["wavelength_upper_um"])
    lower, upper = recipe["wavelength_range_um"]
    resolution = recipe["resolving_power"]
    if (not np.all(np.isfinite([lower, upper, resolution]))
            or not 0 < lower < upper or resolution <= 0
            or not np.all(np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > left))
            or not np.allclose(left[1:], right[:-1], rtol=1e-12, atol=0)
            or not np.isclose(left[0], lower, rtol=1e-12, atol=0)
            or not np.isclose(right[-1], upper, rtol=1e-12, atol=0)
            or not np.allclose(right, np.minimum(upper, left * (1 + 1 / resolution)),
                               rtol=1e-12, atol=0)):
        raise ValueError("bin edges disagree with recipe")
    if np.any(static_mask & ((wave < lower) | (wave > upper))):
        raise ValueError("static_pixel_mask selects wavelengths outside recipe")
    # Match ExoSpecDB: [left, right), with the final recipe endpoint included.
    weights = (static_mask[None, :] & (wave[None, :] >= left[:, None])
               & ((wave[None, :] < right[:, None])
                  | ((right[:, None] == upper) & (wave[None, :] <= right[:, None]))))
    counts = weights.sum(axis=1)
    centers = np.array([wave[mask].mean() if mask.any() else (lo + hi) / 2
                        for mask, lo, hi in zip(weights, left, right)])
    if (not np.array_equal(counts, table["n_pixels"])
            or not np.allclose(centers, table["wavelength_um"], rtol=1e-12, atol=0)):
        raise ValueError("pixel counts or mean wavelengths disagree with diagnostics")
    for key, expected in (("n_bins", len(table)), ("n_accepted", accepted.sum()),
                          ("n_static_pixels", static_mask.sum())):
        if summary.get(key) != expected:
            raise ValueError(f"inconsistent {key}")
    if np.any(counts[accepted] < max(1, recipe["minimum_bin_pixels"])):
        raise ValueError("accepted bin has insufficient selected pixels")
    for name in ("fp_fs", "uncertainty", "red_noise_beta"):
        values = np.asarray(table[name])[accepted]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite {name} in accepted rows")
    if np.any(table["uncertainty"][accepted] <= 0):
        raise ValueError("uncertainty must be positive in accepted rows")
    if np.any(table["red_noise_beta"][accepted] < 1):
        raise ValueError("red_noise_beta must be at least one in accepted rows")
    for name in ("fp_fs", "uncertainty"):
        if not np.allclose(table[name + "_ppm"], table[name] * 1e6,
                           rtol=1e-12, atol=0, equal_nan=True):
            raise ValueError(f"{name}_ppm disagrees with dimensionless {name}")
    return EmissionObservation(table, wave, static_mask, weights, recipe, summary, hashes)


def pixel_flux_ratio(weights, planet_fnu, star_fnu, area_ratio):
    """Return dimensionless Fp/Fstar as area ratio times pixel Fnu sum ratio.

    Inputs are one surface flux density per input pixel, in matching units.
    Evaluate both spectra at the diagnostic wavelengths before calling. Fluxes
    must be finite and stellar bin sums positive; physical validity is the
    forward model's responsibility. ``weights`` must contain nonempty bins.
    The same ratio applies to two spectra per cm^-1 because their common
    conversion to per Hz cancels. Convert F_lambda explicitly before binning.
    """
    weights = jnp.asarray(weights)
    planet_fnu, star_fnu = jnp.asarray(planet_fnu), jnp.asarray(star_fnu)
    if (weights.ndim != 2 or planet_fnu.shape != (weights.shape[1],)
            or star_fnu.shape != planet_fnu.shape or jnp.ndim(area_ratio) != 0):
        raise ValueError("expected (bin, pixel) weights, pixel spectra and scalar area_ratio")
    return area_ratio * (weights @ planet_fnu) / (weights @ star_fnu)


def flambda_to_fnu(wavelength_um, flux_lambda, *, per_wavelength="um"):
    """Convert F_lambda per specified length unit to Fnu per Hz, using JAX.

    Wavelengths are always in micrometers. Energy, area and time units are
    unchanged. For example, erg/s/cm^2/um becomes erg/s/cm^2/Hz. The keyword
    (``um``, ``nm``, ``angstrom``, ``cm`` or ``m``) is static under JIT.
    """
    length_cm = {"um": 1e-4, "nm": 1e-7, "angstrom": 1e-8, "cm": 1.0, "m": 100.0}
    if per_wavelength not in length_cm:
        raise ValueError("unsupported F_lambda wavelength unit")
    wavelength_cm = jnp.asarray(wavelength_um) * 1e-4
    return jnp.asarray(flux_lambda) * wavelength_cm**2 / (2.99792458e10 * length_cm[per_wavelength])
