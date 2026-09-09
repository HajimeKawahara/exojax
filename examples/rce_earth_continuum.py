"""Water-vapor continuum for the Earth RCE tutorial.

Reference data are downloaded separately from AER's MT_CKD_H2O 4.3 release.
Their scientific/research-use license is saved beside the data; neither the
data nor AER's implementation is distributed with this example. See
https://hitran.org/mtckd/ and Mlawer et al. (2023),
https://doi.org/10.1016/j.jqsrt.2023.108645.

The continuum accompanies local water lines truncated at 25 cm-1 with the
Lorentz pedestal removed. The foreign continuum describes air; its use for
a nitrogen background is an approximation.
"""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import netcdf_file


_BASE_URL = "https://raw.githubusercontent.com/AER-RC/MT_CKD_H2O/4.3/"
_FILES = (
    (
        "LICENSE.md",
        "0034503387080d78f3034e8d409ca9b2428e90aa6aff092b9b925defa216a162",
    ),
    (
        "data/absco-ref_wv-mt-ckd.nc",
        "69944eb8b045c268e2daeb2cddf536b99f9e9efe7e91c425067b46a5b287ddb3",
    ),
)


def download_data(cache_path):
    """Explicitly cache the 84 KB reference table and its research-use license.

    Existing files and downloads must match the release's SHA-256 checksums.
    This function alone accesses the network; loading and using the continuum
    are offline. Returns the path to the reference netCDF file.
    """
    cache = Path(cache_path)
    cache.mkdir(parents=True, exist_ok=True)
    for relative_path, expected_hash in _FILES:
        destination = cache / Path(relative_path).name
        if destination.exists():
            content = destination.read_bytes()
        else:
            with urlopen(_BASE_URL + relative_path, timeout=30) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"MT_CKD 4.3 checksum mismatch: {destination}")
        if not destination.exists():
            destination.write_bytes(content)
    return cache / "absco-ref_wv-mt-ckd.nc"


def load_continuum(path):
    """Load a table and return ``sigma(T, P_bar, x_water, nu_cm)``.

    The first three inputs are layer arrays of shape (N,); wavenumbers have
    shape (M,). The result is the combined self and foreign cross section in
    cm2 per water molecule, shape (N, M). Multiply by the water column in
    molecules/cm2 to obtain optical depth. Temperatures must be positive,
    pressures nonnegative, and water volume fractions between zero and one.

    At each native wavenumber, the partner density relative to the reference
    state is (P/P_ref)(T_ref/T), the self coefficient additionally scales as
    (T_ref/T)**self_texp, and the radiation factor is
    nu*tanh(h*c*nu/(2*k*T)). These follow the density and stimulated-emission
    definitions of MT_CKD. The default foreign coefficients are used; the
    alternative aerosol-dependent ``for_closure_absco_ref`` is not used.

    Scaling precedes linear spectral interpolation, a simplification of
    AER's cubic interpolation. The result vanishes outside 0--20000 cm-1.
    All reference coefficients become JAX arrays once, during loading.
    """
    with netcdf_file(path, "r", mmap=False) as dataset:
        values = {
            name: np.array(dataset.variables[name].data, dtype=float, copy=True)
            for name in (
                "wavenumbers",
                "self_absco_ref",
                "for_absco_ref",
                "self_texp",
                "ref_temp",
                "ref_press",
            )
        }
    grid = jnp.asarray(values["wavenumbers"])
    self_reference = jnp.asarray(values["self_absco_ref"])
    foreign_reference = jnp.asarray(values["for_absco_ref"])
    self_exponent = jnp.asarray(values["self_texp"])
    reference_temperature = float(values["ref_temp"])
    reference_pressure_bar = float(values["ref_press"]) / 1000.0

    def sigma(temperature, pressure_bar, water_vmr, wavenumber):
        temperature = jnp.asarray(temperature)[:, None]
        pressure = jnp.asarray(pressure_bar)[:, None]
        water = jnp.asarray(water_vmr)[:, None]
        temperature_ratio = reference_temperature / temperature
        density_ratio = pressure / reference_pressure_bar * temperature_ratio
        radiation = grid * jnp.tanh(1.4387752 * grid / (2.0 * temperature))
        coefficients = density_ratio * radiation * (
            self_reference * temperature_ratio**self_exponent * water
            + foreign_reference * (1.0 - water)
        )
        interpolated = jax.vmap(
            lambda row: jnp.interp(wavenumber, grid, row, left=0.0, right=0.0)
        )(coefficients)
        return jnp.where(jnp.asarray(wavenumber)[None, :] >= 0.0, interpolated, 0.0)

    sigma.metadata = {
        "model": "MT_CKD H2O 4.3",
        "source_url": _BASE_URL + _FILES[1][0],
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    return sigma
