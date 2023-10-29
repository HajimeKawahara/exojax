import jax.numpy as jnp
from jax import vmap
from exojax.atm.polarizability import n_ref_refractive


def xsvector_rayleigh_gas(wavenumber, refractive_index, king_factor=1.0):
    """Computes Rayleigh scattering cross-section of gas from real refractive index

    Args:
        wavenumber: wavenumber (cm-1)
        refractive_index: real refractive index (hint: one can compute the refractive index using the Lorentz-Lonrez formula (atm.lorentz_lorenz) from polarizability)
        king_factor: King correction factor which accounts for the depolarization effect (default=1.0)

    Returns:
        cross section (cm2) for Rayleigh scattering

    Notes:
        References=

    """
    fac = (refractive_index**2 - 1.0) / (
        n_ref_refractive * (refractive_index**2 + 2.0)
    )
    return 24.0 * jnp.pi**3 * (wavenumber**2 * fac) ** 2 * king_factor


if __name__ == "__main__":
    from exojax.atm.polarizability import polarizability
    from exojax.atm.lorentz_lorenz import refractive_index_Lorentz_Lorenz

    p = polarizability["CO"]
    refractive_index = refractive_index_Lorentz_Lorenz(p, n_ref_refractive)
    print(refractive_index)
    from exojax.utils.grids import wavenumber_grid

    nus, wav, res = wavenumber_grid(
        3000.0, 3100.0, 128, xsmode="premodit", wavelength_order="descending", unit="nm"
    )
    sigma = xsvector_rayleigh_gas(nus, refractive_index)
    print(sigma)
