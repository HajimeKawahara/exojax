import jax.numpy as jnp


def xsvector_rayleigh_gas(wavenumber, polarizability, king_factor=1.0):
    """Computes Rayleigh scattering cross-section of gas from polarizability

    Args:
        wavenumber: wavenumber (cm-1)
        polarizability: alpha (cm3)
        king_factor: King correction factor which accounts for the depolarization effect (default=1.0)

    Returns:
        cross section (cm2) for Rayleigh scattering

    """
    fac = jnp.pi**2 * 8.0 * wavenumber**2 * polarizability
    return 2.0 / 3.0 * jnp.pi * fac**2 * king_factor


def xsvector_rayleigh_gas_from_refractive_index(
    wavenumber, refractive_index, number_density, king_factor=1.0
):
    """Computes Rayleigh scattering cross-section of gas from real refractive index

    Args:
        wavenumber: wavenumber (cm-1)
        refractive_index: real refractive index (use atm.lorentz_lorenz to compute this)
        number_density: gas number density (cm-2)
        king_factor: King correction factor which accounts for the depolarization effect (default=1.0)

    Returns:
        cross section (cm2) for Rayleigh scattering

    Notes:
        This function uses the exact form of the gas Rayleigh scattering, which depends on the number density of the gas

    """
    fac = (refractive_index**2 - 1.0) / (
        number_density * (refractive_index**2 + 2.0)
    )
    return 24.0 * jnp.pi * (wavenumber**2 * fac * jnp.pi) ** 2 * king_factor


if __name__ == "__main__":
    from exojax.atm.polarizability import polarizability
    from exojax.atm.polarizability import king_correction_factor
    from exojax.utils.grids import wavenumber_grid

    nus, wav, res = wavenumber_grid(
        3000.0, 3100.0, 128, xsmode="premodit", wavelength_order="descending", unit="nm"
    )
    sigma = xsvector_rayleigh_gas(
        nus, polarizability["CO"], king_correction_factor["CO"]
    )
    print(sigma)
