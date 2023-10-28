import jax.numpy as jnp


def cross_section_rayleigh_gas(wavenumber,
                               refractive_index,
                               number_density,
                               king_factor=1.0):
    """ Computes Rayleigh scattering cross-section of gas from real refractive index
    
        Args:
            wavenumber: wavenumber (cm-1)
            refractive_index: real refractive index (hint: one can compute the refractive index using the Lorentz-Lonrez formula (atm.lorentz_lorenz) from polarizability)
            number_density: gas number density of the molecule (cm-3)
            king_factor: King correction factor which accounts for the depolarization effect (default=1.0)

        Returns:
            cross section (cm2) for Rayleigh scattering

        Notes:
            References=

    """
    fac = (refractive_index**2 - 1.0) / (refractive_index**2 + 2.0)
    return 24.0 * jnp.pi**3 * wavenumber**4 / number_density**2 * fac**2 * king_factor
