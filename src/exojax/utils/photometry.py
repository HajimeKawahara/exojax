import numpy as np
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from exojax.provider.filters import download_filter_from_svo
from exojax.provider.filters import download_zero_magnitude_flux_from_svo


def apparent_magnitude(
    flux_filter, nu_grid_filter, transmission_filter, f0_nu_cgs, factor=1.0e20
):
    """computes apparent magnitude

    Args:
        flux (array): flux in the unit of erg/s/cm^2/cm-1
        nu_grid_filter (array): wavenumber grid (cm-1)
        transmission_filter (array): transmission filter (dimensionless, 0 to 1)
        f0_nu_cgs (float): zero magnitude flux in the unit of erg/s/cm^2/cm-1
        factor (float): factor to prevent numerical error. Defaults to 1.0e20.

    Returns:
        float: apparent magnitude
    """

    logfactor = jnp.log10(factor)
    integrated_flux = trapezoid(
        (flux_filter * factor) * transmission_filter, nu_grid_filter
    ) / trapezoid(transmission_filter, nu_grid_filter)
    return -2.5 * (jnp.log10(integrated_flux / f0_nu_cgs) - logfactor)


def apparent_magnitude_isothermal_sphere(
    temperature, radius, distance, nu_ref, transmission_ref, f0_nu_cgs
):
    """calc apparent magnitude of an isothermal sphere

    Args:
        temperature (float): temperature (K)
        radius (float): radius (RJ)
        distance (float): distance (pc)
        nu_ref (array): wavenumber (cm-1)
        transmission_ref (array): transmission filter (dimensionless, 0 to 1)
        f0_nu_cgs (float): zero magnitude flux in the unit of erg/s/cm^2/cm-1

    Returns:
        float: apparent magnitude

    """
    from exojax.rt.planck import piB
    from exojax.utils.constants import RJ
    from exojax.utils.constants import pc

    absflux = (
        piB(temperature, nu_ref) * (radius) ** 2 / (distance) ** 2 * (RJ / pc) ** 2
    )
    return apparent_magnitude(absflux, nu_ref, transmission_ref, f0_nu_cgs)


def average_resolution(nu_ref):
    """average resolution of the filter

    Args:
        nu_ref (array): wavenumber (cm-1)

    Returns:
        float: average resolution of the filter
    """
    nu_ref_min = np.min(nu_ref)
    nu_ref_max = np.max(nu_ref)
    dnu_ave = (nu_ref_max - nu_ref_min) / len(nu_ref)
    nuave = (nu_ref_max + nu_ref_min) / 2.0
    return nuave / dnu_ave


if __name__ == "__main__":
    from jax import config

    config.update("jax_enable_x64", True)

    radius = 0.85  # RJ
    distance = 17.72  # pc"
    temperature = 1700.0  # K
    filter_name = "Keck/NIRC2.Ks"
    # temperature = 2100.0 #K
    filter_name = "2MASS/2MASS.J"

    nu_ref, transmission_ref = download_filter_from_svo(filter_name)
    nu0, f0_nu_cgs = download_zero_magnitude_flux_from_svo(filter_name, unit="cm-1")
    mag = apparent_magnitude_isothermal_sphere(
        temperature, radius, distance, nu_ref, transmission_ref, f0_nu_cgs
    )
    print(mag)
