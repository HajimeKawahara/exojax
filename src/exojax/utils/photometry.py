import numpy as np
import jax.numpy as jnp
from exojax.spec.unitconvert import wav2nu
from exojax.utils.constants import ccgs
from exojax.utils.url import url_svo_filter
from jax.scipy.integrate import trapezoid


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

    """
    from exojax.spec.planck import piB
    from exojax.utils.constants import RJ
    from exojax.utils.constants import pc

    absflux = (
        piB(temperature, nu_ref) * (radius) ** 2 / (distance) ** 2 * (RJ / pc) ** 2
    )
    return apparent_magnitude(absflux, nu_ref, transmission_ref, f0_nu_cgs)


def download_filter_from_svo(filter_id):
    """download filter transmission data from SVO

    Args:
        filter_id (str): filter id name such as "2MASS/2MASS.Ks" see http://svo2.cab.inta-csic.es/theory/fps/

    Returns:
        array: wavenumber (cm-1)
        array: filter transmission (dimensionless, 0 to 1)
    """

    #
    from astroquery.svo_fps import SvoFps

    print("filter_id = ", filter_id)
    print("You can check the available filters at", url_svo_filter())
    data = SvoFps.get_transmission_data(filter_id)
    unit = str(data["Wavelength"].unit)
    wl_ref = np.array(data["Wavelength"])
    nu_ref, transmission_ref = wav2nu(
        wl_ref, unit=unit, values=np.array(data["Transmission"])
    )
    return nu_ref, transmission_ref


def download_zero_magnitude_flux_from_svo(filter_id, unit="cm-1"):
    """download zero magnitude flux from SVO

    Args:
        filter_id (str): filter id name such as "2MASS/2MASS.Ks" see http://svo2.cab.inta-csic.es/theory/fps/
        unit (str, optional): unit of the output. Defaults to "cm-1".

    Returns:
        float: wavenumber or wavelength (cm-1, um, or AA)
        float: zero magnitude flux (erg/s/cm^2/cm-1, erg/s/cm^2/um, or erg/s/cm^2/AA)
    """
    from astroquery.svo_fps import SvoFps

    facility = filter_id.split("/")[0]
    filters = SvoFps.get_filter_list(facility=facility)
    filter_data = filters[filters["filterID"] == filter_id]
    
    if filter_data["ZeroPointUnit"] != "Jy":
        raise ValueError("ZeroPointUnit should be Jy")

    lambda0_um = filter_data["WavelengthPhot"].value[0] * 1.0e-4
    f0_orig = filter_data["ZeroPoint"].value[0]
    f0_nu_cgs = f0_orig * 1.0e-23 * ccgs  # erg/s/cm^2/cm-1
    
    conversion = {
        "cm-1": (1.0e4 / lambda0_um, f0_nu_cgs),
        "um":   (lambda0_um, f0_nu_cgs * 1.0e4 / lambda0_um**2),
        "AA":   (1.0e4 * lambda0_um, f0_nu_cgs / lambda0_um**2)
    }
    try:
        return conversion[unit]
    except KeyError:
        raise ValueError("unit should be cm-1, um, or AA")


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
