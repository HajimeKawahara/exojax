import numpy as np
import jax.numpy as jnp
from exojax.spec.unitconvert import wav2nu
from exojax.utils.constants import ccgs



def magnitude_isothermal_sphere(temperature, radius, distance, nu_ref, transmission_ref, f0_nu_cgs):
    from exojax.utils.constants import RJ
    from exojax.utils.constants import pc
    from exojax.spec.planck import piB
    absflux = piB(temperature, nu_ref)*(radius*RJ)**2/(distance*pc)**2
    print(absflux)
    f = jnp.trapezoid(absflux*transmission_ref, nu_ref)/jnp.trapezoid(transmission_ref, nu_ref)
    return -2.5*jnp.log10(f/f0_nu_cgs)

def download_filter_from_svo(filter_name):
    """download filter transmission data from SVO

    Args:
        filter_name (str): filter name such as "2MASS/2MASS.Ks" see http://svo2.cab.inta-csic.es/theory/fps/

    Returns:
        array: wavenumber (cm-1)
        array: filter transmission (dimensionless, 0 to 1)
    """

    #
    from astroquery.svo_fps import SvoFps

    data = SvoFps.get_transmission_data(filter_name)
    unit = str(data["Wavelength"].unit)
    wl_ref = np.array(data["Wavelength"])
    nu_ref, transmission_ref = wav2nu(
        wl_ref, unit=unit, values=np.array(data["Transmission"])
    )
    return nu_ref, transmission_ref


def download_zero_magnitude_flux_from_svo(filter_name, unit="cm-1"):
    """download zero magnitude flux from SVO

    Args:
        filter_name (str): filter name such as "2MASS/2MASS.Ks" see http://svo2.cab.inta-csic.es/theory/fps/
        unit (str, optional): unit of the output. Defaults to "cm-1".

    Returns:
        float: wavenumber or wavelength (cm-1, um, or AA)
        float: zero magnitude flux (erg/s/cm^2/cm-1, erg/s/cm^2/um, or erg/s/cm^2/AA)
    """
    from astroquery.svo_fps import SvoFps

    facility = filter_name.split("/")[0]
    filters = SvoFps.get_filter_list(facility=facility)
    filter_data = filters[filters["filterID"] == filter_name]
    lambda0_um = filter_data["WavelengthPhot"] * 1.0e-4
    f0_orig = filter_data["ZeroPoint"]
    if filter_data["ZeroPointUnit"] == "Jy":
        pass
    else:
        raise ValueError("ZeroPointUnit should be Jy")

    f0_nu_cgs = f0_orig.value[0] * 1.0e-23 * ccgs  # erg/s/cm^2/cm-1
    if unit == "cm-1":
        return 1.0e4 / lambda0_um, f0_nu_cgs  # cm-1, erg/s/cm^2/cm-1
    elif unit == "um":
        return lambda0_um, f0_nu_cgs * 1.0e4 / lambda0_um**2  # um, erg/s/cm2/um
    elif unit == "AA":
        return 1.0e4 * lambda0_um, f0_nu_cgs / lambda0_um**2  # AA erg/s/cm2/AA
    else:
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

    radius = 0.85 #RJ
    distance = 17.72 #pc"
    temperature = 1700.0 #K
    filter_name = "Keck/NIRC2.Ks"
    nu_ref, transmission_ref = download_filter_from_svo(filter_name)
    nu0, f0_nu_cgs = download_zero_magnitude_flux_from_svo(filter_name, unit="cm-1")
    mag = magnitude_isothermal_sphere(temperature, radius, distance, nu_ref, transmission_ref, f0_nu_cgs)
    print(mag)