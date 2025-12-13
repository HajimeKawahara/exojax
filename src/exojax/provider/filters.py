import numpy as np
from exojax.utils.grids import wav2nu
from exojax.utils.constants import ccgs
from exojax.provider.url import url_svo_filter


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
    if unit == "Angstrom": # for astropy >= 7.1.0
        unit = "AA"
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
