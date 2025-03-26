"""unit conversion for spectrum."""

import warnings
import numpy as np
from exojax.utils.checkarray import is_sorted


def nu2wav(nus, wavelength_order="descending", unit="AA", values=None):
    """wavenumber to wavelength (AA)

    Args:
        nus: wavenumber (cm-1)
        wavlength order: wavelength order: "ascending" or "descending", default to "descending"
        unit: the unit of the output wavelength, "AA", "nm", or "um"
        values: corresponding values, e.g. f(nus) for nus
    Returns:
        wavelength (unit)
        values (optional): corresponding values in the same order as wavelength, if values is not None.
    """
    conversion_factors = {"nm": 1.0e7, "AA": 1.0e8, "um": 1.0e4}
    wavenumber_order = is_sorted(nus)

    if wavenumber_order == "descending" or wavenumber_order == "unordered":
        raise ValueError("wavenumber should be in ascending order in ExoJAX.")

    try:
        if wavelength_order == "ascending":
            _both_ascending_warning()
            wav = conversion_factors[unit] / nus[::-1]
            val = values[::-1] if values is not None else None
        elif wavelength_order == "descending" or wavenumber_order == "single":
            wav = conversion_factors[unit] / nus
            val = np.copy(values) if values is not None else None
        else:
            raise ValueError("order should be ascending or descending")
        if values is not None:
            return wav, val
        else:
            return wav
    
    except KeyError:
        raise ValueError("unavailable unit")


def wav2nu(wav, unit, values=None):
    """wavelength to wavenumber.

    Args:
        wav: wavelength array in ascending/descending order
        unit: the unit of the output wavelength, "AA", "nm", or "um"
        values: corresponding values, e.g. f(wav) for wav
        
    Returns:
        wavenumber (cm-1) in ascending order
        values (optional): corresponding values in the same order as wavenumber, if values is not None.
    """

    conversion_factors = {"nm": 1.0e7, "AA": 1.0e8, "um": 1.0e4}

    order = is_sorted(wav)
    try:
        if order == "ascending":
            _both_ascending_warning()
            nu = conversion_factors[unit] / wav[::-1]
            val = values[::-1] if values is not None else None
        elif order == "descending" or order == "single":
            nu = conversion_factors[unit] / wav
            val = np.copy(values) if values is not None else None
        else:
            raise ValueError("wavelength array should be ascending or descending")
    
        if values is not None:
            return nu, val
        else:
            return nu
    except KeyError:
        raise ValueError("unavailable unit")



def _both_ascending_warning():
    warnings.warn(
        "Both input wavelength and output wavenumber are in ascending order.",
        UserWarning,
    )
