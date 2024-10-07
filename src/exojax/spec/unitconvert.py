"""unit conversion for spectrum."""

import warnings
from exojax.utils.checkarray import is_sorted


def nu2wav(nus, wavelength_order="descending", unit="AA"):
    """wavenumber to wavelength (AA)

    Args:
        nus: wavenumber (cm-1)
        wavlength order: wavelength order: "ascending" or "descending", default to "descending"
        unit: the unit of the output wavelength, "AA", "nm", or "um"

    Returns:
        wavelength (unit)
    """
    conversion_factors = {"nm": 1.0e7, "AA": 1.0e8, "um": 1.0e4}
    wavenumber_order = is_sorted(nus)

    if wavenumber_order == "descending" or wavenumber_order == "unordered":
        raise ValueError("wavenumber should be in ascending order in ExoJAX.")

    try:
        if wavelength_order == "ascending":
            _both_ascending_warning()
            return conversion_factors[unit] / nus[::-1]
        elif wavelength_order == "descending" or wavenumber_order == "single":
            return conversion_factors[unit] / nus
        else:
            raise ValueError("order should be ascending or descending")
    except KeyError:
        raise ValueError("unavailable unit")


def wav2nu(wav, unit):
    """wavelength to wavenumber.

    Args:
        wav: wavelength array in ascending/descending order
        unit: unit of wavelength

    Returns:
        wavenumber (cm-1) in ascending order
    """

    conversion_factors = {"nm": 1.0e7, "AA": 1.0e8, "um": 1.0e4}

    order = is_sorted(wav)
    try:
        if order == "ascending":
            _both_ascending_warning()
            return conversion_factors[unit] / wav[::-1]
        elif order == "descending" or order == "single":
            return conversion_factors[unit] / wav
        else:
            raise ValueError("wavelength array should be ascending or descending")
    except KeyError:
        raise ValueError("unavailable unit")


def _both_ascending_warning():
    warnings.warn(
        "Both input wavelength and output wavenumber are in ascending order.",
        UserWarning,
    )
