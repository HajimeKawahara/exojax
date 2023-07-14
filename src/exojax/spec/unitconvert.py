"""unit conversion for spectrum."""

from exojax.utils.checkarray import is_sorted

def nu2wav(nus, wavelength_order, unit='AA'):
    """wavenumber to wavelength (AA)

    Args:
        nus: wavenumber (cm-1)
        wavlength order: wavelength order: "ascending" or "descending"
        unit: unit of wavelength

    Returns:
       wavelength (AA)
    """
    conversion_factors = {
        'nm': 1.e7,
        'AA': 1.e8,
        'um': 1.e4
    }

    if is_sorted(nus) != "ascending":
        raise ValueError("wavenumber should be in ascending order in ExoJAX.")

    try:
        if wavelength_order=="ascending":
            return conversion_factors[unit] / nus[::-1]
        elif wavelength_order=="descending":
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

    conversion_factors = {
        'nm': 1.e7,
        'AA': 1.e8,
        'um': 1.e4
    }

    order = is_sorted(wav)

    try:
        if order=="ascending":
            return conversion_factors[unit] / wav[::-1]
        elif order=="descending":
            return conversion_factors[unit] / wav
        else:
            raise ValueError("wavelength array should be ascending or descending")
    except KeyError:
        raise ValueError("unavailable unit")

