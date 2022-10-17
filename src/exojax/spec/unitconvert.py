"""unit conversion for spectrum."""


def nu2wav(nus, unit='AA'):
    """wavenumber to wavelength (AA)

    Args:
       nus: wavenumber (cm-1)
       unit: unit of wavelength

    Returns:
       wavelength (AA)
    """
    if unit == 'nm':
        return 1.e7/nus[::-1]
    elif unit == 'AA':
        return 1.e8/nus[::-1]
    else:
        raise ValueError("unavailable unit")
        

def wav2nu(wav, unit):
    """wavelength to wavenumber.

    Args:
       wav: wavelength
       unit: unit of wavelength

    Returns:
       wavenumber (cm-1)
    """

    if unit == 'nm':
        return 1.e7/wav[::-1]
    elif unit == 'AA':
        return 1.e8/wav[::-1]
    else:
        raise ValueError("unavailable unit")
        