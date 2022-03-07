"""unit conversion for spectrum."""


def nu2wav(nus, outputunit='AA'):
    """wavenumber to wavelength (AA)

    Args:
       nus: wavenumber (cm-1)
       outputunit: unit of wavelength

    Returns:
       wavelength (AA)
    """
    if outputunit == 'nm':
        return 1.e7/nus[::-1]
    elif outputunit == 'AA':
        return 1.e8/nus[::-1]
    else:
        print('Warning: assumes AA as the unit of output.')
        return 1.e8/nus[::-1]


def wav2nu(wav, inputunit):
    """wavelength to wavenumber.

    Args:
       wav: wavelength
       inputunit: unit of wavelength

    Returns:
       wavenumber (cm-1)
    """

    if inputunit == 'nm':
        return 1.e7/wav[::-1]
    elif inputunit == 'AA':
        return 1.e8/wav[::-1]
    else:
        print('Warning: assumes AA as the unit of input.')
        return 1.e8/wav[::-1]
