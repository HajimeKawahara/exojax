""" unit conversion for spectrum


"""

def nu2wav(nus):
    """wavenumber to wavelength (AA)
    
    Args:
       nus: wavenumber (cm-1)

    Returns:
       wavelength (AA)

    """

    return 1.e8/nus[::-1]

