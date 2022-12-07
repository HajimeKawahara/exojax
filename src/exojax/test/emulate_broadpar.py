"""emulate broadening parameters for unit test, used only in premodit_test.py

"""
import numpy as np
def mock_broadpar_exomol():
    """default mock proadening parameters of the ExoMol form for unit test   

    Returns:
        normalized broadening half-width at reference
        temperature exponent  
    """
    ngamma_ref=np.array([0.1,0.11,0.15,0.1,0.15,0.13])
    n_Texp=np.array([0.5,0.4,0.47,0.5,0.47,0.4])
    return ngamma_ref, n_Texp


def mock_broadpar_hitemp():
    """default mock proadening parameters of the HITEMP form for unit test   

    Returns:
        normalized air broadening half-width at reference
        air temperature exponent  
    """
    ngamma_air=np.array([0.1,0.11,0.15,0.12,0.14,0.13])
    n_air=np.array([0.5,0.41,0.47,0.42,0.46,0.4])
    return ngamma_air, n_air
