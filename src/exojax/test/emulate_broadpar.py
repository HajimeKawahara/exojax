"""emulate broadening parameters for unit test

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
