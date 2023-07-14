"""emulate broadening parameters for unit test, used only in premodit_test.py

"""
import numpy as np

def mock_broadpar(db):
    """data base selector for broad par

    Args:
        db (_type_): db name = "exomol", "hitemp"

    Raises:
        ValueError: _description_

    Returns:
        _type_: mdb object
    """
    if db == "exomol":
        ngamma_ref, n_Texp = mock_broadpar_exomol()
    elif db == "hitemp":
        ngamma_ref, n_Texp = mock_broadpar_hitemp()
    else:
        raise ValueError("no exisiting dbname.")
    return ngamma_ref, n_Texp


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
    ngamma_air=np.array([0.1,0.11,0.15,0.12,0.14,0.01])
    n_air=np.array([0.5,0.41,0.47,0.32,0.0,-0.1])
    return ngamma_air, n_air
