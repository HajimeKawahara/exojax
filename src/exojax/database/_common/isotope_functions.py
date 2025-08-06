import numpy as np

def _convert_proper_isotope(isotope):
    """covert isotope (int) to proper type for df

    Args:
        isotope (int or other type): isotope

    Returns:
        str: proper isotope type
    """
    if isotope == 0:
        return None
    elif isotope is not None and type(isotope) == int:
        return str(isotope)
    elif isotope is None:
        return isotope
    else:
        raise ValueError("Invalid isotope type")


def _isotope_index_from_isotope_number(isotope, uniqiso):
    """isotope index given HITRAN/HITEMP isotope number

    Args:
        isotope (int): isotope number
        uniqiso (nd int array): unique isotope array

    Returns:
        int: isotope_index for T_gQT and gQT
    """
    isotope_index = np.where(uniqiso == isotope)[0][0]
    return isotope_index
