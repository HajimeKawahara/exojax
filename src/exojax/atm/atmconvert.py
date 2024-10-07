"""converts quantities in the atmosphere, between mass mixing ratio, volume mixing ratio, and density"""

from exojax.atm.idealgas import number_density
from exojax.utils.constants import m_u

def mmr_to_density(mmr, molmass, Parr, Tarr, unit="g/L"):
    """converts MMR to density

    Args:
        mmr (float or array): mass mixing ratio
        molmass (float): molecular mass
        Parr (float or array): pressure array (bar)
        Tarr (float or array): temperature array (K)
        unit (str): unit of the density ("g/L" or "g/cm3")

    Note:
        mass density (g/L) = fac * MMR

    Returns:
        float or array: density in the specified unit
    """
    if unit == "g/L":
        unit_factor = 1.0e3
    elif unit == "g/cm3":
        unit_factor = 1.0
    else:
        raise ValueError("unit is not correct")
    
    return molmass * m_u * number_density(Parr, Tarr) * unit_factor * mmr

def mmr_to_vmr(mmr, molecular_mass, mean_molecular_weight):
    """converts mass mixing ratio (mmr) to volume mixing ratio (vmr)

    Args:
        mmr (float or array): mass mixing ratio(s)
        molecular_mass (float or array): molecular mass(es)
        mean_molecular_weight (float): mean molecular weight

    Returns:
        float: volume mixing ratio(s)
    """
    return mmr * mean_molecular_weight / molecular_mass


def vmr_to_mmr(vmr, molecular_mass, mean_molecular_weight):
    """converts volume mixing ratio (vmr) to mass mixing ratio (mmr)

    Args:
        vmr (float or array): volume mixing ratio(s)
        molecular_mass (float or array): molecular mass(es)
        mean_molecular_weight (float): mean molecular weight

    Returns:
        float: mass mixing ratio(s)
    """

    return vmr * molecular_mass / mean_molecular_weight
