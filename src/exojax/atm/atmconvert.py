"""converts quantities in the atmosphere, between mass mixing ratio, volume mixing ratio, and density"""

from exojax.atm.idealgas import number_density

def mmr_to_density(mmr, molmass, Parr, Tarr, unit="g/L"):
    """converts MMR to density

    Args:
        mmr: mass mixing ratio
        molmass_nh3 (_type_): molecular mass
        Parr (_type_): _description_
        Tarr (_type_): _description_
        unit str: unit of the density (g/L, g/cm3)

    Note:
        mass density (g/L) = fac*MMR

    Returns:
        _type_: _description_
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

