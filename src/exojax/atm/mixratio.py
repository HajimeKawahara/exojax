"""Mixing ratio definition

"""

def vmr2mmr(vmr, molecular_mass, mean_molecular_weight):
    """
    VMR (Volume Mixing Ratio or Mol Mixing ratio) to MMR (Mass Mixing Ratio)

    Args:
        vmr (_type_): Volume Mixing Ratio 
        molecular_mass (_type_): Molecular Mass (i.e. utils.molinfo.molmass_isotope("H2O") for instance)
        mean_molecular_weight (_type_): Mean Molecular Weight

    Returns:
        _type_: Mass Mixing Ratio
    """
    return molecular_mass/mean_molecular_weight*vmr

def  mmr2vmr(mmr, molecular_mass, mean_molecular_weight):
    """
    MMR (Mass Mixing Ratio) to VMR (Volume Mixing Ratio or Mol Mixing ratio) 


    Args:
        mmr (_type_): Mass Mixing Ratio
        molecular_mass (_type_): Molecular Mass (i.e. utils.molinfo.molmass_isotope("H2O") for instance)
        mean_molecular_weight (_type_): Mean Molecular Weight

    Returns:
        _type_: Volume Mixing Ratio or Mol Mixing Ratio
    """
    return mean_molecular_weight/molecular_mass*mmr
