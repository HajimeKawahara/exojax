def mmr_to_vmr(mmr, molecular_mass, mean_molecular_weight):
    """convert mass mixing ratio (mmr) to volume mixing ratio (vmr)

    Args:
        mmr (float or array): mass mixing ratio(s)
        molecular_mass (float or array): molecular mass(es)
        mean_molecular_weight (float): mean molecular weight

    Returns:
        float: volume mixing ratio(s)
    """
    return mmr * mean_molecular_weight / molecular_mass


def vmr_to_mmr(vmr, molecular_mass, mean_molecular_weight):
    """convert volume mixing ratio (vmr) to mass mixing ratio (mmr)

    Args:
        vmr (float or array): volume mixing ratio(s)
        molecular_mass (float or array): molecular mass(es)
        mean_molecular_weight (float): mean molecular weight

    Returns:
        float: mass mixing ratio(s)
    """

    return vmr * molecular_mass / mean_molecular_weight
