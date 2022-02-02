"""cloud opacity."""


def powerlaw_clouds(nus, kappac0=0.01, nuc0=28571., alphac=1.):
    """power-law cloud model.

    Args:
       kappac0: opacity (cm2/g) at nuc0
       nuc0: wavenumber for kappac0
       alphac: power

    Returns:
       cross section (cm2)

    Note:
       alphac = - gamma of the definition in petitRadtrans. Also, default nuc0 corresponds to lambda0=0.35 um.
    """
    return kappac0*(nus/nuc0)**alphac
