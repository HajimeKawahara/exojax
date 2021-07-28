"""Ackerman and Marley 2001 cloud model

   - Ackerman and Marley (2001) ApJ 556, 872


"""

def vterminal(drho,eta,Nkn):
    """terminal velocity of droplets

    Note:
       Based on equation (B1) in Appendix B of AM01. See also (10-138) p415 in Hans R Pruppacher and James D Klett. Microstructure of atmospheric clouds and precipitation. InMicrophysics of clouds and precipitation, pages 10–73. Springer, 2010

    Args: 
       drho: density difference between condensates and atmosphere
       eta: dynamic viscocity of the atmosphere
       Nkn: Knudsen number (ratio of the molecular mean free path to the droplet radius)
       
    Returns:
       terminal velocity

    """

    vf=2.0/9.0*(1.0+1.26*Nkn)*drho/eta
    
    return vf
    
