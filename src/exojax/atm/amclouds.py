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
    
def eta_H2_Rosner(T):
    """dynamic viscocity of the H2 atmosphere by Rosner (2000)
    
    Args:
       T: temperature (K)
    
    Returns:
       viscosity (g/s/cm)

    Note:
       (B2) in AM01. epsilon=59.7*k, d (H2) =2.827*e-10(m), m=2*m_u

    Example:
       >>> #factor in the code is computed by
       >>> from scipy.constants import m_u, k #SI system
       >>> d=2.827e-10 #m
       >>> si2cgs=10.0 #kg/m/s -> g/cm/s (viscosity)
       >>> 5.0/16.0*np.sqrt(np.pi*k*2*m_u)/(np.pi*d*d)/1.22*(1.0/59.7)**0.16*si2cgs
       2.0127454503595356e-06

    """
    factor=2.01274545e-06 
    eta = factor*(T**0.66)
    return eta


