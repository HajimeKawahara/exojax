"""Atmospheric profile function


"""

from exojax.utils.constants import kB
from exojax.utils.constants import m_u

def Hatm(g,T,mu):
    """pressure scale height assuming an isothermal atmosphere
    
    Args:
        g: gravity acceleration (cm/s2)
        T: isothermal temperature (K)
        mu: mean molecular weight
        
    Returns:
        pressure scale height (cm)
    
    """
    
    return kB*T/(m_u*mu*g)
