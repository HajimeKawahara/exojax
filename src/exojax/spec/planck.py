"""planck functions.

* The units are nu [1/cm], f [1/s], wav [cm]. In this module, the input variable is always wavenumber [cm].
* nu = f/c = 1/lambda.
* Recall nu B_nu = lambda B_lambda = f B_f.
* B_f (erg/s/cm2/Hz) = B_nu/c (erg/s/cm2/cm-1)
"""

import jax.numpy as jnp
from exojax.utils.constants import hcperk

fac_planck = 3.741771790075259e-05


def piBarr(Tarr, nu_grid):
    """pi B_nu (Planck Function)

    Args:
       Tarr (array): temperature in the unit of K [Nlayer]
       nu_grid (array): wavenumber grid in the unit of cm-1 [Nnu]

    Returns:
       pi B_nu (erg/s/cm2/cm-1) [Nlayer, Nnu]

    Note:
       hcperk = hc/k in cgs, fac = 2*h*c*c*pi in cgs
    """
    
    return (fac_planck*nu_grid**3)/(jnp.exp(hcperk*nu_grid/Tarr[:, None])-1.0)

def piB(T, nu_grid):
    """pi B_nu (Planck Function)

    Args:
        T (float): temperature in the unit of K
        nu_grid (array): wavenumber grid in the unit of cm-1 [Nnu]

    Returns:
        pi B_nu (erg/s/cm2/cm-1) [Nnu]

    Note:
        hcperk = hc/k in cgs, fac = 2*h*c*c*pi in cgs
    """
    return (fac_planck * nu_grid**3) / (jnp.exp(hcperk * nu_grid / T) - 1.0)
