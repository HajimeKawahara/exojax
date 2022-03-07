"""planck functions.

* The units are nu [1/cm], f [1/s], wav [cm]. In this module, the input variable is always wavenumber [cm].
* nu = f/c = 1/lambda.
* Recall nu B_nu = lambda B_lambda = f B_f.
* B_f (erg/s/cm2/Hz) = B_nu/c (erg/s/cm2/cm-1)
"""

import jax.numpy as jnp
from exojax.utils.constants import hcperk


def piBarr(T, nus):
    """pi B_nu (Planck Function)

    Args:
       T: temperature [K]
       nus: wavenumber [cm-1]

    Returns:
       pi B_nu (erg/s/cm2/cm-1) [Nlayer x Nnu]

    Note:
       hcperk = hc/k in cgs, fac = 2*h*c*c*pi in cgs
    """
    fac = 3.741771790075259e-05
    return (fac*nus**3)/(jnp.exp(hcperk*nus/T[:, None])-1.0)
