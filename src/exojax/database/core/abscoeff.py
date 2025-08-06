"""Interpolates the logarithm of the absorption coefficient."""

import jax.numpy as jnp
from jax import jit, vmap


@jit
def interp_logacia_matrix(Tarr, nu_grid, nucia, tcia, logac):
    """interpolated function of log10(alpha_CIA)

    Args:
        Tarr (1D array): temperature array (K) [Nlayer]
        nu_grid (1D array): wavenumber array (cm-1) [Nnus]
        nucia: CIA wavenumber (cm-1)
        tcia: CIA temperature (K)
        logac: log10 cia coefficient

    Returns:
        logarithm of absorption coefficient [Nlayer, Nnus] in the unit of cm5

    Example:
        nucia,tcia,ac=read_cia("../../data/CIA/H2-H2_2011.cia",nus[0]-1.0,nus[-1]+1.0)
        logac=jnp.array(np.log10(ac))
        interp_logacia_matrix(Tarr,nus,nucia,tcia,logac)
    """

    def fcia(x, i):
        return jnp.interp(x, tcia, logac[:, i])

    vfcia = vmap(fcia, (None, 0), 0)
    mfcia = vmap(vfcia, (0, None), 0)
    inus = jnp.digitize(nu_grid, nucia)
    return mfcia(Tarr, inus)


@jit
def interp_logacia_vector(T, nu_grid, nucia, tcia, logac):
    """interpolated function of log10(alpha_CIA)

    Args:
        T (float): temperature (K)
        nu_grid: wavenumber array (cm-1)
        nucia: CIA wavenumber (cm-1)
        tcia: CIA temperature (K)
        logac: log10 cia coefficient

    Returns:
        logarithm of absorption coefficient [Nnus] at T in the unit of cm5
        

    """

    def fcia(x, i):
        return jnp.interp(x, tcia, logac[:, i])

    vfcia = vmap(fcia, (None, 0), 0)
    inus = jnp.digitize(nu_grid, nucia)
    return vfcia(T, inus)
