"""Kernels for Discrete Integral Transform.

* Fourier kernels for the Voigt are given in this module
* For coarsed wavenumber grids, folded one is needed to avoid negative values, See discussion by Dirk van den Bekerom at https://github.com/radis/radis/issues/186#issuecomment-764465580 for details.
"""

import jax.numpy as jnp
from jax import jit
from jax.lax import scan


def voigt_kernel(k, beta, gammaL):
    """Fourier Kernel of the Voigt Profile.

    Args:
        k: conjugated of wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentzian Half Width

    Returns:
        kernel (N_x,N_beta,N_gammaL)

    Note:
        Conversions to the (full) width, wG and wL are as follows:
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    """
    val = (jnp.pi*beta[None, :, None]*k[:, None, None])**2 + \
        jnp.pi*gammaL[None, None, :]*k[:, None, None]
    return jnp.exp(-2.0*val)


@jit
def fold_voigt_kernel(k, beta, gammaL, vmax, pmarray):
    """Fourier Kernel of the Voigt Profile.

    Args:
        k: conjugated of wavenumber
        beta: Gaussian standard deviation
        gammaL: Lorentian Half Width
        vmax: Nnu x dq
        pmarray: (+1,-1) array whose length of len(nu_grid)+1

    Returns:
        kernel (N_x,N_beta,N_gammaL)

    Note:
        Conversions to the (full) width, wG and wL are as follows:
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    """

    Nk = len(k)
    valG = jnp.exp(-2.0*(jnp.pi*beta[None, :, None]*k[:, None, None])**2)
    valL = jnp.exp(-2.0*jnp.pi*gammaL[None, None, :]*k[:, None, None])
    q = 2.0*gammaL/(vmax)  # Ngamma w=2*gamma

    w_corr = vmax*(0.39560962 * jnp.exp(0.19461568*q**2))  # Ngamma
    A_corr = q*(0.09432246 * jnp.exp(-0.06592025*q**2))  # Ngamma
    B_corr = q*(0.11202818 * jnp.exp(-0.09048447*q**2))  # Ngamma
    zeroindex = jnp.zeros(Nk, dtype=int)  # Nk
    zeroindex = zeroindex.at[0].add(1.0)
    C_corr = zeroindex[:, None]*2.0*B_corr[None, :]  # Nk x Ngamma
    I_corr = A_corr / \
        (1.0+4.0*jnp.pi**2*w_corr[None, None, :] **
         2*k[:, None, None]**2) + C_corr[:, None, :]
    I_corr = I_corr*pmarray[:, None, None]
    valL = valL - I_corr

    return valG*valL


def voigt_kernel_logst(k, log_nstbeta, log_ngammaL):
    """Fourier Kernel of the Voigt Profile for a common normalized beta.

    Args:
        k: conjugate wavenumber
        log_nstbeta: log normalized Gaussian standard deviation (scalar)
        log_ngammaL: log normalized Lorentian Half Width (Nlines)

    Returns:
        kernel (N_x,N_gammaL)

    Note:
        Conversions to the (full) width, wG and wL are as follows:
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    """

    beta = jnp.exp(log_nstbeta)
    gammaL = jnp.exp(log_ngammaL)
    val = jnp.exp(-2.0*((jnp.pi*beta*k[:, None])
                  ** 2 + jnp.pi*gammaL[None, :]*k[:, None]))

    return val


def fold_voigt_kernel_logst(k, log_nstbeta, log_ngammaL, vmax, pmarray):
    """Folded Fourier Kernel of the Voigt Profile for a common normalized beta.
    See https://github.com/dcmvdbekerom/discrete-integral-
    transform/blob/master/demo/discrete_integral_transform_log.py for the alias
    correction.

    Args:
        k: conjugate wavenumber
        log_nstbeta: log normalized Gaussian standard deviation (scalar)
        log_ngammaL: log normalized Lorentian Half Width (Nlines)
        vmax: vmax
        pmarray: (+1,-1) array whose length of len(nu_grid)+1

    Returns:
        kernel (N_x,N_gammaL)

    Note:
        Conversions to the (full) width, wG and wL are as follows:
        wG=2*sqrt(2*ln2) beta
        wL=2*gamma
    """

    beta = jnp.exp(log_nstbeta)
    gammaL = jnp.exp(log_ngammaL)

    Nk = len(k)
    valG = jnp.exp(-2.0*(jnp.pi*beta*k[:, None])**2)
    valL = jnp.exp(-2.0*jnp.pi*gammaL[None, :]*k[:, None])

    q = 2.0*gammaL/(vmax)  # Ngamma w=2*gamma
    w_corr = vmax*(0.39560962 * jnp.exp(0.19461568*q**2))  # Ngamma
    A_corr = q*(0.09432246 * jnp.exp(-0.06592025*q**2))  # Ngamma
    B_corr = q*(0.11202818 * jnp.exp(-0.09048447*q**2))  # Ngamma
    zeroindex = jnp.zeros(Nk, dtype=int)  # Nk
    zeroindex = zeroindex.at[0].add(1.0)
    C_corr = zeroindex[:, None]*2.0*B_corr[None, :]  # Nk x Ngamma
    I_corr = A_corr/(1.0+4.0*jnp.pi**2 *
                     w_corr[None, :]**2*k[:, None]**2) + C_corr[:, :]
    I_corr = I_corr*pmarray[:, None]
    valL = valL - I_corr

    return valG*valL
