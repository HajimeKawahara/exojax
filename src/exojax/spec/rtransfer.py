"""Radiative transfer module used in exospectral analysis."""
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
from exojax.special.expn import E1
from exojax.spec.hitrancia import interp_logacia_matrix
from exojax.spec.hminus import log_hminus_continuum
from exojax.atm.idealgas import number_density
from exojax.utils.constants import logkB, logm_ucgs
from exojax.utils.constants import opfac
import warnings


def wavenumber_grid(x0, x1, N, unit='cm-1', xsmode='lpf'):
    warn_msg = "Use `grids.wavenumber_grid` instead"
    warnings.warn(warn_msg, FutureWarning)
    from exojax.utils.grids import wavenumber_grid
    return wavenumber_grid(x0, x1, N, unit=unit, xsmode=xsmode)


def pressure_layer(logPtop=-8.,
                   logPbtm=2.,
                   NP=20,
                   mode='ascending',
                   reference_point=0.5,
                   numpy=False):
    """generating the pressure layer.

    Args:
        logPtop: log10(P[bar]) at the top layer
        logPbtm: log10(P[bar]) at the bottom layer
        NP: the number of the layers
        mode: ascending or descending
        reference_point: reference point in the layer. 0.5:center, 1.0:lower boundary, 0.0:upper boundary
        numpy: if True use numpy array instead of jnp array

    Returns:
        Parr: pressure layer
        dParr: delta pressure layer
        k: k-factor, P[i-1] = k*P[i]

    Note:
        dParr[i] = Parr[i] - Parr[i-1], dParr[0] = (1-k) Parr[0] for ascending mode
    """
    dlog10P = (logPbtm - logPtop) / (NP - 1)
    k = 10**-dlog10P
    if numpy:
        Parr = np.logspace(logPtop, logPbtm, NP)
    else:
        Parr = jnp.logspace(logPtop, logPbtm, NP)
    dParr = (1.0 - k**reference_point) * Parr
    if mode == 'descending':
        Parr = Parr[::-1]
        dParr = dParr[::-1]

    return Parr, dParr, k


def dtauCIA(nus, Tarr, Parr, dParr, vmr1, vmr2, mmw, g, nucia, tcia, logac):
    """dtau of the CIA continuum.

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmr1: volume mixing ratio (VMR) for molecules 1 [N_layer]
       vmr2: volume mixing ratio (VMR) for molecules 2 [N_layer]
       mmw: mean molecular weight of atmosphere
       g: gravity (cm2/s)
       nucia: wavenumber array for CIA
       tcia: temperature array for CIA
       logac: log10(absorption coefficient of CIA)

    Returns:
       optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(Parr, Tarr)
    lognarr1 = jnp.log10(vmr1 * narr)  # log number density
    lognarr2 = jnp.log10(vmr2 * narr)  # log number density
    logg = jnp.log10(g)
    ddParr = dParr / Parr
    dtauc = (10**(interp_logacia_matrix(Tarr, nus, nucia, tcia, logac) +
                  lognarr1[:, None] + lognarr2[:, None] + logkB - logg -
                  logm_ucgs) * Tarr[:, None] / mmw * ddParr[:, None])

    return dtauc


def dtauCIA_mmwl(nus, Tarr, Parr, dParr, vmr1, vmr2, mmw, g, nucia, tcia,
                 logac):
    """dtau of the CIA continuum.
       (for the case where mmw is given for each atmospheric layer)

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmr1: volume mixing ratio (VMR) for molecules 1 [N_layer]
       vmr2: volume mixing ratio (VMR) for molecules 2 [N_layer]
       mmw: mean molecular weight of atmosphere [N_layer]
       g: gravity (cm2/s)
       nucia: wavenumber array for CIA
       tcia: temperature array for CIA
       logac: log10(absorption coefficient of CIA)

    Returns:
       optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(Parr, Tarr)
    lognarr1 = jnp.log10(vmr1 * narr)  # log number density
    lognarr2 = jnp.log10(vmr2 * narr)  # log number density
    logg = jnp.log10(g)
    ddParr = dParr / Parr
    dtauc = (10**(interp_logacia_matrix(Tarr, nus, nucia, tcia, logac) +
                  lognarr1[:, None] + lognarr2[:, None] + logkB - logg -
                  logm_ucgs) * Tarr[:, None] / mmw[:, None] * ddParr[:, None])

    return dtauc


def dtauM(dParr, xsm, MR, mass, g):
    """dtau of the molecular cross section.

    Note:
       opfac=bar_cgs/(m_u (g)). m_u: atomic mass unit. It can be obtained by fac=1.e3/m_u, where m_u = scipy.constants.m_u.

    Args:
       dParr: delta pressure profile (bar) [N_layer]
       xsm: cross section matrix (cm2) [N_layer, N_nus]
       MR: volume mixing ratio (VMR) or mass mixing ratio (MMR) [N_layer]
       mass: mean molecular weight for VMR or molecular mass for MMR
       g: gravity (cm/s2)

    Returns:
       optical depth matrix [N_layer, N_nus]
    """

    return opfac * xsm * dParr[:, None] * MR[:, None] / (mass * g)


def dtauM_mmwl(dParr, xsm, MR, mass, g):
    """dtau of the molecular cross section.
       (for the case where mmw is given for each atmospheric layer)

    Note:
       opfac=bar_cgs/(m_u (g)). m_u: atomic mass unit. It can be obtained by fac=1.e3/m_u, where m_u = scipy.constants.m_u.

    Args:
       dParr: delta pressure profile (bar) [N_layer]
       xsm: cross section matrix (cm2) [N_layer, N_nus]
       MR: volume mixing ratio (VMR) or mass mixing ratio (MMR) [N_layer]
       mass: mean molecular weight for VMR or molecular mass for MMR [N_layer]
       g: gravity (cm/s2)

    Returns:
       optical depth matrix [N_layer, N_nus]
    """

    return opfac * xsm * dParr[:, None] * MR[:, None] / (mass[:, None] * g)


@jit
def dtauVALD(dParr, xsm, VMR, mmw, g):
    """dtau of the atomic (+ionic) cross section from VALD.

    Args:
       dParr: delta pressure profile (bar) [N_layer]
       xsm: cross section matrix (cm2) [N_species x N_layer x N_wav]
       VMR: volume mixing ratio [N_species x N_layer]
       mmw: mean molecular weight [N_layer]
       g: gravity (cm/s2)

    Returns:
        dtau: optical depth matrix [N_layer x N_nus]
    """
    dtauS = jit(vmap(dtauM_mmwl, (None, 0, 0, None, None)))( \
                            dParr, xsm, VMR, mmw, g)
    dtau = jnp.abs(jnp.sum(dtauS, axis=0))
    return dtau


def dtauHminus(nus, Tarr, Parr, dParr, vmre, vmrh, mmw, g):
    """dtau of the H- continuum.

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmre: volume mixing ratio (VMR) for e- [N_layer]
       vmrH: volume mixing ratio (VMR) for H atoms [N_layer]
       mmw: mean molecular weight of atmosphere
       g: gravity (cm2/s)

    Returns:
       optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(Parr, Tarr)
    #       number_density_e: number density for e- [N_layer]
    #       number_density_h: number density for H atoms [N_layer]
    number_density_e = vmre * narr
    number_density_h = vmrh * narr
    logg = jnp.log10(g)
    ddParr = dParr / Parr
    logabc = (log_hminus_continuum(nus, Tarr, number_density_e,
                                   number_density_h))
    dtauh = 10**(logabc + logkB - logg -
                 logm_ucgs) * Tarr[:, None] / mmw * ddParr[:, None]

    return dtauh


def dtauHminus_mmwl(nus, Tarr, Parr, dParr, vmre, vmrh, mmw, g):
    """dtau of the H- continuum.
       (for the case where mmw is given for each atmospheric layer)

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmre: volume mixing ratio (VMR) for e- [N_layer]
       vmrH: volume mixing ratio (VMR) for H atoms [N_layer]
       mmw: mean molecular weight of atmosphere [N_layer]
       g: gravity (cm2/s)

    Returns:
       optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(Parr, Tarr)
    #       number_density_e: number density for e- [N_layer]
    #       number_density_h: number density for H atoms [N_layer]
    number_density_e = vmre * narr
    number_density_h = vmrh * narr
    logg = jnp.log10(g)
    ddParr = dParr / Parr
    logabc = (log_hminus_continuum(nus, Tarr, number_density_e,
                                   number_density_h))
    dtauh = 10**(logabc + logkB - logg -
                 logm_ucgs) * Tarr[:, None] / mmw[:, None] * ddParr[:, None]

    return dtauh


@jit
def trans2E3(x):
    """transmission function 2E3 (two-stream approximation with no scattering)
    expressed by 2 E3(x)

    Note:
       The exponetial integral of the third order E3(x) is computed using Abramowitz Stegun (1970) approximation of E1 (exojax.special.E1).

    Args:
       x: input variable

    Returns:
       Transmission function T=2 E3(x)
    """
    return (1.0 - x) * jnp.exp(-x) + x**2 * E1(x)


@jit
def rtrun(dtau, S):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type)

    Args:
        dtau: opacity matrix
        S: source matrix [N_layer, N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus = jnp.shape(dtau)[1]
    TransM = jnp.where(dtau == 0, 1.0, trans2E3(dtau))
    Qv = jnp.vstack([(1 - TransM) * S, jnp.zeros(Nnus)])
    return jnp.sum(Qv *
                   jnp.cumprod(jnp.vstack([jnp.ones(Nnus), TransM]), axis=0),
                   axis=0)


@jit
def rtrun_surface(dtau, S, Sb):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type)
    with a planetary surface.

    Args:
        dtau: opacity matrix
        S: source matrix [N_layer, N_nus]
        Sb: source from the surface [N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus = jnp.shape(dtau)[1]
    TransM = jnp.where(dtau == 0, 1.0, trans2E3(dtau))
    Qv = jnp.vstack([(1 - TransM) * S, Sb])
    return jnp.sum(Qv *
                   jnp.cumprod(jnp.vstack([jnp.ones(Nnus), TransM]), axis=0),
                   axis=0)


@jit
def rtrun_direct(dtau, S):
    """Radiative Transfer using direct integration.

    Note:
        Use dtau/mu instead of dtau when you want to use non-unity, where mu=cos(theta)

    Args:
        dtau: opacity matrix
        S: source matrix [N_layer, N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    taupmu = jnp.cumsum(dtau, axis=0)
    return jnp.sum(S * jnp.exp(-taupmu) * dtau, axis=0)
