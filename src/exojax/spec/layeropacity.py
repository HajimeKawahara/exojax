""" compute opacity difference in atmospheric layers

"""

from jax import jit, vmap
import jax.numpy as jnp
from exojax.spec.hitrancia import interp_logacia_matrix
from exojax.spec.hminus import log_hminus_continuum
from exojax.atm.idealgas import number_density
from exojax.utils.constants import logkB, logm_ucgs
from exojax.utils.constants import opfac
from exojax.spec.dtau_mmwl import dtauM_mmwl


def layer_optical_depth(dParr, xsmatrix, mixing_ratio, mass, gravity):
    """dtau of the molecular cross section.

    Note:
        opfac=bar_cgs/(m_u (g)). m_u: atomic mass unit. It can be obtained by fac=1.e3/m_u, where m_u = scipy.constants.m_u.

    Args:
        dParr: delta pressure profile (bar) [N_layer]
        xsmatrix: cross section matrix (cm2) [N_layer, N_nus]
        mixing_ratio: volume mixing ratio (VMR) or mass mixing ratio (MMR) [N_layer]
        mass: mean molecular weight for VMR or molecular mass for MMR
        gravity: gravity (cm/s2)

    Returns:
        2D array: optical depth matrix, dtau  [N_layer, N_nus]
    """

    return opfac * xsmatrix * dParr[:, None] * mixing_ratio[:, None] / (
        mass * gravity)


def layer_optical_depth_CIA(nu_grid, temperature, pressure, dParr, vmr1, vmr2,
                            mmw, g, nucia, tcia, logac):
    """dtau of the CIA continuum. Not used in art.

    Args:
        nu_grid: wavenumber matrix (cm-1)
        temperature: temperature array (K)
        pressure: pressure array (bar)
        dParr: delta temperature array (bar)
        vmr1: volume mixing ratio (VMR) for molecules 1 [N_layer]
        vmr2: volume mixing ratio (VMR) for molecules 2 [N_layer]
        mmw: mean molecular weight of atmosphere
        g: gravity (cm2/s)
        nucia: wavenumber array for CIA
        tcia: temperature array for CIA
        logac: log10(absorption coefficient of CIA)

    Returns:
        2D array: optical depth matrix, dtau  [N_layer, N_nus]
    """
    narr = number_density(pressure, temperature)
    lognarr1 = jnp.log10(vmr1 * narr)  # log number density
    lognarr2 = jnp.log10(vmr2 * narr)  # log number density
    logg = jnp.log10(g)
    ddParr = dParr / pressure
    dtauc = (
        10**(interp_logacia_matrix(temperature, nu_grid, nucia, tcia, logac) +
             lognarr1[:, None] + lognarr2[:, None] + logkB - logg - logm_ucgs)
        * temperature[:, None] / mmw * ddParr[:, None])

    return dtauc


def layer_optical_depth_VALD(dParr, xsm, VMR, mean_molecular_weight, gravity):
    """dtau of the atomic (+ionic) cross section from VALD.

    Args:
        dParr: delta pressure profile (bar) [N_layer]
        xsm: cross section matrix (cm2) [N_species x N_layer x N_wav]
        VMR: volume mixing ratio [N_species x N_layer]
        mean_molecular_weight: mean molecular weight [N_layer]
        gravity: gravity (cm/s2)

    Returns:
        2D array: optical depth matrix, dtau  [N_layer, N_nus]
    """
    dtauS = jit(vmap(dtauM_mmwl, (None, 0, 0, None, None)))( \
                            dParr, xsm, VMR, mean_molecular_weight, gravity)
    dtau = jnp.abs(jnp.sum(dtauS, axis=0))
    return dtau


def layer_optical_depth_Hminus(nu_grid, temperature, Parr, dParr, vmre, vmrh,
                               mmw, g):
    """dtau of the H- continuum.

    Args:
        nu_grid: wavenumber matrix (cm-1)
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
    narr = number_density(Parr, temperature)
    number_density_e = vmre * narr  # number density for e- [N_layer]
    number_density_h = vmrh * narr  # number density for H atoms [N_layer]
    logg = jnp.log10(g)
    ddParr = dParr / Parr
    logabc = (log_hminus_continuum(nu_grid, temperature, number_density_e,
                                   number_density_h))
    dtauh = 10**(logabc + logkB - logg -
                 logm_ucgs) * temperature[:, None] / mmw * ddParr[:, None]

    return dtauh
