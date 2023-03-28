"""compute dtau (opacity difference in atmospheric layers) using mean molecular weight

"""


import jax.numpy as jnp
from exojax.spec.hitrancia import interp_logacia_matrix
from exojax.spec.hminus import log_hminus_continuum
from exojax.atm.idealgas import number_density
from exojax.utils.constants import logkB, logm_ucgs
from exojax.utils.constants import opfac
import warnings

warnings.warn("dtau_mmwl might be removed in future.", FutureWarning)

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

