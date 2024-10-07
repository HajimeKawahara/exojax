"""Ackerman and Marley 2001 cloud model.

    - based on Ackerman and Marley (2001) ApJ 556, 872, hereafter AM01

"""

from jax import jit
import jax.numpy as jnp
from jax import vmap
from exojax.utils.indexing import get_smooth_index
from exojax.utils.indexing import get_value_at_smooth_index


def mixing_ratio_cloud_pressure(pressure, cloud_base_pressure, fsed, mr_cloud_base, kc):
    """mol mixing ratio of clouds based on AM01 a given single pressure.

    Args:
        pressure (float): pressure (bar) where we want to compute VMR of clouds
        cloud_base_pressure: cloud base pressure (bar)
        fsed: fsed
        mr_cloud_base: mass mixing ratio (MMR) or mol mixing ratio of condensate at cloud base
        kc: constant ratio of condenstates to total mixing ratio

    Returns:
        mol mixing ratio of condensates
    """
    return jnp.where(
        cloud_base_pressure > pressure,
        mr_cloud_base * (pressure / cloud_base_pressure) ** (fsed / kc),
        0.0,
    )


@jit
def mixing_ratio_cloud_profile(
    pressures, cloud_base_pressure, fsed, mr_cloud_base, kc=1
):
    """volume mixing ratio of clouds based on AM01 given pressure.

    Args:
        pressures: pressure array  (Nlayer) (bar) where we want to compute VMR of clouds
        cloud_base_pressure: cloud base pressure (bar)
        fsed: fsed
        mr_cloud_base: mass mixing ratio (MMR) or mol mixing ratio of condensate at cloud base
        kc: constant ratio of condenstates to total mixing ratio

    Returns:
        VMR of condensates
    """
    vmaped_function = vmap(mixing_ratio_cloud_pressure, (0, None, None, None, None), 0)
    return vmaped_function(pressures, cloud_base_pressure, fsed, mr_cloud_base, kc)



def get_pressure_at_cloud_base(pressures, smooth_index):
    """get pressure at cloud base from pressures

    Args:
        pressures: pressure array
        smooth_index: smooth index

    Returns:
        float: pressure at cloud base
    """
    return 10 ** get_value_at_smooth_index(jnp.log10(pressures), smooth_index)


def smooth_index_base_pressure(pressures, saturation_pressure, vmr_vapor):
    """computes smooth_index for cloud base pressure from an intersection of a T-P profile and Psat(T) curves
    Args:
        pressures: pressure array
        saturation_presure: saturation pressure arrau
        vmr_vapor: volume mixing ratio (VMR) for vapor

    Returns:
        float: smooth_index
    """
    return get_smooth_index(
        jnp.log10((saturation_pressure / vmr_vapor) / pressures), 0.0
    )


def compute_cloud_base_pressure(pressures, saturation_pressure, vmr_vapor):
    """compute cloud base pressure from a T-P profile and Psat(T) curves

    Args:
        pressures: pressure array
        saturation_presure: saturation pressure arrau
        vmr_vapor: volume mixing ratio (VMR) for vapor

    Returns:
        float: cloud base pressure
    """
    smooth_index = smooth_index_base_pressure(pressures, saturation_pressure, vmr_vapor)
    return get_pressure_at_cloud_base(pressures, smooth_index)


def get_rw(terminal_velocity, Kzz, L, rarr):
    """compute rw in AM01 implicitly defined by (11)

    Args:
        vfs: terminal velocity (cm/s)
        Kzz: diffusion coefficient (cm2/s)
        L: typical convection scale (cm)
        rarr: condensate scale array

    Returns:
        rw: rw (cm) in AM01. i.e. condensate size that balances an upward transport and sedimentation
    """
    iscale = jnp.searchsorted(terminal_velocity, Kzz / L)
    rw = rarr[iscale]
    return rw


def get_rg(rw, fsed, alpha, sigmag):
    """compute rg of the lognormal size distribution defined by (9) in AM01.
    The computation is based on (13) in AM01.

    Args:
        rw: rw (cm)
        fsed: fsed
        alpha: power of the condensate size distribution
        sigmag:sigmag parameter (geometric standard deviation) 
            in the lognormal distribution of condensate size, 
            defined by (9) in AM01, must be sigmag > 1

    Returns
        rg: rg parameter in the lognormal distribution of condensate size, 
            defined by (9) in AM01

    """
    rg = (
        rw
        * fsed ** (1.0 / alpha)
        * jnp.exp(-(alpha / 2.0 + 3.0) * (jnp.log(sigmag)) ** 2)
    )
    return rg


def find_rw(rarr, terminal_velocity, Kzz_over_L):
    """finding rw from rarr and terminal velocity array.

    Args:
        rarr: particle radius array (cm)
        terminal_velocity: terminal velocity (cm/s)
        Kzz_over_L: Kzz/L in Ackerman and Marley 2001

    Returns:
        rw in Ackerman and Marley 2001
    """
    iscale = jnp.searchsorted(terminal_velocity, Kzz_over_L)
    rw = rarr[iscale]
    return rw


def sigmag_from_effective_radius(reff, fsed, rw, alpha):
    """computes sigmag from reff

    Args:
        reff: effective radius (cm) defined by E_3/E_2
        fsed: fsed
        rw: rw (cm)
        alpha: alpha

    Returns:
        sigmag: sigmag
    """
    factor = jnp.log(rw / reff * fsed ** (1.0 / alpha))
    index = jnp.sqrt(2.0 / (1.0 + alpha) * factor)
    return jnp.exp(index)


def effective_radius(rg, sigmag):
    """computes the paritculate effective radius from lognormal parameters, rg and sigmag

    Args:
        rg (float): rg parameter in lognormal distribution in cgs
        sigmag (float): sigma_g parameter in lognormal distribution


    Note:
        The cross section is given by $S = Q_e \pi r_eff^2$

    Returns:
        _type_: effective radius in cgs
    """
    return jnp.exp(2.5 * jnp.log(sigmag) * jnp.log(sigmag)) * rg


def geometric_radius(rg, sigmag):
    """computes the paritculate geometric radius

    Args:
        rg (float): rg parameter in lognormal distribution in cgs
        sigmag (float): sigma_g parameter in lognormal distribution


    Note:
        The cross section is given by $S = Q_e \pi r_geo^2$

    Returns:
        _type_: geometric radius in cgs
    """
    # logs = jnp.log(sigmag)
    # return jnp.exp(logs*logs)*rg
    return sigmag ** jnp.log(sigmag) * rg
