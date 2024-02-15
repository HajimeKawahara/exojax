"""Ackerman and Marley 2001 cloud model.

    - based on Ackerman and Marley (2001) ApJ 556, 872, hereafter AM01

"""

from jax import jit
import jax.numpy as jnp
from jax import vmap


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


@jit
def compute_cloud_base_pressure(pressure, saturation_pressure, vmr_vapor):
    """computes cloud base pressure from an intersection of a T-P profile and Psat(T) curves
    Args:
        pressure: pressure array
        saturation_presure: saturation pressure arrau
        vmr_vapor: volume mixing ratio (VMR) for vapor

    Returns:
        cloud base pressure
    """
    # ibase=jnp.searchsorted((Psat/VMR)-Parr,0.0) # 231 +- 9.2 us
    ibase = jnp.argmin(
        jnp.abs(jnp.log(pressure) - jnp.log(saturation_pressure) + jnp.log(vmr_vapor))
    )  # 73.8 +- 2.9 us
    return pressure[ibase]


@jit
def compute_cloud_base_pressure_index(pressure, saturation_pressure, vmr_vapor):
    """computes cloud base pressure from an intersection of a T-P profile and Psat(T) curves
    Args:
        pressure: pressure array
        saturation_presure: saturation pressure arrau
        vmr_vapor: volume mixing ratio (VMR) for vapor

    Returns:
        int: cloud base pressure index
    """
    # ibase=jnp.searchsorted((Psat/VMR)-Parr,0.0) # 231 +- 9.2 us
    ibase = jnp.argmin(
        jnp.abs(jnp.log(pressure) - jnp.log(saturation_pressure) + jnp.log(vmr_vapor))
    )  # 73.8 +- 2.9 us
    return ibase


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
        sigmag:sigmag parameter in the lognormal distribution of condensate size, defined by (9) in AM01

    Returns
        rg: rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01

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


def normalization_lognormal(
    rg, sigmag, mmr_condenstate, atmosphere_density, condensate_bulk_density
):
    """normalization N(z) of the lognormal cloud distribution

    Args:
        rg (_type_): rg paramter
        sigmag (_type_): sigmag parameter
        mmr_condenstate (_type_): mass mixing ratio of the condensate
        atmosphere_density (_type_): (mass) density of the atmosphere (g/cm3)
        condensate_bulk_density (_type_): condenstate bulk density (g/cm3)

    Returns:
        _type_: _description_
    """

    fac = 3.0 / 4.0 / jnp.pi * jnp.exp(-4.5 * (jnp.log(sigmag)) ** 2)
    return fac * mmr_condenstate * atmosphere_density / condensate_bulk_density / rg**3


def layer_optical_depth_cloudgeo(Parr, rhoc, MMRc, rg, sigmag, g):
    """the optical depth using a geometric cross-section approximation, based
    on (16) in AM01.

    Args:
        Parr: pressure array (bar)
        rhoc: condensate density (g/cm3)
        MMRc: Mass mixing ratio (array) of condensate [Nlayer]
        rg: rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
        sigmag:sigmag parameter in the lognormal distribution of condensate size, defined by (9) in AM01
    """

    fac = jnp.exp(-2.5 * jnp.log(sigmag) ** 2)
    dtau = 1.5 * MMRc * fac / (rg * rhoc * g) * Parr * 1.0e6
    return dtau
