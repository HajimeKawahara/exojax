"""Ackerman and Marley 2001 cloud model.

- Ackerman and Marley (2001) ApJ 556, 872, hereafter AM01

"""
from jax import jit
import jax.numpy as jnp
from jax import vmap


def vmr_cloud_pressure(pressure, cloud_base_pressure, fsed, vmr_cloud_base, kc):
    """volume mixing ratio of clouds based on AM01 a given single pressure.

    Args:
        pressure (float): pressure (bar) where we want to compute VMR of clouds
        cloud_base_pressure: cloud base pressure (bar)
        fsed: fsed
        vmr_cloud_base: volume mixing ratio (VMR) of condensate at cloud base
        kc: constant ratio of condenstates to total mixing ratio

    Returns:
        VMR of condensates
    """
    return jnp.where(
        cloud_base_pressure > pressure,
        vmr_cloud_base * (pressure / cloud_base_pressure) ** (fsed / kc),
        0.0,
    )


@jit
def vmr_cloud_profile(pressures, cloud_base_pressure, fsed, vmr_cloud_base, kc=1):
    """volume mixing ratio of clouds based on AM01 given pressure.

    Args:
        pressures: pressure array  (Nlayer) (bar) where we want to compute VMR of clouds
        cloud_base_pressure: cloud base pressure (bar)
        fsed: fsed
        vmr_cloud_base: volume mixing ratio (VMR) of condensate at cloud base
        kc: constant ratio of condenstates to total mixing ratio

    Returns:
        VMR of condensates
    """
    vmaped_function = vmap(vmr_cloud_pressure, (0, None, None, None, None), 0)
    return vmaped_function(pressures, cloud_base_pressure, fsed, vmr_cloud_base, kc)


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
       sigmag: sigmag in the lognormal size distribution

    Returns
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


def layer_optical_depth_cloudgeo(Parr, muc, rhoc, mu, VMRc, rg, sigmag, g):
    """the optical depth using a geometric cross-section approximation, based
    on (16) in AM01.

    Args:
       Parr: pressure array (bar)
       muc: mass weight of condensate
       rhoc: condensate density (g/cm3)
       mu: mean molecular weight of atmosphere
       VMRc: VMR array of condensate [Nlayer]
       rg: rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
       sigmag:sigmag parameter in the lognormal distribution of condensate size, defined by (9) in AM01
    """

    fac = jnp.exp(-2.5 * jnp.log(sigmag) ** 2)
    dtau = 1.5 * muc / mu * VMRc * fac / (rg * rhoc * g) * Parr * 1.0e6
    return dtau
