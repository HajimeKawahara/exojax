"""Ackerman and Marley 2001 cloud model.

- Ackerman and Marley (2001) ApJ 556, 872, hereafter AM01
"""
from jax import jit
import jax.numpy as jnp
from jax import vmap


@jit
def VMRcloud(P, Pbase, fsed, VMRbase, kc=1):
    """VMR of clouds based on AM01.

    Args:
        P: Pressure (bar)
        Pbase: base pressure (bar)
        fsed: fsed
        VMRbase: VMR of condensate at cloud base
        kc: constant ratio of condenstates to total mixing ratio

    Returns:
        VMR of condensates
    """
    VMRc = jnp.where(Pbase > P, VMRbase*(P/Pbase)**(fsed/kc), 0.0)
    return VMRc


@jit
def get_Pbase(Parr, Psat, VMR):
    """get Pbase from an intersection of a T-P profile and Psat(T) curves
    Args:
        Parr: pressure array
        Psat: saturation pressure arrau
        VMR: VMR for vapor

    Returns:
        Pbase: base pressure
    """
    # ibase=jnp.searchsorted((Psat/VMR)-Parr,0.0) # 231 +- 9.2 us
    ibase = jnp.argmin(jnp.abs(jnp.log(Parr)-jnp.log(Psat) +
                       jnp.log(VMR)))  # 73.8 +- 2.9 us
    return Parr[ibase]


def get_rw(vfs, Kzz, L, rarr):
    """compute rw in AM01 implicitly defined by (11)

    Args:
       vfs: terminal velocity (cm/s)
       Kzz: diffusion coefficient (cm2/s)
       L: typical convection scale (cm)
       rarr: condensate scale array

    Returns:
       rw: rw (cm) in AM01. i.e. condensate size that balances an upward transport and sedimentation
    """
    iscale = jnp.searchsorted(vfs, Kzz/L)
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
    rg = rw*fsed**(1.0/alpha)*jnp.exp(-(alpha/2.0+3.0)*(jnp.log(sigmag))**2)
    return rg


def find_rw(rarr, vf, KzzpL):
    """finding rw from rarr and terminal velocity array.

    Args:
        rarr: particle radius array (cm)
        vf: terminal velocity (cm/s)
        KzzpL: Kzz/L in Ackerman and Marley 2001

    Returns:
        rw in Ackerman and Marley 2001
    """
    iscale = jnp.searchsorted(vf, KzzpL)
    rw = rarr[iscale]
    return rw


def dtau_cloudgeo(Parr, muc, rhoc, mu, VMRc, rg, sigmag, g):
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

    fac = jnp.exp(-2.5*jnp.log(sigmag)**2)
    dtau = 1.5*muc/mu*VMRc*fac/(rg*rhoc*g)*Parr*1.e6
    return dtau
