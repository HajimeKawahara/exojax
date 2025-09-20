import warnings
import jax.numpy as jnp
import numpy as np
from exojax.database._common.isotope_functions import _isotope_index_from_isotope_number


def _QT_interp(isotope_index, T, T_gQT, gQT):
    """interpolated partition function.

    Note:
        isotope_index is NOT isotope (number for HITRAN).
        isotope_index is index for gQT and T_gQT.
        _isotope_index_from_isotope_number can be used
        to get isotope index from isotope.

    Args:
        isotope index: isotope index, index from 0 to len(uniqiso) - 1
        T: temperature
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT

    Returns:
        Q(idx, T) interpolated in jnp.array
    """

    return jnp.interp(T, T_gQT[isotope_index], gQT[isotope_index])


def _qr_interp(isotope_index, T, T_gQT, gQT, Tref):
    """interpolated partition function ratio.

    Note:
        isotope_index is NOT isotope (number for HITRAN).
        isotope_index is index for gQT and T_gQT.
        _isotope_index_from_isotope_number can be used
        to get isotope index from isotope.

    Args:
        isotope index: isotope index, index from 0 to len(uniqiso) - 1
        T: temperature
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT
        Tref: reference temperature in K

    Returns:
        qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
    """
    return _QT_interp(isotope_index, T, T_gQT, gQT) / _QT_interp(
        isotope_index, Tref, T_gQT, gQT
    )


def _qr_interp_lines(T, isoid, uniqiso, T_gQT, gQT, Tref):
    """Partition Function ratio using HAPI partition data.
    (This function works for JAX environment.)

    Args:
        T: temperature (K)
        isoid:
        uniqiso:
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT
        Tref: reference temperature in K

    Returns:
        Qr_line, partition function ratio array for lines [Nlines]

    Note:
        Nlines=len(self.nu_lines)
    """
    qr_line = jnp.zeros(len(isoid))
    for isotope in uniqiso:
        mask_idx = np.where(isoid == isotope)
        isotope_index = _isotope_index_from_isotope_number(isotope, uniqiso)
        qr_each_isotope = _qr_interp(isotope_index, T, T_gQT, gQT, Tref)
        qr_line = qr_line.at[jnp.index_exp[mask_idx]].set(qr_each_isotope)
    return qr_line


def qr_interp(isotope, uniqiso, T, Tref, T_gQT, gQT):
    """interpolated partition function ratio.

    Args:
        isotope: HITRAN isotope number starting from 1
        uniqiso: unique isotope array
        T: temperature
        Tref: reference temperature
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT

    Returns:
        qr(T)=Q(T)/Q(Tref) interpolated in jnp.array
    """
    if isotope is None or isotope == 0:
        msg1 = "Currently all isotope mode is not fully compatible to MdbCommonHitempHitran."
        msg2 = "QT assumed isotope=1 instead."
        warnings.warn(msg1 + msg2, UserWarning)
        isotope = 1
    isotope_index = _isotope_index_from_isotope_number(isotope, uniqiso)
    return _qr_interp(isotope_index, T, T_gQT, gQT, Tref)
