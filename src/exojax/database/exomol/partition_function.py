import jax.numpy as jnp
import numpy as np


def QT_interp(T, T_gQT, gQT):
    """interpolated partition function.

    Args:
        T: temperature
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT

    Returns:
        Q(T) interpolated in jnp.array
    """
    return jnp.interp(T, T_gQT, gQT)


def QT_interp_numpy(T, T_gQT, gQT):
    """interpolated partition function using numpy.

    Args:
        T: temperature
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT

    Returns:
        Q(T) interpolated in np.array
    """
    return np.interp(T, T_gQT, gQT)


def qr_interp(T, Tref, T_gQT, gQT):
    """interpolated partition function ratio.

    Args:
        T: temperature
        Tref: reference temperature
        gQT: jnp array of partition function grid
        T_gQT: jnp array of temperature grid for gQT

    Returns:
        qr(T)=Q(T)/Q(Tref) interpolated
    """
    return QT_interp(T, T_gQT, gQT) / QT_interp(Tref, T_gQT, gQT)
