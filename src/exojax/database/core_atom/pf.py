"""Atomic partition functions and line-specific ratios."""

import jax.numpy as jnp
from jax import vmap


_FE_I_QT_INDEX = 76


def interp_QT_284(T, T_gQT, gQT_284species, Irwin=False):
    """Interpolate partition functions for the tabulated atomic species.

    Args:
        T: Temperature in K.
        T_gQT: Partition-function temperature grid.
        gQT_284species: Partition-function grid, shaped (284, N_temperature).
        Irwin: Use the Irwin (1981) polynomial for Fe I only.

    Returns:
        Partition functions for all tabulated species.
    """
    qt = vmap(jnp.interp, (None, None, 0))(T, T_gQT, gQT_284species)
    is_fe_i = (jnp.arange(qt.shape[0]) == _FE_I_QT_INDEX).reshape(
        (-1,) + (1,) * (qt.ndim - 1)
    )
    return jnp.where(is_fe_i & Irwin, partfn_Fe(T), qt)


def qr_interp_lines(T, Tref, T_gQT, gQT_284species, QTmask, Irwin=False):
    """Return Q(T)/Q(Tref) for each selected atomic line.

    Args:
        T: Temperature in K.
        Tref: Reference temperature in K.
        T_gQT: Partition-function temperature grid.
        gQT_284species: Partition-function grid, shaped (284, N_temperature).
        QTmask: Partition-function row index for each selected line.
        Irwin: Use the Irwin (1981) polynomial for Fe I only.

    Returns:
        Partition-function ratios with one entry per selected line.
    """
    qt = interp_QT_284(T, T_gQT, gQT_284species, Irwin)
    qtref = interp_QT_284(Tref, T_gQT, gQT_284species, Irwin)
    return qt[QTmask] / qtref[QTmask]


def partfn_Fe(T):
    """Return the Fe I partition function from Irwin (1981).

    Args:
        T: Temperature in K.

    Returns:
        Partition function Q(T).
    """
    # The original polynomial is recentered at log(T) = 8 to reduce cancellation.
    coefficients = jnp.array(
        [
            0.0421182087,
            0.0653837980,
            0.1024689680,
            0.1314333440,
            0.3516477760,
            3.0877158816,
        ]
    )
    return jnp.exp(jnp.polyval(coefficients, jnp.log(T) - 8.0))
