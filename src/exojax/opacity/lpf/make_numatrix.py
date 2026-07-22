import warnings

import jax.numpy as jnp
import numpy as np
from jax import jit

from exojax.utils.checkarray import require_ndim
from exojax.utils.constants import c


def make_numatrix0(nu, hatnu, warning=True):
    """Generate numatrix0.

    Note:
        Use float64 as inputs.

    Args:
        nu: wavenumber matrix (Nnu,)
        hatnu: line center wavenumber vector (Nline,), where Nm is the number of lines
        warning: True=warning on for nu.dtype=float32

    Returns:
        numatrix (Nline,Nnu)
    """
    if nu.dtype != np.float64 and warning:
        warnings.warn(
            "wavenumber grid is not np.float64 but " + str(nu.dtype), UserWarning
        )
    if hatnu.dtype != np.float64 and warning:
        warnings.warn("line center is not np.float64 but " + str(nu.dtype), UserWarning)
    numatrix = nu[None, :] - hatnu[:, None]
    return jnp.array(numatrix)


def doppler_shifted_line_detuning(numatrix0, nu_lines, velocity_los):
    """Add layer-dependent Doppler shifts to a rest-frame LPF numatrix.

    This function applies the velocity correction to an already computed
    rest-frame detuning matrix. Keeping the large absolute wavenumbers out of
    the subtraction avoids loss of small detunings in JAX float32 mode.

    Args:
        numatrix0: Rest-frame detuning matrix in cm-1 with shape
            ``(N_line, N_wavenumber)``.
        nu_lines: Rest line centers in cm-1 with shape ``(N_line,)``.
        velocity_los: Line-of-sight velocity in km/s. A vector has shape
            ``(N_layer,)`` and applies the same velocity to every line. A
            matrix must have shape ``(N_layer, N_line)``. Positive velocity is
            receding/redshift. Values must be greater than ``-c``.

    Returns:
        Detuning tensor with shape ``(N_layer, N_line, N_wavenumber)``.
    """
    numatrix0 = jnp.asarray(numatrix0)
    nu_lines = jnp.asarray(nu_lines)
    velocity_los = jnp.asarray(velocity_los)

    require_ndim("numatrix0", numatrix0, 2)
    require_ndim("nu_lines", nu_lines, 1)
    if numatrix0.shape[0] != nu_lines.shape[0]:
        raise ValueError(
            "numatrix0 line axis must match nu_lines: "
            f"{numatrix0.shape[0]} != {nu_lines.shape[0]}."
        )

    if velocity_los.ndim == 1:
        velocity_los = velocity_los[:, None]
    elif velocity_los.ndim == 2:
        if velocity_los.shape[1] != nu_lines.shape[0]:
            raise ValueError(
                "velocity_los line axis must match nu_lines: "
                f"{velocity_los.shape[1]} != {nu_lines.shape[0]}."
            )
    else:
        raise ValueError(
            "velocity_los must have shape (N_layer,) or (N_layer, N_line), "
            f"got {velocity_los.shape}."
        )

    velocity_ratio = velocity_los / c
    center_correction = nu_lines[None, :] * velocity_ratio / (1.0 + velocity_ratio)
    return numatrix0[None, :, :] + center_correction[:, :, None]


def divwavnum(nu, Nz=1):
    """separate an integer part from a residual.

    Args:
        nu: wavenumber array
        Nz: boost factor (default=1)

    Returns:
        integer part of wavenumber, residual wavenumber, boost factor
    """

    fn = np.floor(nu * Nz)
    dfn = nu * Nz - fn
    return fn, dfn, Nz


@jit
def subtract_nu(dnu, dhatnu):
    """compute nu - hatnu by subtracting an integer part w/JIT

    Args:
        dnu: residual wavenumber array
        dhatnu: residual line center array

    Returns:
        difference matrix

    """
    jdnu = jnp.array(dnu)
    jdhatnu = jnp.array(dhatnu)
    dd = jdnu[None, :] - jdhatnu[:, None]
    return dd


@jit
def add_nu(dd, fnu, fhatnu, Nz):
    """re-adding an interger part w/JIT.

    Args:
        dd: difference matrix
        fnu: integer part of wavenumber
        fhatnu: residual wavenumber
        Nz: boost factor

    Returns:
        an integer part readded value

    """
    jfnu = jnp.array(fnu)
    jfhatnu = jnp.array(fhatnu)
    #    intarray=fnu[None,:]-fhatnu[:,None]
    intarray = jfnu[None, :] - jfhatnu[:, None]
    return (dd + intarray) / Nz


def make_numatrix0_subtract(nu, hatnu, Nz=1, warning=True):
    """Generate numatrix0 using gpu.

    Note:
        This function computes a wavenumber matrix using XLA. Because XLA does not support float64, a direct computation sometimes results in large uncertainity. For instace, let's assume nu=2000.0396123 cm-1 and hatnu=2000.0396122 cm-1. If applying float32, we get np.float32(2000.0396123)-np.float32(2000.0396122) = 0.0. But, after subtracting 2000 from both nu and hatnu, we get np.float32(0.0396123)-np.float32(0.0396122)=1.0058284e-07. make_numatrix0 does such computation. Nz=1 means we subtract a integer part (i.e. 2000), Nz=10 means we subtract 2000.0, and Nz=10 means we subtract 2000.00.

    Args:
        nu: wavenumber matrix (Nnu,)
        hatnu: line center wavenumber vector (Nline,), where Nm is the number of lines
        Nz: boost factor (default=1)
        warning: True=warning on for nu.dtype=float32

    Returns:
        numatrix0: wavenumber matrix w/ no shift
    """

    fnu, dnu, Nz = divwavnum(nu, Nz)
    fhatnu, dhatnu, Nz = divwavnum(hatnu, Nz)
    dd = subtract_nu(dnu, dhatnu)
    numatrix0 = add_nu(dd, fnu, fhatnu, Nz)
    return numatrix0
