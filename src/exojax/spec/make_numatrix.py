from jax import jit
import jax.numpy as jnp
import numpy as np
import warnings


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
    if (nu.dtype != np.float64 and warning):
        warnings.warn('wavenumber grid is not np.float64 but ' + str(nu.dtype),
                      UserWarning)
    if (hatnu.dtype != np.float64 and warning):
        warnings.warn('line center is not np.float64 but ' + str(nu.dtype),
                      UserWarning)
    numatrix = nu[None, :] - hatnu[:, None]
    return jnp.array(numatrix)


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
    dd = (jdnu[None, :] - jdhatnu[:, None])
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from exojax.utils.grids import wavenumber_grid
    from exojax.spec import moldb
    import time
    import numpy as np
    nu_grid, wav, res = wavenumber_grid(22920, 23000, 1000, unit='AA')
    mdbCO = moldb.MdbExomol('.database/CO/12C-16O/Li2015', nu_grid, crit=1.e-46)
    ts = time.time()
    numatrix = make_numatrix0(nu_grid, mdbCO.nu_lines)
    print(np.median(numatrix))
    te = time.time()
    print(te - ts, 'sec')

    ts = time.time()
    numatrixc = make_numatrix0_device(nu_grid, mdbCO.nu_lines)
    print(np.median(numatrix))
    te = time.time()
    print(te - ts, 'sec')

    print(np.sum((numatrixc - numatrix)**2))
