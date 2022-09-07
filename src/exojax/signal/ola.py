import numpy as np
import jax.numpy as jnp
from jax.lax import scan
from jax.numpy import index_exp
from jax.lax import dynamic_update_slice
from jax import jit

@jit
def olaconv(xarr, f):
    """Overlap and Add convolve (jax.numpy version)

    Args:
        xarr (jax/ndarray): reshaped matrix of a long real vector (ndiv, L)
        f (jax/ndarray): real FIR filter, length should be odd
        
    Returns:
        convolved vector w/ length of (len(x) + len(f) - 1)
    """
    ndiv, L = jnp.shape(xarr)
    M = len(f)
    N = L + M - 1
    Nx = ndiv * L
    xzeropad = jnp.zeros((ndiv, N))
    xzeropad = xzeropad.at[index_exp[:, 0:L]].add(xarr)
    fzeropad = jnp.zeros(N)
    fzeropad = fzeropad.at[index_exp[0:M]].add(f)
    ftilde = jnp.fft.rfft(fzeropad)
    xtilde = jnp.fft.rfft(xzeropad, axis=1)
    ytilde = xtilde * ftilde[jnp.newaxis, :]
    ftarr = jnp.fft.irfft(ytilde, axis=1)

    def f(idiv, ft):
        idiv = idiv + 1
        return idiv, dynamic_update_slice(jnp.zeros(Nx + M - 1), ft,
                                          ((idiv - 1) * L, ))

    nscan, fftvalarr = scan(f, 0, ftarr)
    return jnp.sum(fftvalarr, axis=0)


def np_olaconv(xarr, f):
    """Overlap and Add convolve (numpy version)

    Args:
        xarr (ndarray): reshaped matrix of a long real vector (ndiv, L)
        f (ndarray): real FIR filter, length should be odd
        
    Returns:
        convolved vector w/ length of (len(x) + len(f) - 1)
    """
    ndiv, L = np.shape(xarr)
    M = len(f)
    N = L + M - 1
    Nx = ndiv * L
    xzeropad = np.zeros((ndiv, N))
    xzeropad[:, 0:L] = xarr
    fzeropad = np.zeros(N)
    fzeropad[0:M] = f
    ftilde = np.fft.rfft(fzeropad)
    xtilde = np.fft.rfft(xzeropad, axis=1)
    ytilde = xtilde * ftilde[np.newaxis, :]
    ftarr = np.fft.irfft(ytilde, axis=1)
    y = np.zeros(Nx + M - 1)
    for idiv in range(ndiv):
        y[int(idiv * L):int(idiv * L + N)] += ftarr[idiv, :]
    return y
