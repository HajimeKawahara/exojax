import numpy as np

def np_olaconv(x, f, ndiv):
    """Overlap and Add convolve (numpy version)

    Args:
        x (ndarray): long real vector
        f (ndarray): real FIR filter, length should be odd
        ndiv (int): number of division of x

    Returns:
        convolved vector w/ length of (len(x) + len(f) - 1)
    """
    Nx = len(x)
    L = int(Nx / ndiv)
    M = len(f)
    N = L + M - 1

    xarr = x.reshape((ndiv, L))
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
