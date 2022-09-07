import numpy as np

def olaconv(xarr, f, ndiv):
    return
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
    Nx = ndiv*L
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
