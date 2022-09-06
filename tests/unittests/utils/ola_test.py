"""Overlap-add convolve
"""
from scipy.signal import oaconvolve
import numpy as np
import matplotlib.pyplot as plt
from exojax.signal import np_olaconv

def test_np_olaconv():
    fig = False
    np.random.seed(1)
    Nx = 100000
    x = np.zeros(Nx)
    Npulse = 50
    x[np.random.choice(list(range(Nx)), Npulse)] = 1.0
    Nf = 301  #odd
    g = np.linspace(-3, 3, Nf)
    f = np.exp(-g * g / 2.0) / np.sqrt(2 * np.pi)  #FIR filter

    oac = oaconvolve(x, f)  # length = Nx + M -1
    ola = np_olaconv(x, f, 100)
    maxresidual = np.max(np.sqrt((oac - ola)**2))
    assert maxresidual < 1.e-15
    if fig:
        edge = int((Nf - 1) / 2)
        plt.plot(x, label="input")
        plt.plot(oac[edge:-edge], label="oaconvolve")
        plt.plot(ola[edge:-edge], ls="dashed", label="OLA test")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    test_np_olaconv()