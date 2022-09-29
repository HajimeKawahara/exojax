"""Overlap-add convolve
"""
from scipy.signal import oaconvolve
import numpy as np
import matplotlib.pyplot as plt
from exojax.signal.ola import olaconv, ola_lengths, apply_zeropad
from exojax.signal.ola import np_olaconv
from exojax.signal.ola import optimal_div_length
import jax.numpy as jnp

from jax.config import config

config.update('jax_enable_x64', True)

def test_optimal_div_length(fig=False):
    filter_length_arr = np.logspace(1, 6, 60)
    optimal_div_length_arr = []
    for fl in filter_length_arr:
        div_length = optimal_div_length(int(fl))
        optimal_div_length_arr.append(div_length)
    optimal_div_length_arr = np.array(optimal_div_length_arr)
    assert optimal_div_length_arr[0] == 54
    assert optimal_div_length_arr[-1] == 18432000
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(filter_length_arr, optimal_div_length_arr, label="optimal")
        plt.plot([10.0, 1.e6], [10.0, 1.e6],
                 color="gray",
                 lw=0.5,
                 label="filter = block_size")
        plt.plot([10.0, 1.e6], [100.0, 1.e7],
                 color="gray",
                 lw=0.5,
                 ls="dashed",
                 label="filter x10= block_size")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("optimal div length")
        plt.xlabel("filter length")
        plt.legend()
        plt.savefig("optimal_block_size.png")
        plt.show()
    return


def test_olaconv(fig=False):
    np.random.seed(1)
    Nx = 100000
    x = np.zeros(Nx)
    Npulse = 50
    x[np.random.choice(list(range(Nx)), Npulse)] = 1.0
    Nf = 301  #odd
    g = np.linspace(-3, 3, Nf)
    f = jnp.array(np.exp(-g * g / 2.0) / np.sqrt(2 * np.pi))  #FIR filter

    oac = oaconvolve(x, f)  # length = Nx + M -1
    ndiv = 100
    xarr = jnp.array(x.reshape(ndiv, int(Nx / ndiv)))
    ndiv, div_length, filter_length = ola_lengths(xarr, f)
    xarr_hat, f_hat = apply_zeropad(xarr, f, ndiv, div_length, filter_length)
    ola = olaconv(xarr_hat, f_hat, ndiv, div_length, filter_length)
    maxresidual = np.max(np.sqrt((oac - ola)**2))
    assert maxresidual < 1.e-9  #fp64
    #assert maxresidual < 1.e-6 #fp32

    if fig:
        edge = int((Nf - 1) / 2)
        plt.plot(x, label="input")
        plt.plot(oac[edge:-edge], label="oaconvolve")
        plt.plot(ola[edge:-edge], ls="dashed", label="OLA test")
        plt.legend()
        plt.show()


def test_np_olaconv(fig=False):
    np.random.seed(1)
    Nx = 100000
    x = np.zeros(Nx)

    Npulse = 50
    x[np.random.choice(list(range(Nx)), Npulse)] = 1.0
    Nf = 301  #odd
    g = np.linspace(-3, 3, Nf)
    f = np.exp(-g * g / 2.0) / np.sqrt(2 * np.pi)  #FIR filter

    oac = oaconvolve(x, f)  # length = Nx + M -1
    ndiv = 100
    xarr = x.reshape(ndiv, int(Nx / ndiv))
    ola = np_olaconv(xarr, f)
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
    test_optimal_div_length()
    test_olaconv(fig=True)
    #test_np_olaconv(fig=True)
