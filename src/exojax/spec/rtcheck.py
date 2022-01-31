"""Check tool for radiative transfer."""

__all__ = ['CheckRT']
from jax import jit
import jax.numpy as jnp
import numpy as np


class CheckRT(object):
    """Jax Radiative Transfer class."""

    def __init__(self):
        self.xhamin = []
        self.xhamax = []
        self.aha = []

    def check_hjert(self, numatrix, sigmaD, gammaL):
        """cheking ranges of x and a in Voigt-Hjerting function H(x,a)"""
        a, xhmin, xhmax = compminmax(numatrix, sigmaD, gammaL)
        if len(self.xhamin) == 0:
            self.xhamin = np.copy(xhmin)
        else:
            self.xhamin = np.min([self.xhamin, xhmin], axis=0)
        if len(self.xhamax) == 0:
            self.xhamax = np.copy(xhmax)
        else:
            self.xhamax = np.max([self.xhamax, xhmax], axis=0)
            self.aha = np.copy(a)

    def plotxa(self):
        import matplotlib.pyplot as plt
        plt.plot(self.xhamax, self.aha, '.', alpha=0.1, color='C1')
        plt.plot(self.xhamin, self.aha, '.', alpha=0.1, color='C0')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('x')
        plt.ylabel('a')


@jit
def compminmax(numatrix, sigmaD, gammaL):
    minnu = jnp.min(jnp.abs(numatrix), axis=1)
    maxnu = jnp.max(jnp.abs(numatrix), axis=1)
    sfac = 1.0/(np.sqrt(2)*sigmaD)
    xmin = sfac*minnu
    xmax = sfac*maxnu
    a = sfac*gammaL
    return a, xmin, xmax
