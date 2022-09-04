"""Overlap-add convolve
"""
from scipy.signal import oaconvolve
import numpy as np
import matplotlib.pyplot as plt

def test_ola():
    fig=True
    np.random.seed(1)
    Nx = 100000
    x = np.zeros(Nx)
    Npulse = 50
    x[np.random.choice(list(range(Nx)),Npulse)] = 1.0
    Nf = 300
    g = np.linspace(-3, 3, Nf)
    f = np.exp(-g*g/2.0)/np.sqrt(2*np.pi) #FIR filter
    
    if fig:
        plt.plot(x)
        plt.plot(oaconvolve(x,f))
        plt.show()
    
if __name__ == "__main__":
    test_ola()