"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
from exojax.spec.lsd import npgetix
from exojax.utils.constants import hcperk

def make_initial_LSD(nu_grid, nu_lines, Tmax, elower, interval_contrast_lsd=1.0):
    """make initial LSD to compute the power spectrum of the LSD

    Args:
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        Tmax: max temperature you will use.
        elower: E lower
        interval_contrast_lsd: interval contrast of line strength between upper and lower E lower grid

    Returns:
        contribution nu
        index nu
        contribution E lower
        index E lower

    Notes: 
        initial LSD is used to compute the power spectrum of LSD. So, nu_grind should have enough resolution.

    """
    elower_grid=make_elower_grid(Tmax, elower, interval_contrast=interval_contrast_lsd)
    cont_inilsd_elower, index_inilsd_elower = npgetix(elower, elower_grid)
    cont_inilsd_nu, index_inilsd_nu = npgetix(nu_lines, nu_grid)
    return cont_inilsd_nu, index_inilsd_nu, cont_inilsd_elower, index_inilsd_elower, elower_grid



def compute_dElower(T,interval_contrast=0.1):
    """ compute a grid interval of Elower given the grid interval of line strength

    Args: 
        T: temperature in Kelvin
        interval_contrast: putting c = grid_interval_line_strength, then, the contrast of line strength between the upper and lower of the grid becomes c-order of magnitude.

    Returns:
        Required dElower
    """
    return interval_contrast*np.log(10.0)*T/hcperk

    
def make_elower_grid(Tmax, elower, interval_contrast):
    """compute E_lower grid given interval_contrast of line strength

    Args: 
        Tmax: max temperature in Kelvin
        elower: E_lower
        interval_contrast: putting c = grid_interval_line_strength, then, the contrast of line strength between the upper and lower of the grid becomes c-order of magnitude.

    Returns:
       grid of E_lower given interval_contrast

    """
    dE = compute_dElower(Tmax,interval_contrast)
    min_elower=np.min(elower)
    max_elower=np.max(elower)
    Ng_elower = int((max_elower - min_elower)/dE)+2
    return min_elower + np.arange(Ng_elower)*dE


from jax.numpy import index_exp as joi

def inc2D_initlsd(a, w, cx, ix, cy, iy):
    a = a.at[joi[ix, iy]].add(w*(1-cx)*(1-cy))
    a = a.at[joi[ix+1, iy]].add(w*cx*(1-cy))
    a = a.at[joi[ix+1, iy+1]].add(w*cx*cy)
    a = a.at[joi[ix, iy+1]].add(w*(1-cx)*cy)
    return a


def test_determine_initial_nugrid():
    print("test")

def lowpass(fftval,compress_rate):
    Nnu,Nbatch=np.shape(fftval)
    Ncut=int(float(Nnu)/float(compress_rate))
    lowpassed_fftval=fftval[:Ncut,:]
    high_freq_norm_squared=np.sum(np.abs(fftval[Ncut:,:])**2,axis=0)
    lowpassed_fftval[0,:]=np.sqrt(np.abs(lowpassed_fftval[0,:])**2+high_freq_norm_squared)
    return lowpassed_fftval
    
if __name__ == "__main__":
    import jax.numpy as jnp
    from exojax.spec import moldb
    print("premodit")
    print(compute_dElower(1000.0,interval_contrast=1.0))

    nus=np.logspace(np.log10(6020.0), np.log10(6080.0), 40000, dtype=np.float64)
    mdbCH4 = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nus)
    Tmax=1500.0
    cont_inilsd_nu, index_inilsd_nu, cont_inilsd_elower, index_inilsd_elower, elower_grid=make_initial_LSD(nus, mdbCH4.nu_lines, Tmax, mdbCH4.elower, interval_contrast_lsd=0.05)
    print(elower_grid)
    
    Ng_nu = len(nus)
    Ng_elower = len(elower_grid)
    k = jnp.fft.rfftfreq(2*Ng_nu, 1)
    lsd_array = jnp.zeros((Ng_nu,Ng_elower))
    Slsd=inc2D_initlsd(lsd_array, mdbCH4.Sij0, cont_inilsd_nu, index_inilsd_nu, cont_inilsd_elower, index_inilsd_elower)
    import matplotlib.pyplot as plt

    
    
    fftval = np.fft.rfft(Slsd, axis=0)
    fftval_lowpassed=lowpass(fftval,compress_rate=40)

    fftval=np.zeros_like(fftval,dtype=np.complex128)
    Ncut=np.shape(fftval_lowpassed)[0]
    fftval[:Ncut,:]=fftval_lowpassed
    
    Slsdx = np.fft.irfft(fftval, axis=0)
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #    for i in range(0,Ng_elower):
    plt.plot((Slsd[:,60]),alpha=0.3)
    plt.plot((Slsdx[:,60]),alpha=0.3)
    plt.yscale("log")
    plt.show()
    
    vmax = Ng_nu
    #    vk = fold_voigt_kernel_logst(
    #        k, log_nstbeta, log_ngammaL_grid, vmax, pmarray)
    #    fftvalsum = jnp.sum(fftval*vk, axis=(1,))
    #    xs = jnp.fft.irfft(fftvalsum)[:Ng_nu]*R/nus
    
