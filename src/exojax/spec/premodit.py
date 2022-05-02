"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
from exojax.spec.lsd import npgetix
from exojax.utils.constants import hcperk, Tref



def make_initial_biased_LSD(nu_grid, nu_lines, Tmax, elower, interval_contrast_lsd=1.0):
    """make initial biased LSD array to compute the power spectrum of the LSD

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
    """

    """
    a = a.at[joi[ix, iy]].add(w*(1-cx)*(1-cy))
    a = a.at[joi[ix+1, iy]].add(w*cx*(1-cy))
    a = a.at[joi[ix+1, iy+1]].add(w*cx*cy)
    a = a.at[joi[ix, iy+1]].add(w*(1-cx)*cy)
    return a


def lowpass(fftval,compress_rate):
    """lowpass filter for the biased LSD

    """
    Nnu,Nbatch=np.shape(fftval)
    Ncut=int(float(Nnu)/float(compress_rate))
    lowpassed_fftval=fftval[:Ncut,:]
    high_freq_norm_squared=np.sum(np.abs(fftval[Ncut:,:])**2,axis=0)
    lowpassed_fftval[0,:]=np.sqrt(np.abs(lowpassed_fftval[0,:])**2+high_freq_norm_squared)
    return lowpassed_fftval

def f_bias(elower_in,T):
    """f bias function
    """
    return jnp.exp(-hcperk*elower_in * (1./T - 1./Tref))

def g_bias(nu_in,T):
    """g bias function
    """
    return jnp.expm1(-hcperk*nu_in/T) / jnp.expm1(-hcperk*nu_in/Tref)


def unbiased_lsd_lowpass(FT_Slsd_biased,T,nu_grid,elower_grid, qr):
    """ unbias the biased LSD lowpass filtered

    Args:

    Returns:
        LSD (unbiased)

    """
    Nnu=int(len(nu_grid)/2)
    eunbias_FTSlsd = jnp.sum(f_bias(elower_grid,T)*FT_Slsd_biased,axis=1)
    Nft=len(eunbias_FTSlsd)
    eunbias_FTSbuf = jnp.hstack([eunbias_FTSlsd, jnp.zeros(Nnu-Nft+1)])
    eunbias_Slsd = jnp.fft.irfft(eunbias_FTSbuf)
    return g_bias(nu_grid,T)*eunbias_Slsd/qr(T)

def unbiased_lsd(FT_Slsd_biased,T,nu_grid,elower_grid,qr):
    """ unbias the biased LSD

    Args:

    Returns:
        LSD (unbiased)

    """
    Nnu=int(len(nu_grid)/2)
    eunbias_FTSlsd = jnp.sum(f_bias(elower_grid,T)*FT_Slsd_biased,axis=1)
    eunbias_Slsd = jnp.fft.irfft(eunbias_FTSlsd)
    return g_bias(nu_grid,T)*eunbias_Slsd/qr(T)


if __name__ == "__main__":
    import jax.numpy as jnp
    from exojax.spec import moldb
    import matplotlib.pyplot as plt
    print("premodit")

    nus=np.logspace(np.log10(6020.0), np.log10(6080.0), 40000, dtype=np.float64)
    mdbCH4 = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nus)
    Tmax=1500.0
    cont_inilsd_nu, index_inilsd_nu, cont_inilsd_elower, index_inilsd_elower, elower_grid=make_initial_biased_LSD(nus, mdbCH4.nu_lines, Tmax, mdbCH4.elower, interval_contrast_lsd=0.05)
    Ng_nu = len(nus)
    Ng_elower = len(elower_grid)
    k = jnp.fft.rfftfreq(2*Ng_nu, 1)
    lsd_array = jnp.zeros((Ng_nu,Ng_elower))
    initial_biased_lsd=inc2D_initlsd(lsd_array, mdbCH4.Sij0, cont_inilsd_nu, index_inilsd_nu, cont_inilsd_elower, index_inilsd_elower)

    Ttest=1000.0

    #fftval=lowpass(fftval,compress_rate=40)
    #Slsd=unbiased_lsd_lowpass(fftval,Ttest,nus,elower_grid,mdbCH4.qr_interp)
    fftval = np.fft.rfft(initial_biased_lsd, axis=0)
    Slsd=unbiased_lsd(fftval,Ttest,nus,elower_grid,mdbCH4.qr_interp)

    from exojax.spec.hitran import SijT    
    def inc1D(a, w, cx, ix):
        a = a.at[joi[ix]].add(w*(1-cx))
        a = a.at[joi[ix+1]].add(w*cx)
        return a
    cont_inilsd_nu, index_inilsd_nu = npgetix(mdbCH4.nu_lines, nus)
    qT= mdbCH4.qr_interp(Ttest)
    S=SijT(Ttest, mdbCH4.logsij0, mdbCH4.nu_lines, mdbCH4.elower, qT)
    Slsd_direct = jnp.zeros_like(nus)
    Slsd_direct = inc1D(Slsd_direct, S, cont_inilsd_nu, index_inilsd_nu)
    
    ### just checking
    fig=plt.figure()
    ax=fig.add_subplot(211)
    plt.plot((Slsd),alpha=0.3)
    plt.plot((Slsd_direct),alpha=0.3)
    plt.yscale("log")
    ax=fig.add_subplot(212)
    plt.plot((Slsd/Slsd_direct-1.0),alpha=0.3)
    plt.show()
    ###



    
    #    vk = fold_voigt_kernel_logst(
    #        k, log_nstbeta, log_ngammaL_grid, vmax, pmarray)
    #    fftvalsum = jnp.sum(fftval*vk, axis=(1,))
    #    xs = jnp.fft.irfft(fftvalsum)[:Ng_nu]*R/nus
    
