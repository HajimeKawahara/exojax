"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
from exojax.spec.lsd import npgetix, npadd2D, npadd1D
from exojax.utils.constants import hcperk, Tref

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


def npgetix_exp(x, xv, Ttyp):
    """numpy version of getix.

    Args:
        x: x array
        xv: x grid
        Ttyp: typical temperature for the temperature correction

    Returns:
        cont (contribution)
        index (index)

    Note:
       cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.
    """

    if Ttyp is not None:
        x=np.exp(-hcperk*x*(1.0/Ttyp-1.0/Tref))
        xv=np.exp(-hcperk*xv*(1.0/Ttyp-1.0/Tref))
    
    indarr = np.arange(len(xv))
    pos = np.interp(x, xv, indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont, index    


def make_LBD(Sij0, nu_lines, nu_grid, elower, elower_grid, Ttyp):
    """make logarithm biased LSD (LBD) array

    Args:
        Sij0: line strength at the refrence temepreature Tref (should be F64)
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)
        elower: E lower
        elower_grid: E lower grid
        Ttyp: typical temperature you will use.

    Returns:
        lbd

    Notes: 
        LBD (jnp array)

    """
    logmin=-np.inf
    lsd = np.zeros((len(nu_grid), len(elower_grid)),dtype=np.float64)

    cx, ix = npgetix(nu_lines, nu_grid)
    cy, iy = npgetix_exp(elower, elower_grid, Ttyp)
    lsd=npadd2D(lsd, Sij0, cx, ix, cy, iy)
    
    lsd[lsd>0.0]=np.log(lsd[lsd>0.0])
    lsd[lsd==0.0]=logmin       
    return jnp.array(lsd)

def logf_bias(elower_in,T):
    """logarithm f bias function
    """
    return -hcperk*elower_in * (1./T - 1./Tref)

def g_bias(nu_in,T):
    """g bias function
    """
    #return jnp.expm1(-hcperk*nu_in/T) / jnp.expm1(-hcperk*nu_in/Tref)
    return  (1.0-jnp.exp(-hcperk*nu_in/T)) / (1.0-jnp.exp(-hcperk*nu_in/Tref))

def unbiased_lsd(lbd_biased,T,nu_grid,elower_grid,qr):
    """ unbias the biased LSD

    Args:

    Returns:
        LSD (unbiased)

    """
    Nnu=int(len(nu_grid)/2)
    eunbias_lbd = jnp.sum(jnp.exp(logf_bias(elower_grid,T)+lbd_biased),axis=1)
#    eunbias_Slsd = np.fft.irfft(eunbias_FTSlsd)
#    return g_bias(nu_grid,T)*eunbias_Slsd/qr(T)
    return g_bias(nu_grid,T)*eunbias_lbd/qr(T)


def lowpass(fftval,compress_rate):
    """lowpass filter for the biased LSD

    """
    Nnu,Nbatch=np.shape(fftval)
    Ncut=int(float(Nnu)/float(compress_rate))
    lowpassed_fftval=fftval[:Ncut,:]
    high_freq_norm_squared=np.sum(np.abs(fftval[Ncut:,:])**2,axis=0)
    lowpassed_fftval[0,:]=np.sqrt(np.abs(lowpassed_fftval[0,:])**2+high_freq_norm_squared)
    return lowpassed_fftval




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



if __name__ == "__main__":
    import jax.numpy as jnp
    from exojax.spec import moldb
    import matplotlib.pyplot as plt
    print("premodit")

    nus=np.logspace(np.log10(6020.0), np.log10(6080.0), 40000, dtype=np.float64)
    mdbCH4 = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nus)
    Tmax=1500.0
    i=0
#    j=1000000
#    n=j*100
#    j=1000
#    n=j*1000

    n=len(mdbCH4.nu_lines)
    j=1

    Ttest=1500.0
    #fftval=lowpass(fftval,compress_rate=40)
    #Slsd=unbiased_lsd_lowpass(fftval,Ttest,nus,elower_grid,mdbCH4.qr_interp)
    #val = np.fft.rfft(initial_biased_lsd, axis=0)

    #Slsd=unbiased_lsd_fft(val,Ttest,nus,elower_grid,mdbCH4.qr_interp)*1.e-26
    elower_grid=make_elower_grid(Ttest, mdbCH4.elower[i:n:j], interval_contrast=0.01)
    lbd=make_LBD(mdbCH4.Sij0[i:n:j], mdbCH4.nu_lines[i:n:j], nus, mdbCH4.elower[i:n:j], elower_grid, Ttest)
    Slsd=unbiased_lsd(lbd,Ttest,nus,elower_grid,mdbCH4.qr_interp)
    
    #DIRECT COMPUTATION of LSD
    from exojax.spec.hitran import SijT
    cont_inilsd_nu, index_inilsd_nu = npgetix(mdbCH4.nu_lines[i:n:j], nus)
    qT = mdbCH4.qr_interp(Ttest)
    S=SijT(Ttest, mdbCH4.logsij0[i:n:j], mdbCH4.nu_lines[i:n:j], mdbCH4.elower[i:n:j], qT)
    Slsd_direct = np.zeros_like(nus,dtype=np.float64)
    Slsd_direct = npadd1D(Slsd_direct, S, cont_inilsd_nu, index_inilsd_nu)
    print(np.mean(Slsd/Slsd_direct-1.0))


    ### just checking
    fig=plt.figure()
    ax=fig.add_subplot(211)
    plt.plot((Slsd),alpha=0.3)
    plt.plot((Slsd_direct),alpha=0.3)
    plt.yscale("log")
    ax=fig.add_subplot(212)
    plt.plot((Slsd/Slsd_direct-1.0),Slsd,".",alpha=0.3)
    plt.xlabel("error (premodit - direct)/direct")
    plt.yscale("log")
    plt.show()
    ###



    #k = np.fft.rfftfreq(2*Ng_nu, 1)
    
    #    vk = fold_voigt_kernel_logst(
    #        k, log_nstbeta, log_ngammaL_grid, vmax, pmarray)
    #    fftvalsum = jnp.sum(fftval*vk, axis=(1,))
    #    xs = jnp.fft.irfft(fftvalsum)[:Ng_nu]*R/nus
    
