"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
import jax.numpy as jnp
from exojax.spec.lsd import npgetix, npadd2D, npadd3D_uniqidx
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
    """numpy version of getix weigthed by exp(-hc/kT).

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


def make_lbd2D(Sij0, nu_lines, nu_grid, elower, elower_grid, Ttyp):
    """make logarithm biased LSD (LBD) array (2D)

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

def make_lbd3D_uniqidx(Sij0, nu_lines, nu_grid, elower, elower_grid, uidx_broadpar, Ttyp):
    """make logarithm biased LSD (LBD) array (2D)

    Args:
        Sij0: line strength at the refrence temepreature Tref (should be F64)
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)
        elower: E lower
        elower_grid: E lower grid
        uidx_broadpar: broadening parameter index
        Ttyp: typical temperature you will use.

    Returns:
        lbd

    Notes: 
        LBD (jnp array)

    """
    logmin=-np.inf
    lsd = np.zeros((len(nu_grid), np.max(uidx_broadpar)+1, len(elower_grid)),dtype=np.float64)
    cx, ix = npgetix(nu_lines, nu_grid)
    cy, iy = npgetix_exp(elower, elower_grid, Ttyp)
    lsd=npadd3D_uniqidx(lsd, Sij0, cx, ix, cy, iy, uidx_broadpar)
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
        lbd_biased: log biased LSD
        T: temperature for unbiasing
        nu_grid: wavenumber grid
        elower_grid: Elower grid
        qr: partition function ratio Q(T)/Q(Tref)

    Returns:
        LSD (unbiased)

    """
    Nnu=int(len(nu_grid)/2)
    eunbias_lbd = jnp.sum(jnp.exp(logf_bias(elower_grid,T)+lbd_biased),axis=1)
    return (eunbias_lbd.T*g_bias(nu_grid,T)/qr(T)).T


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

def compare_with_direct3d(mdb,Ttest=1000.0,interval_contrast=0.1,Ttyp=2000.0):
    """ compare the premodit LSD with the direct computation of LSD

    """
    from exojax.spec.lsd import npadd1D, npgetix, uniqidx_2D
    from exojax.spec.hitran import SijT

    broadpar=np.array([mdb._n_Texp,mdb._alpha_ref]).T
    uidx_broadpar=uniqidx_2D(broadpar)
    elower_grid=make_elower_grid(Ttyp, mdb._elower, interval_contrast=interval_contrast)
    lbd=make_lbd3D_uniqidx(mdb.Sij0, mdb.nu_lines, nus, mdb._elower, elower_grid, uidx_broadpar, Ttyp)
    Slsd=unbiased_lsd(lbd,Ttest,nus,elower_grid,mdb.qr_interp)

    
    cont_inilsd_nu, index_inilsd_nu = npgetix(mdb.nu_lines, nus)
    qT = mdb.qr_interp(Ttest)
    logsij0 = jnp.array(np.log(mdb.Sij0))
    S=SijT(Ttest, logsij0, mdb.nu_lines, mdb._elower, qT)
    Slsd_direct = np.zeros_like(nus,dtype=np.float64)
    Slsd_direct = npadd1D(Slsd_direct, S, cont_inilsd_nu, index_inilsd_nu)
    print("Number of the E_lower grid=",len(elower_grid))
    print("max deviation=",np.max(np.abs(Slsd/Slsd_direct-1.0)))
    return Slsd, Slsd_direct


if __name__ == "__main__":
    import jax.numpy as jnp
    from exojax.spec import moldb
    import matplotlib.pyplot as plt
    print("premodit")

    nus=np.logspace(np.log10(6020.0), np.log10(6080.0), 40000, dtype=np.float64)
    mdb = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nus, gpu_transfer=False)

    from exojax.spec.lsd import uniqidx_2D
    broadpar=np.array([mdb._n_Texp,mdb._alpha_ref]).T
    uidx_broadpar=uniqidx_2D(broadpar)

    Ttyp=2000.0
    interval_contrast=0.1
    elower_grid=make_elower_grid(Ttyp, mdb._elower, interval_contrast=interval_contrast)
    lbd=make_lbd3D_uniqidx(mdb.Sij0, mdb.nu_lines, nus, mdb._elower, elower_grid, uidx_broadpar, Ttyp)
    Ttest=1000.0

    print(np.shape(lbd))
    Slsd=unbiased_lsd(lbd,Ttest,nus,elower_grid,mdb.qr_interp)
    print(np.shape(Slsd))
    import sys
    sys.exit()

    
    Slsd,Slsd_direct=compare_with_direct2D(mdb,Ttest=1000.0,interval_contrast=0.1,Ttyp=2000.0)    

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


    
    #fftval=lowpass(fftval,compress_rate=40)
    #Slsd=unbiased_lsd_lowpass(fftval,Ttest,nus,elower_grid,mdbCH4.qr_interp)
    #val = np.fft.rfft(initial_biased_lsd, axis=0)
    #Slsd=unbiased_lsd_fft(val,Ttest,nus,elower_grid,mdbCH4.qr_interp)*1.e-26

    #k = np.fft.rfftfreq(2*Ng_nu, 1)    
    #    vk = fold_voigt_kernel_logst(
    #        k, log_nstbeta, log_ngammaL_grid, vmax, pmarray)
    #    fftvalsum = jnp.sum(fftval*vk, axis=(1,))
    #    xs = jnp.fft.irfft(fftvalsum)[:Ng_nu]*R/nus
