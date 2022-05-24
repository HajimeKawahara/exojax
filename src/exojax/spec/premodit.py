"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from exojax.spec.lsd import npgetix, npgetix_exp, npadd2D, npadd3D_uniqidx
from exojax.utils.constants import hcperk, Tref
from exojax.spec.ditkernel import fold_voigt_kernel_logst
from exojax.spec.modit import calc_xsection_from_lsd
from exojax.spec.exomol import gamma_exomol

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

def make_lbd3D_uniqidx(Sij0, cont_nu, index_nu, Ng_nu, elower, elower_grid, uidx_broadpar, Ttyp):
    """make logarithm biased LSD (LBD) array (2D)

    Args:
        Sij0: line strength at the refrence temepreature Tref (should be F64)
        cont_nu: (wavenumber contribution) jnp.array
        index_nu: (wavenumber index) jnp.array
        Ng_nu: number of wavenumber bins (len(nu_grid))
        elower: E lower
        elower_grid: E lower grid
        uidx_broadpar: broadening parameter index
        Ttyp: typical temperature you will use.

    Returns:
        lbd

    Notes: 
        LBD (jnp array), contribution fow wavenumber, index for wavenumber

    """
    logmin=-np.inf
    lsd = np.zeros((Ng_nu, np.max(uidx_broadpar)+1, len(elower_grid)),dtype=np.float64)
    cy, iy = npgetix_exp(elower, elower_grid, Ttyp)
    lsd=npadd3D_uniqidx(lsd, Sij0, cont_nu, index_nu, cy, iy, uidx_broadpar)
    lsd[lsd>0.0]=np.log(lsd[lsd>0.0])
    lsd[lsd==0.0]=logmin       
    return jnp.array(lsd)


def logf_bias(elower_in,T):
    """logarithm f bias function
    
    Args:
        elower_in: Elower in cm-1
        T: temperature for unbiasing in Kelvin

    Returns:
        logarithm of bias f function

    """
    return -hcperk*elower_in * (1./T - 1./Tref)

def g_bias(nu_in,T):
    """g bias function

    Args:
        nu_in: wavenumber in cm-1
        T: temperature for unbiasing in Kelvin

    Returns:
        bias g function

    """
    #return jnp.expm1(-hcperk*nu_in/T) / jnp.expm1(-hcperk*nu_in/Tref)
    return  (1.0-jnp.exp(-hcperk*nu_in/T)) / (1.0-jnp.exp(-hcperk*nu_in/Tref))

def unbiased_lsd(lbd_biased,T,nu_grid,elower_grid,qr):
    """ unbias the biased LSD

    Args:
        lbd_biased: log biased LSD
        T: temperature for unbiasing in Kelvin
        nu_grid: wavenumber grid in cm-1
        elower_grid: Elower grid in cm-1
        qr: partition function ratio Q(T)/Q(Tref)

    Returns:
        Unbiased 2D LSD, shape = (number_of_wavenumber_bin, number_of_broadening_parameters)

    """
    Nnu=int(len(nu_grid)/2)
    eunbias_lbd = jnp.sum(jnp.exp(logf_bias(elower_grid,T)+lbd_biased),axis=-1)
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


def compare_cross_section(mdb,Ttest=1000.0,interval_contrast=0.1,Ttyp=2000.0):
    """ compare the premodit cross section with the direct computation of LSD

    """
    from exojax.spec.lsd import npadd1D, npgetix, uniqidx_2D
    from exojax.spec.hitran import SijT

    broadpar=np.array([mdb._n_Texp,mdb._alpha_ref]).T
    uidx_broadpar, uniq_broadpar=uniqidx_2D(broadpar)
    elower_grid=make_elower_grid(Ttyp, mdb._elower, interval_contrast=interval_contrast)
    cont_nu, index_nu = npgetix(nus, nu_grid)
    lbd=make_lbd3D_uniqidx(mdb.Sij0, cont_nu, index_nu, len(nus), mdb._elower, elower_grid, uidx_broadpar, Ttyp)
    Slsd=unbiased_lsd(lbd,Ttest,nus,elower_grid,mdb.qr_interp)
    Slsd=np.sum(Slsd,axis=1)
    
#    cont_inilsd_nu, index_inilsd_nu = npgetix(mdb.nu_lines, nus)
#    qT = mdb.qr_interp(Ttest)
#    logsij0 = jnp.array(np.log(mdb.Sij0))
#    S=SijT(Ttest, logsij0, mdb.nu_lines, mdb._elower, qT)
#    Slsd_direct = np.zeros_like(nus,dtype=np.float64)
#    Slsd_direct = npadd1D(Slsd_direct, S, cont_inilsd_nu, index_inilsd_nu)
#    print("Number of the E_lower grid=",len(elower_grid))
    print("max deviation=",np.max(np.abs(Slsd/Slsd_direct-1.0)))
    return Slsd, Slsd_direct

def exomol(mdb, lbd, uniq_broadpar, Tarr, Parr, R, molmass):
    """compute molecular line information required for PreMODIT using Exomol mdb.

    Args:
       mdb: mdb instance
       lbd: log biased LSD
       Tarr: Temperature array
       Parr: Pressure array
       R: spectral resolution
       molmass: molecular mass

    Returns:
       Slsd: line shape density
       ngammaLM: normalized gammaL matrix,
       nsigmaDl: normalized sigmaD matrix
    """
    qt = vmap(mdb.qr_interp)(Tarr)
    gammaLM = jit(vmap(gamma_exomol, (0, 0, None, None)))(
        Parr, Tarr, uniq_braodpar[0], uniq_braodpar[1])
    
    #Not include natural width yet    
    #gammaLMN = gamma_natural(mdb._A)
    #gammaLM = gammaLMP+gammaLMN[None, :]
    ngammaLM = gammaLM/(mdb.nu_lines/R)

    nsigmaDl = normalized_doppler_sigma(Tarr, molmass, R)[:, jnp.newaxis]
    elower_grid=make_elower_grid(Ttyp, mdb._elower, interval_contrast=interval_contrast)
    Slsd=unbiased_lsd(lbd,Ttest,nus,elower_grid,qt)

    return Slsd, ngammaLM, nsigmaDl

@jit
def xsvector(T, lbd, R, pmarray, nu_grid, broadpar, uidx_broadpar, elower_grid, log_ngammaL_grid):
    """Cross section vector (PreMODIT)

    The original code is rundit_fold_logredst in `addit package <https://github.com/HajimeKawahara/addit>`_ ). MODIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant normalized beta

    Args:
       R: spectral resolution
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       nsigmaD: normaized Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid

    Returns:
       Cross section in the linear nu grid
    """

    xs = calc_xsection_from_lsd(Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid)

    return xs

if __name__ == "__main__":
    import jax.numpy as jnp
    from exojax.spec import moldb
    import matplotlib.pyplot as plt
    from exojax.spec.lsd import npadd1D, npgetix, uniqidx_2D
    from exojax.spec.hitran import SijT
    from exojax.spec.initspec import init_premodit
    from exojax.spec.hitran import normalized_doppler_sigma
    from exojax.spec import molinfo
    
    nus=np.logspace(np.log10(6020.0), np.log10(6080.0), 40000, dtype=np.float64)
    mdb = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nus, gpu_transfer=False)

    Ttyp=1500.0
    broadpar=np.array([mdb._n_Texp,mdb._alpha_ref]).T
    cont_nu, index_nu, elower_grid, uidx_broadpar, uniq_broadpar, R, pmarray=init_premodit(mdb.nu_lines, nus, mdb._elower, Ttyp=Ttyp, broadpar=broadpar)

    lbd=make_lbd3D_uniqidx(mdb.Sij0, cont_nu, index_nu, len(nus), mdb._elower, elower_grid, uidx_broadpar, Ttyp)

    Tfix = 1000.0
    Pfix = 1.e-3
    molmassCH4=molinfo.molmass("CH4")


    nsigmaD = normalized_doppler_sigma(Tfix, molmassCH4, R)
    qt = mdb.qr_interp(Tfix)
    print( uniq_broadpar[:,0], uniq_broadpar[:,1], "ui")
    gammaL = gamma_exomol(Pfix, Tfix, uniq_broadpar[:,0], uniq_broadpar[:,1])
    print(gammaL,"gma")
#    log_ngammaL_grid = jnp.log(gammaL/(/R))
#    elower_grid=make_elower_grid(Ttyp, mdb._elower, interval_contrast=interval_contrast)
#    Slsd=unbiased_lsd(lbd,Tfix,nus,elower_grid,qt)
#    xs = calc_xsection_from_lsd(Slsd, R, pmarray, nsigmaD, nus, log_ngammaL_grid)
