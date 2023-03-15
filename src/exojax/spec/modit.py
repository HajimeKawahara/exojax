"""Line profile computation using Discrete Integral Transform.

* MODIT is Modified version of the `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekerom and E.Pannier. See Section 2.1.2 in `Paper I <https://arxiv.org/abs/2105.14782>`_.
* This module consists of selected functions in `addit package <https://github.com/HajimeKawahara/addit>`_.
* The concept of "folding" can be understood by reading `the discussion <https://github.com/radis/radis/issues/186#issuecomment-764465580>`_ by D.C.M van den Bekerom.
* See also `DIT for non evenly-spaced linear grid <https://github.com/dcmvdbekerom/discrete-integral-transform/blob/master/demo/discrete_integral_transform_log.py>`_ by  D.C.M van den Bekerom, as a reference of this code.
"""
import warnings
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from exojax.spec.ditkernel import fold_voigt_kernel_logst
from exojax.spec.lsd import inc2D_givenx
from exojax.spec.set_ditgrid import minmax_ditgrid_matrix
from exojax.spec.set_ditgrid import precompute_modit_ditgrid_matrix

# exomol
from exojax.spec.exomol import gamma_exomol
from exojax.spec import gamma_natural
from exojax.spec.hitran import line_strength
from exojax.spec import normalized_doppler_sigma

# hitran/hitemp
from exojax.spec.hitran import gamma_hitran

# vald
from exojax.spec.atomll import gamma_vald3, interp_QT284


def calc_xsection_from_lsd(Slsd, R, pmarray, nsigmaD, nu_grid,
                           log_ngammaL_grid):
    """Compute cross section from LSD in MODIT algorithm

    The original code is rundit_fold_logredst in `addit package <https://github.com/HajimeKawahara/addit>`_ ). MODIT folded voigt for ESLOG for reduced wavenumebr inputs (against the truncation error) for a constant normalized beta

    Args:
        Slsd: line shape density
        R: spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaD: normaized Gaussian STD
        nu_grid: linear wavenumber grid
        log_gammaL_grid: logarithm of gammaL grid

    Note: 
    When you have the error such as: 
    "failed to initialize batched cufft plan with customized allocator: 
    Allocating 8000000160 bytes exceeds the memory limit of 4294967296 bytes."
    consider to use moditscanfft.calc_xsection_from_lsd, instead.
    
    Returns:
        Cross section in the log nu grid
    """

    Sbuf = jnp.vstack([Slsd, jnp.zeros_like(Slsd)])
    fftval = jnp.fft.rfft(Sbuf, axis=0)
    Ng_nu = len(nu_grid)
    # -----------------------------------------------
    # MODIT w/o new folding
    # til_Voigt=voigt_kernel_logst(k, log_nstbeta,log_ngammaL_grid)
    # til_Slsd = jnp.fft.rfft(Sbuf,axis=0)
    # fftvalsum = jnp.sum(til_Slsd*til_Voigt,axis=(1,))
    # return jnp.fft.irfft(fftvalsum)[:Ng_nu]*R/nu_grid
    # -----------------------------------------------
    vk = fold_voigt_kernel_logst(jnp.fft.rfftfreq(2 * Ng_nu, 1),
                                 jnp.log(nsigmaD), log_ngammaL_grid, Ng_nu,
                                 pmarray)
    fftvalsum = jnp.sum(fftval * vk, axis=(1, ))
    return jnp.fft.irfft(fftvalsum)[:Ng_nu] * R / nu_grid


@jit
def xsvector(cnu, indexnu, R, pmarray, nsigmaD, ngammaL, S, nu_grid,
             ngammaL_grid):
    """Cross section vector (MODIT)

    Notes:
        Currently due to #277, we recommend to use 
        modit_scanfft.xsvector_scanfft instead of xsvector.
        However, this will be changed when cufft fixes the 4GB limit.

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       nsigmaD: normaized Gaussian STD 
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       gammaL_grid: gammaL grid

    Returns:
       Cross section in the log nu grid
    """

    log_ngammaL_grid = jnp.log(ngammaL_grid)
    lsd_array = jnp.zeros((len(nu_grid), len(ngammaL_grid)))
    Slsd = inc2D_givenx(lsd_array, S, cnu, indexnu, jnp.log(ngammaL),
                        log_ngammaL_grid)
    xs = calc_xsection_from_lsd(Slsd, R, pmarray, nsigmaD, nu_grid,
                                log_ngammaL_grid)
    return xs


@jit
def xsmatrix(cnu, indexnu, R, pmarray, nsigmaDl, ngammaLM, SijM, nu_grid,
             dgm_ngammaL):
    """Cross section matrix for xsvector (MODIT)

    Notes:
        Currently due to #277, we recommend to use 
        modit_scanfft.xsmatrix_scanfft instead of xsmatrix.
        However, this will be changed when cufft fixes the 4GB limit.


    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       R: spectral resolution
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       nu_lines: line center (Nlines)
       nsigmaDl: normalized doppler sigma in layers in R^(Nlayer x 1)
       ngammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_ngammaL: DIT Grid Matrix for normalized gammaL R^(Nlayer, NDITgrid)

    Return:
       cross section matrix in R^(Nlayer x Nwav)
    """
    NDITgrid = jnp.shape(dgm_ngammaL)[1]
    Nline = len(cnu)
    Mat = jnp.hstack([nsigmaDl, ngammaLM, SijM, dgm_ngammaL])

    def fxs(x, arr):
        carry = 0.0
        nsigmaD = arr[0:1]
        ngammaL = arr[1:Nline + 1]
        Sij = arr[Nline + 1:2 * Nline + 1]
        ngammaL_grid = arr[2 * Nline + 1:2 * Nline + NDITgrid + 1]
        arr = xsvector(cnu, indexnu, R, pmarray, nsigmaD, ngammaL, Sij,
                       nu_grid, ngammaL_grid)
        return carry, arr

    val, xsm = scan(fxs, 0.0, Mat)
    return xsm


def exomol(mdb, Tarr, Parr, R, molmass):
    """compute molecular line information required for MODIT using Exomol mdb.

    Args:
        mdb: mdb instance
        Tarr: Temperature array
        Parr: Pressure array
        R: spectral resolution
        molmass: molecular mass
        wavmask: mask for wavenumber #Issue 341
        
    Returns:
        line intensity matrix,
        normalized gammaL matrix,
        normalized sigmaD matrix
    """
    qt = vmap(mdb.qr_interp)(Tarr)
    SijM = jit(vmap(line_strength, (0, None, None, None, 0)))(Tarr, mdb.logsij0,
                                                     mdb.dev_nu_lines,
                                                     mdb.elower, qt)
    gammaLMP = jit(vmap(gamma_exomol,
                        (0, 0, None, None)))(Parr, Tarr, mdb.n_Texp,
                                             mdb.alpha_ref)
    gammaLMN = gamma_natural(mdb.A)
    gammaLM = gammaLMP + gammaLMN[None, :]
    ngammaLM = gammaLM / (mdb.dev_nu_lines / R)
    nsigmaDl = normalized_doppler_sigma(Tarr, molmass, R)[:, jnp.newaxis]
    return SijM, ngammaLM, nsigmaDl


def setdgm_exomol(mdb, fT, Parr, R, molmass, dit_grid_resolution, *kargs):
    warn_msg = " Use `modit.set_ditgrid_matrix_exomol` instead"
    warnings.warn(warn_msg, FutureWarning)
    return set_ditgrid_matrix_exomol(mdb, fT, Parr, R, molmass,
                                     dit_grid_resolution, *kargs)


def set_ditgrid_matrix_exomol(mdb, fT, Parr, R, molmass, dit_grid_resolution,
                              *kargs):
    """Easy Setting of DIT Grid Matrix (dgm) using Exomol.

    Args:
       mdb: mdb instance
       fT: function of temperature array
       Parr: pressure array
       R: spectral resolution
       molmass: molecular mass
       dit_grid_resolution: resolution of dgm
       *kargs: arguments for fT

    Returns:
       DIT Grid Matrix (dgm) of normalized gammaL

    Example:

       >>> fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
       >>> T0_test=np.array([1100.0,1500.0,1100.0,1500.0])
       >>> alpha_test=np.array([0.2,0.2,0.05,0.05])
       >>> dit_grid_resolution=0.2
       >>> dgm_ngammaL=setdgm_exomol(mdbCH4,fT,Parr,R,molmassCH4,dit_grid_resolution,T0_test,alpha_test)
    """
    set_dgm_minmax = []
    Tarr_list = fT(*kargs)
    for Tarr in Tarr_list:
        SijM, ngammaLM, nsigmaDl = exomol(mdb, Tarr, Parr, R, molmass)
        set_dgm_minmax.append(
            minmax_ditgrid_matrix(ngammaLM, dit_grid_resolution))
    dgm_ngammaL = precompute_modit_ditgrid_matrix(
        set_dgm_minmax, dit_grid_resolution=dit_grid_resolution)
    return jnp.array(dgm_ngammaL)


def hitran(mdb, Tarr, Parr, Pself, R, molmass):
    """compute molecular line information required for MODIT using HITRAN/HITEMP mdb.

    Args:
       mdb: mdb instance
       Tarr: Temperature array
       Parr: Pressure array
       Pself: Partial pressure array
       R: spectral resolution
       molmass: molecular mass

    Returns:
       line intensity matrix,
       normalized gammaL matrix,
       normalized sigmaD matrix
    """
    qt = vmap(mdb.qr_interp_lines)(Tarr)
    SijM = jit(vmap(line_strength, (0, None, None, None, 0)))(Tarr, mdb.logsij0,
                                                     mdb.dev_nu_lines,
                                                     mdb.elower, qt)
    gammaLMP = jit(vmap(gamma_hitran,
                        (0, 0, 0, None, None, None)))(Parr, Tarr, Pself,
                                                      mdb.n_air, mdb.gamma_air,
                                                      mdb.gamma_self)
    gammaLMN = gamma_natural(mdb.A)
    gammaLM = gammaLMP + gammaLMN[None, :]
    ngammaLM = gammaLM / (mdb.dev_nu_lines / R)
    nsigmaDl = normalized_doppler_sigma(Tarr, molmass, R)[:, jnp.newaxis]
    return SijM, ngammaLM, nsigmaDl


def setdgm_hitran(mdb, fT, Parr, Pself_ref, R, molmass, dit_grid_resolution,
                  *kargs):
    warn_msg = " Use `modit.set_ditgrid_matrix_hitran` instead"
    warnings.warn(warn_msg, FutureWarning)
    return set_ditgrid_matrix_hitran(mdb, fT, Parr, Pself_ref, R, molmass,
                                     dit_grid_resolution, *kargs)


def set_ditgrid_matrix_hitran(mdb, fT, Parr, Pself_ref, R, molmass,
                              dit_grid_resolution, *kargs):
    """Easy Setting of DIT Grid Matrix (dgm) using HITRAN/HITEMP.

    Args:
       mdb: mdb instance
       fT: function of temperature array
       Parr: pressure array
       Pself_ref: reference partial pressure array
       R: spectral resolution
       molmass: molecular mass
       dit_grid_resolution: resolution of dgm
       *kargs: arguments for fT

    Returns:
       DIT Grid Matrix (dgm) of normalized gammaL

    Example:

       >>> fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
       >>> T0_test=np.array([1100.0,1500.0,1100.0,1500.0])
       >>> alpha_test=np.array([0.2,0.2,0.05,0.05])
       >>> dit_grid_resolution=0.2
       >>> dgm_ngammaL=setdgm_hitran(mdbCH4,fT,Parr,Pself,R,molmassCH4,dit_grid_resolution,T0_test,alpha_test)
    """
    set_dgm_minmax = []
    Tarr_list = fT(*kargs)
    for Tarr in Tarr_list:
        SijM, ngammaLM, nsigmaDl = hitran(mdb, Tarr, Parr, Pself_ref, R,
                                          molmass)
        set_dgm_minmax.append(
            minmax_ditgrid_matrix(ngammaLM, dit_grid_resolution))
    dgm_ngammaL = precompute_modit_ditgrid_matrix(
        set_dgm_minmax, dit_grid_resolution=dit_grid_resolution)
    return jnp.array(dgm_ngammaL)


@jit
def vald_each(Tarr, PH, PHe, PHH, R, qt_284_T, QTmask, \
               ielem, iion, atomicmass, ionE, dev_nu_lines, logsij0, elower, eupper, gamRad, gamSta, vdWdamp):
    """Compute atomic line information required for MODIT for separated EACH species, using parameters attributed in VALD separated atomic database (asdb).

    Args:
        Tarr:  temperature array [N_layer]
        PH:  partial pressure array of neutral hydrogen (H) [N_layer]
        PHe:  partial pressure array of neutral helium (He) [N_layer]
        PHH:  partial pressure array of molecular hydrogen (H2) [N_layer]
        R:  spectral resolution [scalar]
        qt_284_T:  partition function at the temperature T Q(T), for 284 species
        QTmask:  array of index of Q(Tref) grid (gQT) for each line
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)
        atomicmass:  atomic mass (amu)
        ionE:  ionization potential (eV)
        dev_nu_lines:  line center (cm-1) in device
        logsij0:  log line strength at T=Tref
        elower:  the lower state energy (cm-1)
        eupper:  the upper state energy (cm-1)
        gamRad:  log of gamma of radiation damping (s-1)
        gamSta:  log of gamma of Stark damping (s-1)
        vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)

    Returns:
        SijM:  line intensity matrix [N_layer x N_line]
        ngammaLM:  normalized gammaL matrix [N_layer x N_line]
        nsigmaDl:  normalized sigmaD matrix [N_layer x 1]
    """
    # Compute normalized partition function for each species
    qt = qt_284_T[:, QTmask]

    # Compute line strength matrix
    SijM = jit(vmap(line_strength,(0,None,None,None,0)))\
        (Tarr, logsij0, dev_nu_lines, elower, qt)

    # Compute gamma parameters for the pressure and natural broadenings
    gammaLM = jit(vmap(gamma_vald3,(0,0,0,0,None,None,None,None,None,None,None,None,None,None,None)))\
            (Tarr, PH, PHH, PHe, ielem, iion, dev_nu_lines, elower, eupper, atomicmass, ionE, gamRad, gamSta, vdWdamp, 1.0)
    ngammaLM = gammaLM / (dev_nu_lines / R)
    # Do NOT remove NaN because "set_ditgrid_matrix_vald_each" makes good use of them. # ngammaLM = jnp.nan_to_num(ngammaLM, nan = 0.0)

    # Compute doppler broadening
    nsigmaDl = normalized_doppler_sigma(Tarr, atomicmass, R)[:, jnp.newaxis]
    return SijM, ngammaLM, nsigmaDl


def vald_all(asdb, Tarr, PH, PHe, PHH, R):
    """Compute atomic line information required for MODIT for separated ALL species, using VALD separated atomic database (asdb).

    Args:
        asdb:  asdb instance made by the AdbSepVald class in moldb.py
        Tarr:  temperature array [N_layer]
        PH:  partial pressure array of neutral hydrogen (H) [N_layer]
        PHe:  partial pressure array of neutral helium (He) [N_layer]
        PHH:  partial pressure array of molecular hydrogen (H2) [N_layer]
        R:  spectral resolution [scalar]

    Returns:
        SijMS:  line intensity matrix [N_species x N_layer x N_line]
        ngammaLMS:  normalized gammaL matrix [N_species x N_layer x N_line]
        nsigmaDlS:  normalized sigmaD matrix [N_species x N_layer x 1]
    """
    gQT_284species = asdb.gQT_284species
    T_gQT = asdb.T_gQT
    qt_284_T = vmap(interp_QT284, (0, None, None))(Tarr, T_gQT, gQT_284species)

    SijMS, ngammaLMS, nsigmaDlS = jit(vmap(vald_each, (None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, )))\
        (Tarr, PH, PHe, PHH, R, qt_284_T, \
                 asdb.QTmask, asdb.ielem, asdb.iion, asdb.atomicmass, asdb.ionE, \
                       asdb.dev_nu_lines, asdb.logsij0, asdb.elower, asdb.eupper, asdb.gamRad, asdb.gamSta, asdb.vdWdamp)

    return SijMS, ngammaLMS, nsigmaDlS


def setdgm_vald_each(ielem, iion, atomicmass, ionE, dev_nu_lines, logsij0, elower, eupper, gamRad, gamSta, vdWdamp, \
                QTmask, T_gQT, gQT_284species, PH, PHe, PHH, R, fT, dit_grid_resolution, *kargs):
    warn_msg = " Use `modit.set_ditgrid_matrix_vald_each` instead"
    warnings.warn(warn_msg, FutureWarning)
    return set_ditgrid_matrix_vald_each(ielem, iion, atomicmass, ionE,
                                        dev_nu_lines, logsij0, elower, eupper,
                                        gamRad, gamSta, vdWdamp, QTmask, T_gQT,
                                        gQT_284species, PH, PHe, PHH, R, fT,
                                        dit_grid_resolution, *kargs)


def set_ditgrid_matrix_vald_each(ielem, iion, atomicmass, ionE, dev_nu_lines,
                                 logsij0, elower, eupper, gamRad, gamSta,
                                 vdWdamp, QTmask, T_gQT, gQT_284species, PH,
                                 PHe, PHH, R, fT, dit_grid_resolution, *kargs):
    """Easy Setting of DIT Grid Matrix (dgm) using VALD.

    Args:
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly ionized=2, etc.)
        atomicmass:  atomic mass (amu)
        ionE:  ionization potential (eV)
        dev_nu_lines:  line center (cm-1) in device
        logsij0:  log line strength at T=Tref
        elower:  the lower state energy (cm-1)
        eupper:  the upper state energy (cm-1)
        gamRad:  log of gamma of radiation damping (s-1)
        gamSta:  log of gamma of Stark damping (s-1)
        vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
        T_gQT:  temperature in the grid obtained from the adb instance
        gQT_284species:  partition function in the grid from the adb instance
        QTmask:  array of index of Q(Tref) grid (gQT) for each line
        PH:  partial pressure array of neutral hydrogen (H) [N_layer]
        PHe:  partial pressure array of neutral helium (He) [N_layer]
        PHH:  partial pressure array of molecular hydrogen (H2) [N_layer]
        R:  spectral resolution
        fT:  function of temperature array
        dit_grid_resolution:  resolution of dgm
        *kargs:  arguments for fT

    Returns:
        dgm_ngammaL:  DIT Grid Matrix (dgm) of normalized gammaL [N_layer x N_DITgrid]
    """
    set_dgm_minmax = []
    Tarr_list = fT(*kargs)
    for Tarr in Tarr_list:
        qt_284_T = vmap(interp_QT284, (0, None, None))(Tarr, T_gQT,
                                                       gQT_284species)
        SijM, ngammaLM, nsigmaDl = vald_each(Tarr, PH, PHe, PHH, R, qt_284_T, \
             QTmask, ielem, iion, atomicmass, ionE, \
                   dev_nu_lines, logsij0, elower, eupper, gamRad, gamSta, vdWdamp)
        floop = lambda c, arr: (c,
                                jnp.nan_to_num(arr,
                                               nan=jnp.nanmin(arr),
                                               posinf=jnp.nanmin(arr),
                                               neginf=jnp.nanmin(arr)))
        ngammaLM = scan(floop, 0, ngammaLM)[1]
        set_dgm_minmax.append(
            minmax_ditgrid_matrix(ngammaLM, dit_grid_resolution))
    dgm_ngammaL = precompute_modit_ditgrid_matrix(
        set_dgm_minmax, dit_grid_resolution=dit_grid_resolution)
    return jnp.array(dgm_ngammaL)


def setdgm_vald_all(asdb, PH, PHe, PHH, R, fT, dit_grid_resolution, *kargs):
    warn_msg = " Use `modit.set_ditgrid_matrix_vald_all` instead"
    warnings.warn(warn_msg, FutureWarning)
    return set_ditgrid_matrix_vald_all(asdb, PH, PHe, PHH, R, fT,
                                       dit_grid_resolution, *kargs)


def set_ditgrid_matrix_vald_all(asdb, PH, PHe, PHH, R, fT, dit_grid_resolution,
                                *kargs):
    """Easy Setting of DIT Grid Matrix (dgm) using VALD.
    
    Args:
        asdb:  asdb instance made by the AdbSepVald class in moldb.py
        PH:  partial pressure array of neutral hydrogen (H) [N_layer]
        PHe:  partial pressure array of neutral helium (He) [N_layer]
        PHH:  partial pressure array of molecular hydrogen (H2) [N_layer]
        R:  spectral resolution
        fT:  function of temperature array
        dit_grid_resolution:  resolution of dgm
        *kargs:  arguments for fT

    Returns:
        dgm_ngammaLS:  DIT Grid Matrix (dgm) of normalized gammaL [N_species x N_layer x N_DITgrid]

    Example:
       >>> fT = lambda T0,alpha:  T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
       >>> T0_test=np.array([3000.0,4000.0,3000.0,4000.0])
       >>> alpha_test=np.array([0.2,0.2,0.05,0.05])
       >>> dit_grid_resolution=0.2
       >>> dgm_ngammaLS = setdgm_vald_all(asdb, PH, PHe, PHH, R, fT, dit_grid_resolution, T0_test, alpha_test)
    """
    T_gQT = asdb.T_gQT
    gQT_284species = asdb.gQT_284species

    dgm_ngammaLS_BeforePadding = []
    lendgm = []
    for i in range(asdb.N_usp):
        dgm_ngammaL_sp = set_ditgrid_matrix_vald_each(asdb.ielem[i], asdb.iion[i], asdb.atomicmass[i], asdb.ionE[i], \
            asdb.dev_nu_lines[i], asdb.logsij0[i], asdb.elower[i], asdb.eupper[i], asdb.gamRad[i], asdb.gamSta[i], asdb.vdWdamp[i], \
            asdb.QTmask[i], T_gQT, gQT_284species, PH, PHe, PHH, R, fT, dit_grid_resolution, *kargs)
        dgm_ngammaLS_BeforePadding.append(dgm_ngammaL_sp)
        lendgm.append(dgm_ngammaL_sp.shape[1])
    Lmax_dgm = np.max(np.array(lendgm))

    # Padding to unity the length of all the DIT Grid Matrix (dgm) and convert them into jnp.array
    pad2Dm = lambda arr, L: jnp.pad(arr, ((0, 0), (0, L - arr.shape[1])),
                                    mode='maximum')
    dgm_ngammaLS = np.zeros([asdb.N_usp, len(PH), Lmax_dgm])
    for i_sp, dgmi in enumerate(dgm_ngammaLS_BeforePadding):
        dgm_ngammaLS[i_sp] = pad2Dm(dgmi, Lmax_dgm)
    return jnp.array(dgm_ngammaLS)


@jit
def xsmatrix_vald(cnuS, indexnuS, R, pmarray, nsigmaDlS, ngammaLMS, SijMS,
                  nu_grid, dgm_ngammaLS):
    """Cross section matrix for xsvector (MODIT) for VALD lines (asdb)

    Args:
        cnuS: contribution by npgetix for wavenumber [N_species x N_line]
        indexnuS: index by npgetix for wavenumber [N_species x N_line]
        R: spectral resolution
        pmarray: (+1,-1) array whose length of len(nu_grid)+1
        nsigmaDlS: normalized sigmaD matrix [N_species x N_layer x 1]
        ngammaLMS: normalized gammaL matrix [N_species x N_layer x N_line]
        SijMS: line intensity matrix [N_species x N_layer x N_line]
        nu_grid: linear wavenumber grid
        dgm_ngammaLS: DIT Grid Matrix (dgm) of normalized gammaL [N_species x N_layer x N_DITgrid]

    Return:
        xsmS: cross section matrix [N_species x N_layer x N_wav]
    """
    xsmS = jit(vmap(xsmatrix, (0, 0, None, None, 0, 0, 0, None, 0)))(\
                    cnuS, indexnuS, R, pmarray, nsigmaDlS, ngammaLMS, SijMS, nu_grid, dgm_ngammaLS)
    xsmS = jnp.abs(xsmS)
    return xsmS


#
def precompute_dgmatrix(set_gm_minmax, dit_grid_resolution=0.1, adopt=True):
    """Precomputing MODIT GRID MATRIX for normalized GammaL.

    Args:
        set_gm_minmax: set of gm_minmax for different parameters [Nsample, Nlayers, 2], 2=min,max
        dit_grid_resolution: grid resolution. dit_grid_resolution=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x. In this case, the grid width does not need to be dit_grid_resolution exactly.

    Returns:
        grid for DIT (Nlayer x NDITgrid)
    """
    warn_msg = " Use `set_ditgrid.precompute_modit_ditgrid_matrix` instead"
    warnings.warn(warn_msg, FutureWarning)
    return precompute_modit_ditgrid_matrix(set_gm_minmax, dit_grid_resolution,
                                           adopt)


def minmax_dgmatrix(x, dit_grid_resolution=0.1, adopt=True):
    """compute MIN and MAX DIT GRID MATRIX.

    Args:
        x: gammaL matrix (Nlayer x Nline)
        dit_grid_resolution: grid resolution. dit_grid_resolution=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x. In this case, the grid width does not need to be dit_grid_resolution exactly.

    Returns:
        minimum and maximum for DIT (dgm_minmax)
    """
    warn_msg = "Deprecated Use `set_ditgrid.minmax_ditgrid_matrix` instead"
    warnings.warn(warn_msg, FutureWarning)
    return minmax_ditgrid_matrix(x, dit_grid_resolution, adopt)


def dgmatrix(x, dit_grid_resolution=0.1, adopt=True):
    """DIT GRID MATRIX (alias)

    Args:
        x: simgaD or gammaL matrix (Nlayer x Nline)
        dit_grid_resolution: grid resolution. dit_grid_resolution=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x.
               In this case, the grid width does not need to be dit_grid_resolution exactly.

    Returns:
        grid for DIT (Nlayer x NDITgrid)
    """
    warn_msg = "Deprecated Use `set_ditgrid.ditgrid_matrix` instead"
    warnings.warn(warn_msg, FutureWarning)
    from exojax.spec.set_ditgrid import ditgrid_matrix
    return ditgrid_matrix(x, dit_grid_resolution, adopt)


def ditgrid(x, dit_grid_resolution=0.1, adopt=True):
    """DIT GRID (deprecated).

    Args:
        x: simgaD or gammaL array (Nline)
        dit_grid_resolution: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x.
               In this case, the grid width does not need to be res exactly.

    Returns:
        grid for DIT
    """

    warn_msg = "Deprecated Use `set_ditgrid.ditgrid_log_interval` instead"
    warnings.warn(warn_msg, FutureWarning)
    from exojax.spec.set_ditgrid import ditgrid_log_interval
    return ditgrid_log_interval(x, dit_grid_resolution, adopt)


####################################################################
