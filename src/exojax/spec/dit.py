"""Line profile computation using Discrete Integral Transform.

* Line profile computation of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekerom and E.Pannier.
* This module consists of selected functions in `addit package <https://github.com/HajimeKawahara/addit>`_.
* The concept of "folding" can be understood by reading `the discussion <https://github.com/radis/radis/issues/186#issuecomment-764465580>`_ by D.C.M van den Bekerom.
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from exojax.spec.ditkernel import fold_voigt_kernel
from jax.ops import index as joi
from exojax.spec.atomll import padding_2Darray_for_each_atom
from exojax.spec.rtransfer import dtauM


def getix(x, xv):
    """jnp version of getix.

    Args:
        x: x array
        xv: x grid 

    Returns:
        cont (contribution)
        index (index)

    Note:
       cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.

    Example:

       >>> from exojax.spec.dit import getix
       >>> import jax.numpy as jnp
       >>> y=jnp.array([1.1,4.3])
       >>> yv=jnp.arange(6)
       >>> getix(y,yv)
       (DeviceArray([0.10000002, 0.3000002 ], dtype=float32), DeviceArray([1, 4], dtype=int32))
    """
    indarr = jnp.arange(len(xv))
    pos = jnp.interp(x, xv, indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont, index


def npgetix(x, xv):
    """numpy version of getix.

    Args:
        x: x array
        xv: x grid

    Returns:
        cont (contribution)
        index (index)

    Note:
       cont is the contribution for i=index+1. 1 - cont is the contribution for i=index. For other i, the contribution should be zero.
    """
    indarr = np.arange(len(xv))
    pos = np.interp(x, xv, indarr)
    index = (pos).astype(int)
    cont = (pos-index)
    return cont, index


@jit
def inc3D_givenx(a, w, cx, ix, y, z, xv, yv, zv):
    """The lineshape distribution matrix = integrated neighbouring contribution for 3D (memory reduced sum) but using given contribution and index for x .

    Args:
        a: lineshape density array (jnp.array)
        w: weight (N)
        cx: given contribution for x 
        ix: given index for x 
        y: y values (N)
        z: z values (N)
        xv: x grid
        yv: y grid
        zv: z grid            

    Returns:
        lineshape distribution matrix (integrated neighbouring contribution for 3D)

    Note:
        This function computes \sum_n w_n fx_n \otimes fy_n \otimes fz_n, 
        where w_n is the weight, fx_n, fy_n, and fz_n are the n-th NCFs for 1D. 
        A direct sum uses huge RAM. 
        In this function, we use jax.lax.scan to compute the sum

    """

    cy, iy = getix(y, yv)
    cz, iz = getix(z, zv)

    a = a.at[joi[ix, iy, iz]].add(w*(1-cx)*(1-cy)*(1-cz))
    a = a.at[joi[ix, iy+1, iz]].add(w*(1-cx)*cy*(1-cz))
    a = a.at[joi[ix+1, iy, iz]].add(w*cx*(1-cy)*(1-cz))
    a = a.at[joi[ix+1, iy+1, iz]].add(w*cx*cy*(1-cz))
    a = a.at[joi[ix, iy, iz+1]].add(w*(1-cx)*(1-cy)*cz)
    a = a.at[joi[ix, iy+1, iz+1]].add(w*(1-cx)*cy*cz)
    a = a.at[joi[ix+1, iy, iz+1]].add(w*cx*(1-cy)*cz)
    a = a.at[joi[ix+1, iy+1, iz+1]].add(w*cx*cy*cz)

    return a


@jit
def xsvector(cnu, indexnu, pmarray, sigmaD, gammaL, S, nu_grid, sigmaD_grid, gammaL_grid):
    """Cross section vector (DIT/2D+ version; default)

    The original code is rundit in [addit package](https://github.com/HajimeKawahara/addit)

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       sigmaD: Gaussian STD (Nlines)
       gammaL: Lorentzian half width (Nlines)
       S: line strength (Nlines)
       nu_grid: linear wavenumber grid
       sigmaD_grid: sigmaD grid
       gammaL_grid: gammaL grid

    Returns:
       Cross section in the linear nu grid

    Note:
       This function uses the precomputed neibouring contribution function for wavenumber (nu_ncf). Use npnc1D to compute nu_ncf in float64 precision.
    """
    Ng_nu = len(nu_grid)
    Ng_sigmaD = len(sigmaD_grid)
    Ng_gammaL = len(gammaL_grid)

    log_sigmaD = jnp.log(sigmaD)
    log_gammaL = jnp.log(gammaL)

    log_sigmaD_grid = jnp.log(sigmaD_grid)
    log_gammaL_grid = jnp.log(gammaL_grid)
    dnu = (nu_grid[-1]-nu_grid[0])/(Ng_nu-1)
    k = jnp.fft.rfftfreq(2*Ng_nu, dnu)

    lsda = jnp.zeros(
        (len(nu_grid), len(log_sigmaD_grid), len(log_gammaL_grid)))
    val = inc3D_givenx(lsda, S, cnu, indexnu, log_sigmaD,
                       log_gammaL, nu_grid, log_sigmaD_grid, log_gammaL_grid)

    valbuf = jnp.vstack([val, jnp.zeros_like(val)])
    fftval = jnp.fft.rfft(valbuf, axis=0)
    vmax = Ng_nu*dnu
    vk = fold_voigt_kernel(k, sigmaD_grid, gammaL_grid, vmax, pmarray)
    fftvalsum = jnp.sum(fftval*vk, axis=(1, 2))
    xs = jnp.fft.irfft(fftvalsum)[:Ng_nu]/dnu
    return xs


@jit
def xsmatrix(cnu, indexnu, pmarray, sigmaDM, gammaLM, SijM, nu_grid, dgm_sigmaD, dgm_gammaL):
    """Cross section matrix (DIT/2D+ version)

    Args:
       cnu: contribution by npgetix for wavenumber
       indexnu: index by npgetix for wavenumber
       pmarray: (+1,-1) array whose length of len(nu_grid)+1
       sigmaDM: doppler sigma matrix in R^(Nlayer x Nline)
       gammaLM: gamma factor matrix in R^(Nlayer x Nline)
       SijM: line strength matrix in R^(Nlayer x Nline)
       nu_grid: linear wavenumber grid
       dgm_sigmaD: DIT Grid Matrix for sigmaD R^(Nlayer, NDITgrid)
       dgm_gammaL: DIT Grid Matrix for gammaL R^(Nlayer, NDITgrid)

    Return:
       cross section matrix in R^(Nlayer x Nwav)

    Warning:
       This function have not been well tested.
    """
    NDITgrid = jnp.shape(dgm_sigmaD)[1]
    Nline = len(cnu)
    Mat = jnp.hstack([sigmaDM, gammaLM, SijM, dgm_sigmaD, dgm_gammaL])

    def fxs(x, arr):
        carry = 0.0
        sigmaD = arr[0:Nline]
        gammaL = arr[Nline:2*Nline]
        Sij = arr[2*Nline:3*Nline]
        sigmaD_grid = arr[3*Nline:3*Nline+NDITgrid]
        gammaL_grid = arr[3*Nline+NDITgrid:3*Nline+2*NDITgrid]
        arr = xsvector(cnu, indexnu, pmarray, sigmaD, gammaL,
                       Sij, nu_grid, sigmaD_grid, gammaL_grid)
        return carry, arr

    val, xsm = scan(fxs, 0.0, Mat)
    return xsm


def ditgrid(x, res=0.1, adopt=True):
    """DIT GRID.

    Args:
        x: simgaD or gammaL array (Nline)
        res: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x.
               In this case, the grid width does not need to be res exactly.

    Returns:
        grid for DIT
    """
    if np.min(x) <= 0.0:
        print('Warning: there exists negative or zero gamma. MODIT/DIT does not support this case.')

    lxmin = np.log(np.min(x))
    lxmax = np.log(np.max(x))
    lxmax = np.nextafter(lxmax, np.inf, dtype=lxmax.dtype)

    dlog = lxmax-lxmin
    Ng = int(dlog/res)+2
    if adopt == False:
        grid = np.exp(np.linspace(lxmin, lxmin+(Ng-1)*res, Ng))
    else:
        grid = np.exp(np.linspace(lxmin, lxmax, Ng))
    return grid


def set_ditgrid(x, res=0.1, adopt=True):
    """alias of ditgrid.

    Args:
        x: simgaD or gammaL array (Nline)
        res: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x.
               In this case, the grid width does not need to be res exactly.

    Returns:
        grid for DIT
    """
    return ditgrid(x, res, adopt)


def dgmatrix(x, res=0.1, adopt=True):
    """DIT GRID MATRIX.

    Args:
        x: simgaD or gammaL matrix (Nlayer x Nline)
        res: grid resolution. res=0.1 (defaut) means a grid point per digit
        adopt: if True, min, max grid points are used at min and max values of x.
               In this case, the grid width does not need to be res exactly.

    Returns:
        grid for DIT (Nlayer x NDITgrid)
    """
    mmin = np.log(np.min(x, axis=1))
    mmax = np.log(np.max(x, axis=1))
    mmax = np.nextafter(mmax, np.inf, dtype=mmax.dtype)

    Nlayer = np.shape(mmax)[0]
    gm = []
    dlog = np.max(mmax-mmin)
    Ng = (dlog/res).astype(int)+2
    for i in range(0, Nlayer):
        lxmin = mmin[i]
        lxmax = mmax[i]
        if adopt == False:
            grid = np.exp(np.linspace(lxmin, lxmin+(Ng-1)*res, Ng))
        else:
            grid = np.exp(np.linspace(lxmin, lxmax, Ng))
        gm.append(grid)
    gm = np.array(gm)
    return gm


def sigma_voigt(dgm_sigmaD, dgm_gammaL):
    """compute sigma of the Voigt profile.

    Args:
       dgm_sigmaD: DIT grid matrix for sigmaD
       dgm_gammaL: DIT grid matrix for gammaL

    Returns:
       sigma
    """
    fac = 2.*np.sqrt(2.*np.log(2.0))
    fdgm_gammaL = jnp.min(dgm_gammaL, axis=1)*2.0
    fdgm_sigmaD = jnp.min(dgm_sigmaD, axis=1)*fac
    fv = jnp.min(0.5346*fdgm_gammaL +
                 jnp.sqrt(0.2166*fdgm_gammaL**2+fdgm_sigmaD**2))
    sigma = fv/fac
    return sigma


def vald(adb, Tarr, PH, PHe, PHH):
    """(alias of lpf.vald)
    
    Args:
       adb: adb instance made by the AdbVald class in moldb.py
       Tarr: Temperature array
       PH: Partial pressure array of neutral hydrogen (H)
       PHe: Partial pressure array of neutral helium (He)
       PHH: Partial pressure array of molecular hydrogen (H2)

    Returns:
       SijM: line intensity matrix
       gammaLM: gammaL matrix
       sigmaDM: sigmaD matrix
    
    """
    from exojax.spec.lpf import vald as vald_
    return(vald_(adb, Tarr, PH, PHe, PHH))


def dtauM_vald_old(dParr, xsm, g, uspecies, mods_uspecies_list, MMR_uspecies_list, atomicmass_uspecies_list):
    """Compute dtau caused by VALD lines from cross section xs (DIT)
    
    Args:
       dParr: delta pressure profile (bar) [N_layer]
       xsm: cross section matrix (cm^2) [N_layer, N_nus]
       g: gravity (cm/s^2)
       uspecies: unique elements of the combination of ielem and iion [N_UniqueSpecies x 2(ielem and iion)]
       mods_uspecies_list: jnp.array of abundance deviation from the Sun [dex] for each species in "uspecies" [N_UniqueSpecies]
       MMR_uspecies_list: jnp.array of mass mixing ratio in the Sun of each species in "uspecies" [N_UniqueSpecies]
       atomicmass_uspecies_list: jnp.array of atomic mass [amu] of each species in "uspecies" [N_UniqueSpecies]
    
    Returns:
       dtauatom: optical depth matrix [N_layer, N_nus]
    
    """
    zero_to_ones = lambda arr: jnp.where(arr!=0, arr, 1.)
    def floop(xi, null):
        i, dtauatom = xi
        # process---->
        sp = uspecies[i]
        MMRmetalMod = mods_uspecies_list[i] #add_to_deal_with_individual_elemental_abundance
        MMR_X_I = MMR_uspecies_list[i] *10**MMRmetalMod #modify this into individual elemental abundances shortly... (tako)
        mass_X_I = atomicmass_uspecies_list[i]
        
        dtau_each = dtauM(dParr, xsm, MMR_X_I*jnp.ones_like(dParr), mass_X_I, g)
        # Note that the same mixing ratio is assumed for all atmospheric layers here...
        dtauatom = dtauatom + dtau_each
        # <----process
        i = i+1
        xi = [i, dtauatom]
        return xi, null

    def f_dtaual(xi0):
        xi, null = scan(floop, xi0, None, length)
        return xi

    length = len(uspecies)
    dtauatom_init = jnp.zeros_like(xsm)
    xi_init = [0, dtauatom_init]

    dtauatom = f_dtaual(xi_init)[1]
    return(dtauatom)


def dtauM_vald(dParr, g, adb, nus, cnu, indexnu, pmarray, SijM, gammaLM, sigmaDM, \
        uspecies, mods_uspecies_list, MMR_uspecies_list, atomicmass_uspecies_list, dgm_sigmaD, dgm_gammaL):
    """Compute dtau caused by VALD lines from cross section xs (DIT)
    
    Args:
       dParr: delta pressure profile (bar) [N_layer]
       g: gravity (cm/s^2)
       adb: adb instance made by the AdbVald class in moldb.py
       nus: wavenumber matrix (cm-1) [N_nus]
       cnu: cont (contribution) jnp.array [N_line]
       indexnu: index (index) jnp.array [N_line]
       pmarray: (+1,-1) array [len(nu_grid)+1,]
       SijM: line intensity matrix [N_layer x N_line]
       gammaLM: gammaL matrix [N_layer x N_line]
       sigmaDM: sigmaD matrix [N_layer x N_line]
       uspecies: unique elements of the combination of ielem and iion [N_UniqueSpecies x 2(ielem and iion)]
       mods_uspecies_list: jnp.array of abundance deviation from the Sun [dex] for each species in "uspecies" [N_UniqueSpecies]
       MMR_uspecies_list: jnp.array of mass mixing ratio in the Sun of each species in "uspecies" [N_UniqueSpecies]
       atomicmass_uspecies_list: jnp.array of atomic mass [amu] of each species in "uspecies" [N_UniqueSpecies]
    
    Returns:
       dtauatom: optical depth matrix [N_layer, N_nus]
    
    """
    zero_to_ones = lambda arr: jnp.where(arr!=0, arr, 1.)
    def floop(xi, null):
        i, dtauatom = xi
        # process---->
        #test220208 dgm_sigmaD = dgml_sigmaD[i]
        sp = uspecies[i]
        cnu_p = padding_2Darray_for_each_atom(cnu[None,:], adb, sp).reshape(cnu.shape)
        indexnu_p = jnp.array(\
                padding_2Darray_for_each_atom(indexnu[None,:], adb, sp).reshape(indexnu.shape)\
                , dtype='int32')
        sigmaDM_p = zero_to_ones(padding_2Darray_for_each_atom(sigmaDM, adb, sp))
        gammaLM_p = zero_to_ones(padding_2Darray_for_each_atom(gammaLM, adb, sp))
        SijM_p = padding_2Darray_for_each_atom(SijM, adb, sp)
        #test220207 dgm_sigmaD_p = dgmatrix(sigmaDM_p)
        #test220207 dgm_gammaL_p = dgmatrix(gammaLM_p)
        xsm_p = xsmatrix(cnu_p, indexnu_p, pmarray, sigmaDM_p, gammaLM_p, SijM_p, nus, dgm_sigmaD, dgm_gammaL)
        xsm_p = jnp.abs(xsm_p)

        MMRmetalMod = mods_uspecies_list[i] #add_to_deal_with_individual_elemental_abundance
        MMR_X_I = MMR_uspecies_list[i] *10**MMRmetalMod #modify this into individual elemental abundances shortly... (tako)
        mass_X_I = atomicmass_uspecies_list[i]
        
        dtau_each = dtauM(dParr, xsm_p, MMR_X_I*jnp.ones_like(dParr), mass_X_I, g)
        # Note that the same mixing ratio is assumed for all atmospheric layers here...
        dtauatom = dtauatom + dtau_each
        # <----process
        i = i+1
        xi = [i, dtauatom]
        return xi, null

    def f_dtaual(xi0):
        xi, null = scan(floop, xi0, None, length)
        return xi

    length = len(uspecies)
    dtauatom_init = jnp.zeros([len(dParr), len(nus)])
    xi_init = [0, dtauatom_init]

    dtauatom = f_dtaual(xi_init)[1]
    return(dtauatom)

