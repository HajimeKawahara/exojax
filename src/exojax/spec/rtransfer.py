"""Radiative transfer module used in exospectral analysis."""
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
from exojax.special.expn import E1
from exojax.spec.hitrancia import logacia
from exojax.spec.hminus import log_hminus_continuum
from exojax.atm.idealgas import number_density
from exojax.spec.unitconvert import nu2wav, wav2nu
from exojax.spec.check_nugrid import check_scale_xsmode, check_scale_nugrid, warn_resolution
from exojax.utils.constants import kB, logm_ucgs
from exojax.utils.instfunc import resolution_eslog, resolution_eslin


def nugrid(x0, x1, N, unit='cm-1', xsmode='lpf'):
    """generating the recommended wavenumber grid based on the cross section
    computation mode.

    Args:
       x0: start wavenumber (cm-1) or wavelength (nm) or (AA)
       x1: end wavenumber (cm-1) or wavelength (nm) or (AA)
       N: the number of the wavenumber grid
       unit: unit of the input grid
       xsmode: cross section computation mode (lpf, dit, modit, redit)

    Returns:
       wavenumber grid evenly spaced in log space
       corresponding wavelength grid
       resolution
    """
    if check_scale_xsmode(xsmode) == 'ESLOG':
        if unit == 'cm-1':
            nus = np.logspace(np.log10(x0), np.log10(x1), N, dtype=np.float64)
            wav = nu2wav(nus)
        elif unit == 'nm' or unit == 'AA':
            wav = np.logspace(np.log10(x0), np.log10(x1), N, dtype=np.float64)
            nus = wav2nu(wav, unit)

        resolution = resolution_eslog(nus)
        warn_resolution(resolution)

    elif check_scale_xsmode(xsmode) == 'ESLIN':
        if unit == 'cm-1':
            nus = np.linspace((x0), (x1), N, dtype=np.float64)
            wav = nu2wav(nus)
        elif unit == 'nm' or unit == 'AA':
            cx1, cx0 = wav2nu(np.array([x0, x1]), unit)
            nus = np.linspace((cx0), (cx1), N, dtype=np.float64)
            wav = nu2wav(nus, unit)

        minr, resolution, maxr = resolution_eslin(nus)
        warn_resolution(minr)

    return nus, wav, resolution


def pressure_layer(logPtop=-8., logPbtm=2., NP=20, mode='ascending'):
    """generating the pressure layer.

    Args:
       logPtop: log10(P[bar]) at the top layer
       logPbtm: log10(P[bar]) at the bottom layer
       NP: the number of the layers

    Returns:
         Parr: pressure layer
         dParr: delta pressure layer
         k: k-factor, P[i-1] = k*P[i]

    Note:
        dParr[i] = Parr[i] - Parr[i-1], dParr[0] = (1-k) Parr[0] for ascending mode
    """
    dlogP = (logPbtm-logPtop)/(NP-1)
    k = 10**-dlogP
    Parr = jnp.logspace(logPtop, logPbtm, NP)
    dParr = (1.0-k)*Parr
    if mode == 'descending':
        Parr = Parr[::-1]
        dParr = dParr[::-1]

    return jnp.array(Parr), jnp.array(dParr), k


def dtauCIA(nus, Tarr, Parr, dParr, vmr1, vmr2, mmw, g, nucia, tcia, logac):
    """dtau of the CIA continuum.

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmr1: volume mixing ratio (VMR) for molecules 1 [N_layer]
       vmr2: volume mixing ratio (VMR) for molecules 2 [N_layer]
       mmw: mean molecular weight of atmosphere
       g: gravity (cm2/s)
       nucia: wavenumber array for CIA
       tcia: temperature array for CIA
       logac: log10(absorption coefficient of CIA)

    Returns:
       optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(Parr, Tarr)
    lognarr1 = jnp.log10(vmr1*narr)  # log number density
    lognarr2 = jnp.log10(vmr2*narr)  # log number density
    logkb = np.log10(kB)
    logg = jnp.log10(g)
    ddParr = dParr/Parr
    dtauc = (10**(logacia(Tarr, nus, nucia, tcia, logac)
                  + lognarr1[:, None]+lognarr2[:, None]+logkb-logg-logm_ucgs)
             * Tarr[:, None]/mmw*ddParr[:, None])

    return dtauc


def dtauCIA_mmwl(nus, Tarr, Parr, dParr, vmr1, vmr2, mmw, g, nucia, tcia, logac):
    """dtau of the CIA continuum.
       (for the case where mmw is given for each atmospheric layer)

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmr1: volume mixing ratio (VMR) for molecules 1 [N_layer]
       vmr2: volume mixing ratio (VMR) for molecules 2 [N_layer]
       mmw: mean molecular weight of atmosphere [N_layer]
       g: gravity (cm2/s)
       nucia: wavenumber array for CIA
       tcia: temperature array for CIA
       logac: log10(absorption coefficient of CIA)

    Returns:
       optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(Parr, Tarr)
    lognarr1 = jnp.log10(vmr1*narr)  # log number density
    lognarr2 = jnp.log10(vmr2*narr)  # log number density
    logkb = np.log10(kB)
    logg = jnp.log10(g)
    ddParr = dParr/Parr
    dtauc = (10**(logacia(Tarr, nus, nucia, tcia, logac)
                  + lognarr1[:, None]+lognarr2[:, None]+logkb-logg-logm_ucgs)
             * Tarr[:, None]/mmw[:, None]*ddParr[:, None])

    return dtauc


def dtauM(dParr, xsm, MR, mass, g):
    """dtau of the molecular cross section.

    Note:
       fac=bar_cgs/(m_u (g)). m_u: atomic mass unit. It can be obtained by fac=1.e3/m_u, where m_u = scipy.constants.m_u.

    Args:
       dParr: delta pressure profile (bar) [N_layer]
       xsm: cross section matrix (cm2) [N_layer, N_nus]
       MR: volume mixing ratio (VMR) or mass mixing ratio (MMR) [N_layer]
       mass: mean molecular weight for VMR or molecular mass for MMR
       g: gravity (cm/s2)

    Returns:
       optical depth matrix [N_layer, N_nus]
    """

    fac = 6.022140858549162e+29
    return fac*xsm*dParr[:, None]*MR[:, None]/(mass*g)


def dtauM_mmwl(dParr, xsm, MR, mass, g):
    """dtau of the molecular cross section.
       (for the case where mmw is given for each atmospheric layer)

    Note:
       fac=bar_cgs/(m_u (g)). m_u: atomic mass unit. It can be obtained by fac=1.e3/m_u, where m_u = scipy.constants.m_u.

    Args:
       dParr: delta pressure profile (bar) [N_layer]
       xsm: cross section matrix (cm2) [N_layer, N_nus]
       MR: volume mixing ratio (VMR) or mass mixing ratio (MMR) [N_layer]
       mass: mean molecular weight for VMR or molecular mass for MMR [N_layer]
       g: gravity (cm/s2)

    Returns:
       optical depth matrix [N_layer, N_nus]
    """

    fac = 6.022140858549162e+29
    return fac*xsm*dParr[:, None]*MR[:, None]/(mass[:, None]*g)


@jit
def dtauVALD(dParr, xsm, VMR, mmw, g):
    """dtau of the atomic (+ionic) cross section from VALD.

    Args:
       dParr: delta pressure profile (bar) [N_layer]
       xsm: cross section matrix (cm2) [N_species x N_layer x N_wav]
       VMR: volume mixing ratio [N_species x N_layer]
       mmw: mean molecular weight [N_layer]
       g: gravity (cm/s2)

    Returns:
        dtau: optical depth matrix [N_layer x N_nus]
    """
    dtauS = jit(vmap(dtauM_mmwl, (None, 0, 0, None, None)))( \
                            dParr, xsm, VMR, mmw, g)
    dtau = jnp.abs(jnp.sum(dtauS, axis=0))
    return dtau
    
    
def dtauHminus(nus, Tarr, Parr, dParr, vmre, vmrh, mmw, g):
    """dtau of the H- continuum.

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmre: volume mixing ratio (VMR) for e- [N_layer]
       vmrH: volume mixing ratio (VMR) for H atoms [N_layer]
       mmw: mean molecular weight of atmosphere
       g: gravity (cm2/s)

    Returns:
       optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(Parr, Tarr)
    #       number_density_e: number density for e- [N_layer]
    #       number_density_h: number density for H atoms [N_layer]
    number_density_e = vmre*narr
    number_density_h = vmrh*narr
    logkb = np.log10(kB)
    logg = jnp.log10(g)
    ddParr = dParr/Parr
    logabc = (log_hminus_continuum(
        nus, Tarr, number_density_e, number_density_h))
    dtauh = 10**(logabc+logkb-logg-logm_ucgs)*Tarr[:, None]/mmw*ddParr[:, None]

    return dtauh


def dtauHminus_mmwl(nus, Tarr, Parr, dParr, vmre, vmrh, mmw, g):
    """dtau of the H- continuum.
       (for the case where mmw is given for each atmospheric layer)

    Args:
       nus: wavenumber matrix (cm-1)
       Tarr: temperature array (K)
       Parr: temperature array (bar)
       dParr: delta temperature array (bar)
       vmre: volume mixing ratio (VMR) for e- [N_layer]
       vmrH: volume mixing ratio (VMR) for H atoms [N_layer]
       mmw: mean molecular weight of atmosphere [N_layer]
       g: gravity (cm2/s)

    Returns:
       optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(Parr, Tarr)
    #       number_density_e: number density for e- [N_layer]
    #       number_density_h: number density for H atoms [N_layer]
    number_density_e = vmre*narr
    number_density_h = vmrh*narr
    logkb = np.log10(kB)
    logg = jnp.log10(g)
    ddParr = dParr/Parr
    logabc = (log_hminus_continuum(
        nus, Tarr, number_density_e, number_density_h))
    dtauh = 10**(logabc+logkb-logg-logm_ucgs)*Tarr[:, None]/mmw[:, None]*ddParr[:, None]

    return dtauh


@jit
def trans2E3(x):
    """transmission function 2E3 (two-stream approximation with no scattering)
    expressed by 2 E3(x)

    Note:
       The exponetial integral of the third order E3(x) is computed using Abramowitz Stegun (1970) approximation of E1 (exojax.special.E1).

    Args:
       x: input variable

    Returns:
       Transmission function T=2 E3(x)
    """
    return (1.0-x)*jnp.exp(-x) + x**2*E1(x)


@jit
def rtrun(dtau, S):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type)

    Args:
        dtau: opacity matrix
        S: source matrix [N_layer, N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus = jnp.shape(dtau)[1]
    TransM = jnp.where(dtau == 0, 1.0, trans2E3(dtau))
    Qv = jnp.vstack([(1-TransM)*S, jnp.zeros(Nnus)])
    return jnp.sum(Qv*jnp.cumprod(jnp.vstack([jnp.ones(Nnus), TransM]), axis=0), axis=0)


@jit
def rtrun_surface(dtau, S, Sb):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type)
    with a planetary surface.

    Args:
        dtau: opacity matrix
        S: source matrix [N_layer, N_nus]
        Sb: source from the surface [N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus = jnp.shape(dtau)[1]
    TransM = jnp.where(dtau == 0, 1.0, trans2E3(dtau))
    Qv = jnp.vstack([(1-TransM)*S, Sb])
    return jnp.sum(Qv*jnp.cumprod(jnp.vstack([jnp.ones(Nnus), TransM]), axis=0), axis=0)


@jit
def rtrun_direct(dtau, S):
    """Radiative Transfer using direct integration.

    Note:
        Use dtau/mu instead of dtau when you want to use non-unity, where mu=cos(theta)

    Args:
        dtau: opacity matrix
        S: source matrix [N_layer, N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    taupmu = jnp.cumsum(dtau, axis=0)
    return jnp.sum(S*jnp.exp(-taupmu)*dtau, axis=0)
