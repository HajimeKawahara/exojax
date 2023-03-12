""" radiative transfer
"""
from jax import jit
import jax.numpy as jnp
from exojax.special.expn import E1
import warnings
from exojax.spec.layeropacity import layer_optical_depth
from exojax.spec.layeropacity import layer_optical_depth_CIA
from exojax.spec.layeropacity import layer_optical_depth_Hminus
from exojax.spec.layeropacity import layer_optical_depth_VALD

##################################################################################
# deprecated features, will be removed by Release v2.0
##################################################################################


def dtauM(dParr, xsm, MR, mass, g):
    warn_msg = "Use `spec.layeropacity.layer_optical_depth` instead"
    warnings.warn(warn_msg, FutureWarning)
    return layer_optical_depth(dParr, xsm, MR, mass, g)


def dtauCIA(nus, Tarr, Parr, dParr, vmr1, vmr2, mmw, g, nucia, tcia, logac):
    warn_msg = "Use `spec.layeropacity.layer_optical_depth_CIA` instead"
    warnings.warn(warn_msg, FutureWarning)
    return layer_optical_depth_CIA(nus, Tarr, Parr, dParr, vmr1, vmr2, mmw, g,
                                   nucia, tcia, logac)


def dtauHminus(nus, Tarr, Parr, dParr, vmre, vmrh, mmw, g):
    warn_msg = "Use `spec.layeropacity.layer_optical_depth_Hminus` instead"
    warnings.warn(warn_msg, FutureWarning)
    return layer_optical_depth_Hminus(nus, Tarr, Parr, dParr, vmre, vmrh, mmw,
                                      g)


def dtauVALD(dParr, xsm, VMR, mmw, g):
    warn_msg = "Use `spec.layeropacity.layer_optical_depth_VALD` instead"
    warnings.warn(warn_msg, FutureWarning)
    return layer_optical_depth_VALD(dParr, xsm, VMR, mmw, g)


def pressure_layer(log_pressure_top=-8.,
                   log_pressure_btm=2.,
                   NP=20,
                   mode='ascending',
                   numpy=False):
    warn_msg = "Use `atm.atmprof.pressure_layer_logspace` instead"
    warnings.warn(warn_msg, FutureWarning)
    from exojax.atm.atmprof import pressure_layer_logspace
    return pressure_layer_logspace(log_pressure_top, log_pressure_btm, NP,
                                   mode, numpy)


def rtrun(dtau, S):
    warnings.warn("Use rtrun_emis_pure_absorption instead", FutureWarning)
    return rtrun_emis_pure_absorption(dtau, S)


##########################################################################################


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
    return (1.0 - x) * jnp.exp(-x) + x**2 * E1(x)


@jit
def rtrun_emis_pure_absorption(dtau, source_matrix):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type)

    Args:
        dtau: opacity matrix
        source_matrix: source matrix [N_layer, N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus = jnp.shape(dtau)[1]
    TransM = jnp.where(dtau == 0, 1.0, trans2E3(dtau))
    Qv = jnp.vstack([(1 - TransM) * source_matrix, jnp.zeros(Nnus)])
    return jnp.sum(Qv *
                   jnp.cumprod(jnp.vstack([jnp.ones(Nnus), TransM]), axis=0),
                   axis=0)


@jit
def rtrun_emis_pure_absorption_surface(dtau, source_matrix, source_surface):
    """Radiative Transfer using two-stream approximaion + 2E3 (Helios-R1 type)
    with a planetary surface.

    Args:
        dtau: opacity matrix
        source_matrix: source matrix [N_layer, N_nus]
        source_surface: source from the surface [N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus = jnp.shape(dtau)[1]
    TransM = jnp.where(dtau == 0, 1.0, trans2E3(dtau))
    Qv = jnp.vstack([(1 - TransM) * source_matrix, source_surface])
    return jnp.sum(Qv *
                   jnp.cumprod(jnp.vstack([jnp.ones(Nnus), TransM]), axis=0),
                   axis=0)


@jit
def rtrun_emis_pure_absorption_direct(dtau, source_matrix):
    """Radiative Transfer using direct integration.

    Note:
        Use dtau/mu instead of dtau when you want to use non-unity, where mu=cos(theta)

    Args:
        dtau: opacity matrix
        source_matrix: source matrix [N_layer, N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    taupmu = jnp.cumsum(dtau, axis=0)
    return jnp.sum(source_matrix * jnp.exp(-taupmu) * dtau, axis=0)


def rtrun_trans_pure_absorption(dtau_chord, radius_chord):
    """Radiative transfer assuming pure absorption 

    Args:
        dtau_chord (2D array): chord opacity (Nlayer, N_wavenumber)
        radius_chord (1D array): (normalized) radius (Nlayer), (radius[1], radius[2], ..., radius_btm)
        
    Returns:
        1D array: transit squared radius in the same unit as sqaure of the radius/radius_btm

    Notes:
        This function gives the sqaure of the transit radius.
        If you would like to obtain the transit radius, take sqaure root of the output.
        If you would like to compute the transit depth, devide the output by the square of stellar radius

    """
    deltaRp2 = 2.0 * jnp.trapz(
        (1.0 - jnp.exp(-dtau_chord)) * radius_chord[::-1, None],
        x=radius_chord[::-1],
        axis=0)
    return deltaRp2 + radius_chord[-1]**2
