""" radiative transfer
"""
import warnings
from jax import jit
import jax.numpy as jnp
from exojax.special.expn import E1
from exojax.spec.layeropacity import layer_optical_depth
from exojax.spec.layeropacity import layer_optical_depth_CIA
from exojax.spec.layeropacity import layer_optical_depth_Hminus
from exojax.spec.layeropacity import layer_optical_depth_VALD
from jax import vmap


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
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix (2D array): source matrix (N_layer, N_nus)

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
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix (2D array): source matrix (N_layer, N_nus)
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
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix (2D array): source matrix (N_layer, N_nus)

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    taupmu = jnp.cumsum(dtau, axis=0)
    return jnp.sum(source_matrix * jnp.exp(-taupmu) * dtau, axis=0)


def rtrun_trans_pure_absorption(dtau_chord, radius_lower):
    """Radiative transfer assuming pure absorption 

    Args:
        dtau_chord (2D array): chord opacity (Nlayer, N_wavenumber)
        radius_lower (1D array): (normalized) radius at the lower boundary, underline(r) (Nlayer). 

    Notes:
        The n-th radius is defined as the lower boundary of the n-th layer. So, radius[0] corresponds to R0.   
        
    Returns:
        1D array: transit squared radius in the same unit as sqaure of the radius/radius_btm

    Notes:
        This function gives the sqaure of the transit radius.
        If you would like to obtain the transit radius, take sqaure root of the output.
        If you would like to compute the transit depth, devide the output by the square of stellar radius

    """
    deltaRp2 = 2.0 * jnp.trapz(
        (1.0 - jnp.exp(-dtau_chord)) * radius_lower[::-1, None],
        x=radius_lower[::-1],
        axis=0)
    return deltaRp2 + radius_lower[-1]**2


from exojax.spec.twostream import set_scat_trans_coeffs
from exojax.spec.twostream import compute_tridiag_diagonals
from exojax.spec.toon import zetalambda_coeffs
from exojax.spec.toon import params_hemispheric_mean
from exojax.spec.toon import reduced_source_function
from exojax.linalg.tridiag import solve_tridiag


def debug_imshow_ts(val1, val2, title1, title2):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_title(title1)
    a = plt.imshow(val1[:, 10300:10700])
    plt.colorbar(a, shrink=0.5)
    ax = fig.add_subplot(212)
    ax.set_title(title2)
    a = plt.imshow(val2[:, 10300:10700])
    plt.colorbar(a, shrink=0.5)
    plt.show()


def rtrun_emis_scat_toon_hemispheric_mean(dtau, single_scattering_albedo,
                                          asymmetric_parameter, source_matrix):

    gamma_1, gamma_2, mu1 = params_hemispheric_mean(single_scattering_albedo,
                                                    asymmetric_parameter)
    zeta_plus, zeta_minus, lambdan = zetalambda_coeffs(gamma_1, gamma_2)
    trans_coeff, scat_coeff = set_scat_trans_coeffs(zeta_plus, zeta_minus,
                                                    lambdan, dtau)
    #delta = 1.e-10
    delta = 0.0
    trans_coeff = trans_coeff + delta

    #debug
    debug_imshow_ts(jnp.log10(trans_coeff), jnp.log10(scat_coeff),
                    "Transmission Coefficient $\mathcal{T}$",
                    "Scattering Coefficient $\mathcal{S}$")

    #temporary
    source_function_derivative = jnp.zeros_like(source_matrix)

    piBplus = reduced_source_function(single_scattering_albedo, gamma_1,
                                      gamma_2, source_matrix,
                                      source_function_derivative, 1.0)

    piBminus = reduced_source_function(single_scattering_albedo, gamma_1,
                                       gamma_2, source_matrix,
                                       source_function_derivative, -1.0)

    debug_imshow_ts(piBplus, piBminus, "piBplus", "piBminus")

    # Boundary condition
    ## top layer
    fac = 1.e10
    diagonal_top = -1.0 * fac  #b0
    #diagonal_top = 0.0  #debug
    upper_diagonal_top = trans_coeff[0] * fac
    #upper_diagonal_top = 0.0 #debug
    vector = jnp.zeros_like(dtau)


    vector = vector.at[0,:].set(scat_coeff[0] * piBminus[0] * fac)

    ##### KOREGA HEIRETU NI NATTENAIIYO
    print()
    print("KOREGA HEIRETU NI NATTENAIIYO")
    diagonal, lower_diagonal, upper_diagonal = compute_tridiag_diagonals(
        scat_coeff, trans_coeff, upper_diagonal_top, diagonal_top)

    #bottom layer
    Fs = 0.0
    vector = vector.at[-1,:].set(- upper_diagonal[-1]*Fs * fac)  


    debug_imshow_ts(diagonal, jnp.log10(jnp.abs(diagonal)), "diagonal",
                    "log abs diagonal")
    debug_imshow_ts(upper_diagonal, lower_diagonal, "upper diagonal",
                    "lower diagonal")

    #print(diagonal, lower_diagonal, upper_diagonal)

    vmap_solve_tridiag = vmap(solve_tridiag, (0, 0, 0, 0), 0)
    canonical_flux_upward = vmap_solve_tridiag(diagonal.T[0:20000, 0:100],
                                               lower_diagonal.T[0:20000, 0:99],
                                               upper_diagonal.T[0:20000, 0:99],
                                               vector.T[0:20000, 0:100])

    #check tridiagonal solver's result
    iwav = 10450
    nwav  = 20000
    nlayer = 100
    fp = canonical_flux_upward[iwav, :]
    di = diagonal.T[iwav, :]
    li = lower_diagonal.T[iwav, :]
    ui = upper_diagonal.T[iwav, :]
    vi = vector.T[iwav, :]

    recovered_vector = jnp.zeros((nwav,nlayer))
    recovered_vector = recovered_vector.at[0].set(di[0] * fp[0] + ui[0] * fp[1])

    print(di[0] * fp[0] + ui[0] * fp[1], vi[0], "<=")
    print(
        li[0:nlayer - 2] * fp[0:nlayer - 2] +
        di[1:nlayer - 1] * fp[1:nlayer - 1] + ui[1:nlayer - 1] * fp[2:nlayer],
        vi[1:nlayer - 1], "<=")
    print(li[nlayer - 2] * fp[nlayer - 2] + di[nlayer - 1] * fp[nlayer - 1],
          vi[nlayer - 1], "<=")
    ####

    Fplus = canonical_flux_upward.T + piBplus
    debug_imshow_ts(canonical_flux_upward.T, piBplus, "f+", "piB+")
    #canonical_flux_upward = vmap_solve_tridiag(diagonal, lower_diagonal,
    #                                      upper_diagonal, vector)
    #Fplus = canonical_flux_upward  #+ piBplus

    import matplotlib.pyplot as plt
    plt.plot(Fplus[0, 10300:10700])
    plt.plot(Fplus[1, 10300:10700])
    plt.plot(Fplus[2, 10300:10700])

    plt.show()
    Ftop = Fplus[0, :]
    return Ftop


##################################################################################
# Raise Error since v1.5
# Deprecated features, will be completely removed by Release v2.0
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
                   reference_point=0.5,
                   numpy=False):
    warn_msg = "Use `atm.atmprof.pressure_layer_logspace` instead"
    warnings.warn(warn_msg, FutureWarning)
    from exojax.atm.atmprof import pressure_layer_logspace
    return pressure_layer_logspace(log_pressure_top, log_pressure_btm, NP,
                                   mode, reference_point, numpy)


def rtrun(dtau, S):
    warnings.warn("Use rtrun_emis_pure_absorption instead", FutureWarning)
    return rtrun_emis_pure_absorption(dtau, S)


##########################################################################################
