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
        1D array: transit squared radius normalized by radius_lower[-1], i.e. it returns (radius/radius_lower[-1])**2

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
from exojax.spec.twostream import compute_tridiag_diagonals_and_vector
from exojax.spec.toon import zetalambda_coeffs
from exojax.spec.toon import params_hemispheric_mean
from exojax.spec.toon import reduced_source_function
from exojax.linalg.tridiag import solve_tridiag
#from exojax.linalg.tridiag import solve_vmap_semitridiag_naive as vmap_solve_tridiag
from exojax.linalg.tridiag import solve_vmap_semitridiag_naive_array as vmap_solve_tridiag


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
    delta = 1.e-9
    #delta = 0.0
    trans_coeff = trans_coeff + delta

    print(jnp.prod(trans_coeff, axis=0))
    print(jnp.sum(jnp.log(trans_coeff), axis=0))

    #import sys
    #sys.exit()

    #debug
    debug = True
    if debug:
        debug_imshow_ts((trans_coeff), (scat_coeff),
                        "Transmission Coefficient $\mathcal{T}$",
                        "Scattering Coefficient $\mathcal{S}$")
        debug_imshow_ts(jnp.log10(trans_coeff), jnp.log10(scat_coeff),
                        "Transmission Coefficient $\mathcal{T} log$",
                        "Scattering Coefficient $\mathcal{S} log$")

    #temporary
    source_function_derivative = jnp.zeros_like(source_matrix)

    piBplus = reduced_source_function(single_scattering_albedo, gamma_1,
                                      gamma_2, source_matrix,
                                      source_function_derivative, 1.0)

    piBminus = reduced_source_function(single_scattering_albedo, gamma_1,
                                       gamma_2, source_matrix,
                                       source_function_derivative, -1.0)

    if debug:
        debug_imshow_ts(piBplus, piBminus, "piBplus", "piBminus")

    # Boundary condition
    diagonal_top = 1.0 * jnp.ones_like(trans_coeff[0, :])  #b0
    upper_diagonal_top = trans_coeff[0, :]
    vector_top = piBplus[
        0, :]   - trans_coeff[0, :] * piBplus[1, :] - scat_coeff[0, :] * piBminus[0, :]

    #tridiagonal elements
    diagonal, lower_diagonal, upper_diagonal, vector = compute_tridiag_diagonals_and_vector(
        scat_coeff, trans_coeff, piBplus, upper_diagonal_top, diagonal_top,
        vector_top)

    if debug:
        debug_imshow_ts(jnp.log10(upper_diagonal), jnp.log10(lower_diagonal),
                        "log(a)", "log(c)")
        debug_imshow_ts(jnp.log10(diagonal), jnp.log10(vector), "log(b)",
                        "log(d)")

        debug_imshow_ts(vector, jnp.log10((vector)), "vector", "log  vector")
        debug_imshow_ts(diagonal, jnp.log10(jnp.abs(diagonal)), "diagonal",
                        "log abs diagonal")
        debug_imshow_ts(upper_diagonal, lower_diagonal, "upper diagonal",
                        "lower diagonal")
        debug_imshow_ts(jnp.log10(jnp.abs(upper_diagonal)),
                        jnp.log10(jnp.abs(lower_diagonal)),
                        "log upper diagonal", "log lower diagonal")

    #### TEST IMPLEMENTATION
    import numpy as np
    nlayer, nwav = diagonal.shape
    Ttilde = np.zeros_like(trans_coeff)
    Qtilde = np.zeros_like(trans_coeff)

    Ttilde[0, :] = upper_diagonal[0, :] / diagonal[0, :]
    Qtilde[0, :] = vector[0, :] / diagonal[0, :]

    for i in range(1, nlayer):
        gamma = diagonal[i, :] - lower_diagonal[i - 1, :] * Ttilde[i - 1, :]
        Ttilde[i, :] = upper_diagonal[i, :] / gamma
        Qtilde[i, :] = (vector[i, :] +
                        lower_diagonal[i - 1, :] * Qtilde[i - 1, :]) / gamma
        
    Qpure0 = np.zeros_like(Qtilde)
    Qpure1 = np.zeros_like(Qtilde)

    for i in range(0, nlayer - 1):
        Qpure1[i, :] = piBplus[i, :] - trans_coeff[i, :] * piBplus[i + 1, :]
        Qpure0[i, :] = (1.0 - trans_coeff[i, :])*piBplus[i, :]  

    print("1",Qpure1[0,:])
    print("tilde",Qtilde[0,:])
    
    print("1",Qpure1[1,:])
    print("tilde",Qtilde[1,:])
    
    #negative removal
    Qpure1[Qpure1<0.0] = 0.0
    Qtilde[Qtilde<0.0] = 0.0

    debug_imshow_ts((Ttilde), (trans_coeff), "Ttilde", "trans")
    Ttilde = np.array(Ttilde)
    debug_imshow_ts((Qtilde), (Qpure1), "Qtilde", "Qpure")
    
    debug_imshow_ts((Qpure0/Qtilde), (Qpure1/Qtilde), "0/tilde", "1/tilde")
    

    #Ttilde[Ttilde > 1.0] = 1.0
    cumTtilde = np.cumprod(Ttilde, axis=0)
    debug_imshow_ts((Qtilde), cumTtilde, "Qtilde", "cumprod T")
    contri = cumTtilde * Qtilde
    debug_imshow_ts(np.log10(contri), contri, "log (cumprod T)*Qtilde",
                    "(cumprod T)*Qtilde")

    spectrum = np.nansum(cumTtilde * Qtilde, axis=0)

    cumTpure = np.cumprod(trans_coeff, axis=0)
    spectrum_tpure = np.nansum(cumTtilde * Qpure0, axis=0)
    spectrum_pure = np.nansum(cumTpure * Qpure0, axis=0)
    spectrum_pure1 = np.nansum(cumTpure * Qpure1, axis=0)
    spectrum_pt = np.nansum(cumTpure * Qtilde, axis=0)
    spectrum = np.nansum(cumTtilde * Qtilde, axis=0)
    
    import matplotlib.pyplot as plt
    plt.plot(spectrum,label="tilde * tilde ")
    plt.plot(spectrum_pt,label="pure * tilde ")
    plt.plot(spectrum_pure, label="pure * pure")
    plt.plot(spectrum_pure1, label="pure * pure1")
    plt.plot(spectrum_tpure, label="tilde * pure")
    
    plt.legend()
    plt.show()

    #debug_imshow_ts(np.log10(Ttilde), np.log10(Qtilde), "Ttilde", "Qtilde")

    import sys
    sys.exit()
    #print(diagonal, lower_diagonal, upper_diagonal)

    # Using complete tridiag solver
    #vmap_solve_tridiag = vmap(solve_tridiag, (0, 0, 0, 0), 0)
    beta, delta = vmap_solve_tridiag(diagonal.T[0:20000, 0:nlayer],
                                     lower_diagonal.T[0:20000, 0:nlayer - 1],
                                     upper_diagonal.T[0:20000, 0:nlayer - 1],
                                     vector.T[0:20000, 0:nlayer])
    import numpy as np
    #beta[np.abs(beta)>100.0]=0.0
    #delta[np.abs(delta)>100.0]=0.0

    debug_imshow_ts(np.log10(np.abs(beta.T)), np.log10(np.abs(delta.T)),
                    "beta", "delta")

    flux_upward = delta[:, 1]

    import matplotlib.pyplot as plt
    plt.plot(flux_upward[10300:10700])
    plt.show()
    return flux_upward


def manual_recover_tridiagonal(diagonal, lower_diagonal, upper_diagonal,
                               canonical_flux_upward, iwav):
    import numpy as np
    nlayer, nwav = diagonal.shape
    fp = canonical_flux_upward[iwav, :]
    di = diagonal.T[iwav, :]
    li = lower_diagonal.T[iwav, :]
    ui = upper_diagonal.T[iwav, :]

    recovered_vector = jnp.zeros((nwav, nlayer))
    recovered_vector = recovered_vector.at[0].set(di[0] * fp[0] +
                                                  ui[0] * fp[1])

    head = di[0] * fp[0] + ui[0] * fp[1]
    manual = list(li[0:nlayer - 2] * fp[0:nlayer - 2] +
                  di[1:nlayer - 1] * fp[1:nlayer - 1] +
                  ui[1:nlayer - 1] * fp[2:nlayer])
    end = li[nlayer - 2] * fp[nlayer - 2] + di[nlayer - 1] * fp[nlayer - 1]
    manual.insert(0, head)
    manual.append(end)
    manual = np.array(manual)
    return manual


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
