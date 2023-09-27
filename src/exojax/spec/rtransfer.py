""" Runs radiative transfer

    The classification of rtrun(s):

    - flux-based emission
    -- pure absoprtion 
    --- 2stream: rtrun_emis_pureabs_flux2st, rtrun_emis_pureabs_flux2st_surface
    -- scattering
    --- 2stream
    ---- LART: rtrun_emis_scat_lart_toonhm
    - intensity-based emission
    - transmision: rtrun_trans_pureabs




"""
from jax import jit
import jax.numpy as jnp
from jax.lax import scan
from exojax.spec.twostream import solve_lart_twostream_numpy
from exojax.spec.twostream import solve_lart_twostream

from exojax.spec.toon import reduced_source_function_isothermal_layer
from exojax.spec.toon import params_hemispheric_mean
from exojax.spec.toon import zetalambda_coeffs
from exojax.spec.twostream import compute_tridiag_diagonals_and_vector
from exojax.spec.twostream import set_scat_trans_coeffs
from exojax.special.expn import E1
from exojax.spec.layeropacity import layer_optical_depth
from exojax.spec.layeropacity import layer_optical_depth_CIA
from exojax.spec.layeropacity import layer_optical_depth_Hminus
from exojax.spec.layeropacity import layer_optical_depth_VALD
import warnings


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
def rtrun_emis_pureabs_fbased2st(dtau, source_matrix):
    """Radiative Transfer for emission spectrum using flux-based two-stream pure absorption with no surface
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
def rtrun_emis_pureabs_fbased2st_surface(dtau, source_matrix, source_surface):
    """Radiative Transfer for emission spectrum using flux-based two-stream pure absorption with a planetary surface.

    Args:
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix (2D array): source matrix (N_layer, N_nus)
        source_surface: source from the surface [N_nus]

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    Nnus = jnp.shape(dtau)[1]
    trans = jnp.where(dtau == 0, 1.0, trans2E3(dtau))
    Qv = jnp.vstack([(1 - trans) * source_matrix, source_surface])
    return jnp.sum(Qv *
                   jnp.cumprod(jnp.vstack([jnp.ones(Nnus), trans]), axis=0),
                   axis=0)


@jit
def rtrun_emis_pureabs_ibased(dtau, source_matrix, mus, weights):
    """Radiative Transfer for emission spectrum using intensity-based n-stream pure absorption with no surface (NEMESIS, pRT-like)
    Args:
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix (2D array): source matrix (N_layer, N_nus)
        mus (list): mu (cos theta) list for integration
        weights (list): weight list for mu
        
    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """

    Nnus = jnp.shape(dtau)[1]
    tau = jnp.cumsum(dtau, axis=0)

    #The following scan part is equivalent to this for-loop
    #spec = jnp.zeros(Nnus)
    #for i, mu in enumerate(mus):
    #    dtrans = - jnp.diff(jnp.exp(-tau/mu), prepend=1.0, axis=0)
    #    spec = spec + weights[i]*2.0*mu*jnp.sum(source_matrix*dtrans, axis=0)

    #scan part
    muws = [mus, weights]

    def f(carry_fmu, muw):
        mu, w = muw
        dtrans = -jnp.diff(jnp.exp(-tau / mu), prepend=1.0, axis=0)
        carry_fmu = carry_fmu + 2.0 * mu * w * jnp.sum(source_matrix * dtrans,
                                                       axis=0)
        return carry_fmu, None

    spec, _ = scan(f, jnp.zeros(Nnus), muws)

    return spec


def initialize_gaussian_quadrature(nstream):
    from scipy.special import roots_legendre
    if nstream % 2 == 0:
        norder = int(nstream / 2)
    else:
        raise ValueError("nstream should be even number larger than 2.")
    mus, weights = roots_legendre(norder)

    #correction because integration should be between 0 to 1, but roots_legendre uses -1 to 1.
    mus = 0.5 * (mus + 1.0)
    weights = 0.5 * weights
    print("Gaussian Quadrature Parameters: ")
    print("mu = ", mus)
    print("weight =", weights)

    return mus, weights


def rtrun_emis_pureabs_ibased_linsap(dtau, source_matrix_up, mus,
                                     weights):
    """Radiative Transfer for emission spectrum using intensity-based n-stream pure absorption with no surface w/ linear source approximation = linsap (HELIOS-R2 like)

    Args:
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix_up (2D array): source matrix at the layer upper boundary (N_layer, N_nus)
        mus (list): mu (cos theta) list for integration
        weights (list): weight list for mu
        
    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.

    Notes:
        See Olson and Kunasz as well as HELIOS-R2 paper (Kitzmann+) for the derivation.
    
    
    """

    Nnus = jnp.shape(dtau)[1]
    source_matrix_layertop_p1 = jnp.roll(source_matrix_up, -1,
                                         axis=0)  # S_{n+1}


    # NOT IMPLEMENTED YET
    # need to replace the last element of the above
    #

    # no surface
    BT_B = jnp.zeros(Nnus)

    #scan part
    muws = [mus, weights]
    
    
    def f(carry_fmu, muw):
        mu, w = muw

        dtau_per_mu = dtau/mu
        trans = jnp.exp(-dtau_per_mu) # hat{T}
        beta, gamma = coeffs_linsap(dtau_per_mu, trans)
        dI = jnp.vstack([beta * source_matrix_up + gamma * source_matrix_layertop_p1, BT_B])

        intensity_for_mu = jnp.sum(dI *
                    jnp.cumprod(jnp.vstack([jnp.ones(Nnus), trans]), axis=0),
                    axis=0)
        
        carry_fmu = carry_fmu + 2.0 * mu * w * intensity_for_mu

        return carry_fmu, None

    spec, _ = scan(f, jnp.zeros(Nnus), muws)
    return spec

def coeffs_linsap(dtau_per_mu, trans):
    """coefficients of the linsap

    Args:
        dtau_per_mu (_type_): opacity difference divided by mu (cos theta)
        trans: transmission of the layers
    Returns:
        _type_: beta coefficient, gamma coefficient
    """
    fac = (1.0 - trans) / dtau_per_mu
    beta = 1.0 - fac
    gamma = -trans + fac
    return beta, gamma


@jit
def rtrun_trans_pureabs(dtau_chord, radius_lower):
    """Radiative transfer for transmission spectrum assuming pure absorption 

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


@jit
def rtrun_emis_scat_lart_toonhm(dtau, single_scattering_albedo,
                                asymmetric_parameter, source_matrix):
    """Radiative Transfer for emission spectrum using flux-based two-stream scattering LART solver w/ Toon Hemispheric Mean with no surface.

    Args:
        dtau (_type_): _description_
        single_scattering_albedo (_type_): _description_
        asymmetric_parameter (_type_): _description_
        source_matrix (_type_): _description_
        
    Returns:
        _type_: _description_
    """
    trans_coeff, scat_coeff, piB, diagonal, lower_diagonal, upper_diagonal, vector = setrt_toonhm(
        dtau, single_scattering_albedo, asymmetric_parameter, source_matrix)

    nlayer, Nnus = diagonal.shape
    cumTtilde, Qtilde, spectrum = solve_lart_twostream(diagonal,
                                                       lower_diagonal,
                                                       upper_diagonal, vector,
                                                       jnp.zeros(Nnus))

    return spectrum, cumTtilde, Qtilde, trans_coeff, scat_coeff, piB


@jit
def rtrun_emis_scat_lart_toonhm_surface(dtau, single_scattering_albedo,
                                        asymmetric_parameter, source_matrix,
                                        source_surface):
    """Radiative Transfer for emission spectrum using flux-based two-stream scattering LART solver w/ Toon Hemispheric Mean with surface.

    Args:
        dtau (_type_): _description_
        single_scattering_albedo (_type_): _description_
        asymmetric_parameter (_type_): _description_
        source_matrix (_type_): _description_
        source_surface: source from the surface (N_nus)

    Returns:
        _type_: _description_
    """
    trans_coeff, scat_coeff, piB, diagonal, lower_diagonal, upper_diagonal, vector = setrt_toonhm(
        dtau, single_scattering_albedo, asymmetric_parameter, source_matrix)

    nlayer, Nnus = diagonal.shape
    cumTtilde, Qtilde, spectrum = solve_lart_twostream(diagonal,
                                                       lower_diagonal,
                                                       upper_diagonal, vector,
                                                       source_surface)

    return spectrum, cumTtilde, Qtilde, trans_coeff, scat_coeff, piB


def setrt_toonhm(dtau, single_scattering_albedo, asymmetric_parameter,
                 source_matrix):
    """sets rt for rtrun assming Toon Hemispheric Mean 

    Args:
        dtau (_type_): _description_
        single_scattering_albedo (_type_): _description_
        asymmetric_parameter (_type_): _description_
        source_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    gamma_1, gamma_2, mu1 = params_hemispheric_mean(single_scattering_albedo,
                                                    asymmetric_parameter)
    zeta_plus, zeta_minus, lambdan = zetalambda_coeffs(gamma_1, gamma_2)
    trans_coeff, scat_coeff = set_scat_trans_coeffs(zeta_plus, zeta_minus,
                                                    lambdan, dtau)

    piB = reduced_source_function_isothermal_layer(single_scattering_albedo,
                                                   gamma_1, gamma_2,
                                                   source_matrix)
    # Boundary condition
    diagonal_top = 1.0 * jnp.ones_like(trans_coeff[0, :])  # setting b0=1
    upper_diagonal_top = trans_coeff[0, :]

    zeta_plus0 = zeta_plus[0, :]
    zeta_minus0 = zeta_minus[0, :]

    # emission (no reflection)
    trans_func0 = jnp.exp(-lambdan[0, :] * dtau[0, :])
    denom = zeta_plus0**2 - (zeta_minus0 * trans_func0)**2
    omtrans = 1.0 - trans_func0
    fac = (zeta_plus0 * omtrans - zeta_minus0 * trans_func0 * omtrans)
    vector_top = (zeta_plus0**2 - zeta_minus0**2) / denom * fac * piB[0, :]

    # tridiagonal elements
    diagonal, lower_diagonal, upper_diagonal, vector = compute_tridiag_diagonals_and_vector(
        scat_coeff, trans_coeff, piB, upper_diagonal_top, diagonal_top,
        vector_top)

    return trans_coeff, scat_coeff, piB, diagonal, lower_diagonal, upper_diagonal, vector


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


# def rtrun(dtau, S):
#    warnings.warn("Use rtrun_emis_pureabs_flux2st instead", FutureWarning)
#    return rtrun_emis_pureabs_flux2st(dtau, S)

##########################################################################################
