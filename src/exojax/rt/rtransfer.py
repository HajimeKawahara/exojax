""" Runs radiative transfer

    The classification of rtrun(s):

    - flux-based emission
    -- pure absoprtion 
    --- 2stream: rtrun_emis_pureabs_flux2st, rtrun_emis_pureabs_flux2st_surface
    -- scattering
    --- 2stream
    ---- LART: rtrun_emis_scat_lart_toonhm
    ---- flux-adding: rtrun_emis_scat_fluxadding_toonhm
    -- relfection
    --- 2stream
    ---- flux-adding: rtrun_reflect_fluxadding_toonhm

    - intensity-based emission
    -- pure absorption 
    --- isothermal: rtrun_emis_pureabs_ibased
    --- linear source approximation: rtrun_emis_pureabs_ibased_linsap
    -- scattering
    --- SFM-2st: rtrun_emis_scat_sfm2st_toonhm,
        rtrun_emis_scat_sfm2st_toonhm_surface

    - intensity-based reflection
    -- scattering
    --- SFM-2st: rtrun_reflect_sfm2st_toonhm

    - transmision: 
    -- trapezoid integration: rtrun_trans_pureabs_trapezoid
    -- simpson integration: rtrun_trans_pureabs_simpson

"""

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from jax.scipy.integrate import trapezoid

from exojax.signal.integrate import simpson
from exojax.rt.toon import (
    params_hemispheric_mean,
    params_quadrature,
    zetalambda_coeffs,
)
from exojax.rt.direct_sfm import _direct_layer_sources
from exojax.rt.twostream import (
    compute_tridiag_diagonals_and_vector,
    set_scat_trans_absorption_coeffs,
    set_scat_trans_coeffs,  # re-exported for backward compatibility
    solve_fluxadding_twostream,
    solve_fluxadding_twostream_fluxes,
    solve_lart_twostream,
)
from exojax.special.expn import E1


_TRANS2E3_COMPLEMENT_SWITCH = 0.4190354


def _trans2E3_coefficients(x):
    """Returns transmission and absorption coefficients for pure absorption."""
    x = jnp.asarray(x)
    float_dtype = jnp.result_type(x, 1.0)
    # Keep x**2 and its reverse-mode cotangent below dtype overflow.
    too_large = x >= jnp.sqrt(jnp.finfo(float_dtype).max)
    x_eval = jnp.where(too_large, 1.0, x)
    x_e1 = jnp.where(x_eval == 0.0, 1.0, x_eval)
    exp_negative_x = jnp.exp(-x_eval)
    e1_term = x_eval**2 * E1(x_e1)
    transmission = (1.0 - x_eval) * exp_negative_x + e1_term
    absorption = (
        -jnp.expm1(-x_eval) + x_eval * exp_negative_x - e1_term
    )
    return (
        jnp.where(too_large, 0.0, transmission),
        jnp.where(too_large, 1.0, absorption),
    )


def _trans2E3_stable_coefficient(x):
    """Returns the smaller coefficient and its complementary-form selector."""
    trans_coeff, absorption_coeff = _trans2E3_coefficients(x)
    # 2 E3(x) = 0.5 at x = 0.419..., where both complementary forms
    # are equally well conditioned.
    use_absorption = jnp.asarray(x) <= _TRANS2E3_COMPLEMENT_SWITCH
    stable_coeff = jnp.where(
        use_absorption,
        absorption_coeff,
        trans_coeff,
    )
    return stable_coeff, use_absorption


def _solve_pure_absorption_emission(
    stable_coeff, use_absorption, source_matrix, source_surface
):
    """Integrates pure-absorption emission using stable complementary forms."""
    calculation_dtype = jnp.result_type(
        stable_coeff, source_matrix, source_surface
    )
    stable_coeff = jnp.asarray(stable_coeff, dtype=calculation_dtype)
    source_matrix = jnp.asarray(source_matrix, dtype=calculation_dtype)
    source_surface = jnp.asarray(source_surface, dtype=calculation_dtype)
    source_surface = jnp.broadcast_to(source_surface, source_matrix.shape[1:])

    def integrate_layer(carry, layer):
        flux, correction = carry
        coeff_layer, use_absorption_layer, source_layer = layer
        source_difference = (source_layer - flux) - correction
        absorption_increment = correction + coeff_layer * source_difference
        transmission_increment = -coeff_layer * source_difference
        base = jnp.where(use_absorption_layer, flux, source_layer)
        # Use the smaller complementary coefficient to avoid cancellation in
        # both optically thin and optically thick layers.
        increment = jnp.where(
            use_absorption_layer,
            absorption_increment,
            transmission_increment,
        )

        updated_flux = base + increment
        # Carry the rounded-off part of the update into the next layer.
        correction = increment - (updated_flux - base)
        return (updated_flux, correction), None

    (flux, correction), _ = scan(
        integrate_layer,
        (source_surface, jnp.zeros_like(source_surface)),
        (stable_coeff, use_absorption, source_matrix),
        reverse=True,
    )
    return flux + correction


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
    transmission, _ = _trans2E3_coefficients(x)
    return transmission


@jit
def rtrun_emis_pureabs_fbased2st(dtau, source_matrix):
    """Radiative Transfer for emission spectrum using flux-based two-stream pure absorption with no surface
    Args:
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix (2D array): source matrix (N_layer, N_nus)

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    stable_coeff, use_absorption = _trans2E3_stable_coefficient(dtau)
    return _solve_pure_absorption_emission(
        stable_coeff,
        use_absorption,
        source_matrix,
        jnp.zeros_like(source_matrix[0]),
    )


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
    stable_coeff, use_absorption = _trans2E3_stable_coefficient(dtau)
    return _solve_pure_absorption_emission(
        stable_coeff, use_absorption, source_matrix, source_surface
    )


@jit
def rtrun_emis_pureabs_ibased_intensity(dtau, source_matrix, mus):
    """Emergent intensities for intensity-based pure absorption RT.

    Args:
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix (2D array): source matrix (N_layer, N_nus)
        mus (list): mu (cos theta) list for integration

    Returns:
        2D array: emergent intensity matrix (N_mu, N_nus)
    """

    Nnus = jnp.shape(dtau)[1]
    tau = jnp.cumsum(dtau, axis=0)
    tau_upper = jnp.concatenate((jnp.zeros_like(dtau[:1]), tau[:-1]), axis=0)

    def f(carry, mu):
        # Preserve absorption in optically thin layers.
        dtrans = jnp.exp(-tau_upper / mu) * -jnp.expm1(-dtau / mu)
        intensity = jnp.sum(source_matrix * dtrans, axis=0)
        return carry, intensity

    _, intensity = scan(f, jnp.zeros(Nnus), mus)
    return intensity


@jit
def rtrun_emis_pureabs_ibased_intensity_surface(
    dtau, source_matrix, source_surface, mus
):
    """Emergent pure-absorption intensities with a lower surface.

    Args:
        dtau: Layer optical depths with shape ``(N_layer, N_nus)``.
        source_matrix: Layer source functions with shape
            ``(N_layer, N_nus)``.
        source_surface: Isotropic lower-boundary source with shape ``(N_nus,)``.
        mus: Positive ray-angle cosines with shape ``(N_mu,)``.

    Returns:
        Emergent intensity matrix with shape ``(N_mu, N_nus)``.
    """
    intensity = rtrun_emis_pureabs_ibased_intensity(dtau, source_matrix, mus)
    tau_bottom = jnp.sum(dtau, axis=0)
    transmission_surface = jnp.exp(-tau_bottom[None, :] / mus[:, None])
    return intensity + source_surface[None, :] * transmission_surface


@jit
def rtrun_emis_pureabs_ibased_flux_from_intensity(intensity, mus, weights):
    """Integrate emergent intensities over angle using the ibased quadrature."""

    return jnp.einsum("i,i,ij->j", 2.0 * mus, weights, intensity)


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

    intensity = rtrun_emis_pureabs_ibased_intensity(dtau, source_matrix, mus)
    return rtrun_emis_pureabs_ibased_flux_from_intensity(intensity, mus, weights)


def initialize_gaussian_quadrature(nstream):
    """Initialization of Gaussian Quadrature

    Args:
        nstream (int): the number of the stream

    Raises:
        ValueError: odd nstream error

    Returns:
        array, array: cosine angle array (mu), weight array
    """
    from scipy.special import roots_legendre

    if nstream % 2 == 0:
        norder = int(nstream / 2)
    else:
        raise ValueError("nstream should be even number larger than 2.")
    mus, weights = roots_legendre(norder)

    # correction because integration should be between 0 to 1, but roots_legendre uses -1 to 1.
    mus = 0.5 * (mus + 1.0)
    weights = 0.5 * weights
    return mus, weights


@jit
def rtrun_emis_pureabs_ibased_linsap(dtau, source_matrix_boundary, mus, weights):
    """Radiative Transfer for emission spectrum using intensity-based n-stream pure absorption with no surface w/ linear source approximation = linsap (HELIOS-R2 like)

    Args:
        dtau (2D array): optical depth matrix, dtau  (N_layer, N_nus)
        source_matrix_booundary (2D array): source matrix at the layer upper boundary (N_layer + 1, N_nus)
        mus (list): mu (cos theta) list for integration
        weights (list): weight list for mu

    Returns:
        flux in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.

    Notes:
        See Olson and Kunasz as well as HELIOS-R2 paper (Kitzmann+) for the derivation.


    """

    Nnus = jnp.shape(dtau)[1]
    source_matrix_boundary_p1 = jnp.roll(source_matrix_boundary, -1, axis=0)  # S_{n+1}

    # NOT IMPLEMENTED YET
    # need to replace the last element of the above
    #

    # scan part
    muws = [mus, weights]

    def f(carry_fmu, muw):
        mu, w = muw
        dtau_per_mu = dtau / mu
        trans = jnp.exp(-dtau_per_mu)  # hat{T}
        beta, gamma = coeffs_linsap(dtau_per_mu, trans)

        # adds coeffs at the bottom of the layers
        beta = jnp.vstack([beta, jnp.ones(Nnus)])
        gamma = jnp.vstack([gamma, jnp.zeros(Nnus)])

        dI = beta * source_matrix_boundary + gamma * source_matrix_boundary_p1
        intensity_for_mu = jnp.sum(
            dI * jnp.cumprod(jnp.vstack([jnp.ones(Nnus), trans]), axis=0), axis=0
        )

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
    # Use the series near zero; safe_dtau also keeps the inactive branch AD-safe.
    small = jnp.abs(dtau_per_mu) < 1.0e-3
    safe_dtau = jnp.where(small, 1.0, dtau_per_mu)
    fac = -jnp.expm1(-dtau_per_mu) / safe_dtau

    beta_small = dtau_per_mu * (
        0.5 + dtau_per_mu * (-1.0 / 6.0 + dtau_per_mu / 24.0)
    )
    gamma_small = dtau_per_mu * (
        0.5 + dtau_per_mu * (-1.0 / 3.0 + dtau_per_mu / 8.0)
    )
    beta = jnp.where(small, beta_small, 1.0 - fac)
    gamma = jnp.where(small, gamma_small, -trans + fac)
    return beta, gamma


@jit
def rtrun_trans_pureabs_trapezoid(dtau_chord, radius_lower, radius_top):
    """Radiative transfer for transmission spectrum assuming pure absorption with the trapezoid integration (jax.scipy.integrate.trapezoid)

    Args:
        dtau_chord (2D array): chord optical depth (Nlayer, N_wavenumber)
        radius_lower (1D array): (normalized) radius at the lower boundary, underline(r) (Nlayer). R0 = radius_lower[-1] corresponds to the most bottom of the layers.
        radius_top (float): (normalized) radius at the ToA, i.e. the radius at the most top of the layers

    Returns:
        1D array: transit squared radius normalized by radius_lower[-1], i.e. it returns (radius/radius_lower[-1])**2

    Notes:
        This function gives the sqaure of the transit radius.
        If you would like to obtain the transit radius, take sqaure root of the output.
        If you would like to compute the transit depth, devide the output by the square of stellar radius

    Notes:
        We need the edge correction because the trapezoid integration with radius_lower lacks the edge point integration.
        i.e. the integration of the 0-th layer from radius_lower[0] to radius_top.
        We assume tau = 0 at the radius_top. then, the edge correction should be (1-T_0)*(delta r_0), but usually negligible though.

    """
    dr = radius_top - radius_lower[0]
    edge_cor = (1.0 - jnp.exp(-dtau_chord[0, :])) * radius_top * dr

    # the negative sign is because the radius_lower is in a descending order
    deltaRp2 = (
        -2.0
        * trapezoid(
            (1.0 - jnp.exp(-dtau_chord)) * radius_lower[:, None], x=radius_lower, axis=0
        )
        + edge_cor
    )
    return deltaRp2 + radius_lower[-1] ** 2


@jit
def rtrun_trans_pureabs_simpson(
    dtau_chord_modpoint, dtau_chord_lower, radius_lower, height
):
    """Radiative transfer for transmission spectrum assuming pure absorption with the Simpson integration (signals.integration.simpson)

    Args:
        dtau_chord_midpoint (2D array): chord opatical depth at the midpoint (Nlayer, N_wavenumber)
        dtau_chord_lower (2D array): chord opatical depth at the lower boundary (Nlayer, N_wavenumber)
        radius_lower (1D array): (normalized) radius at the lower boundary, underline(r) (Nlayer). R0 = radius_lower[-1] corresponds to the most bottom of the layers.
        height (1D array): (normalized) height of the layers

    Returns:
        1D array: transit squared radius normalized by radius_lower[-1], i.e. it returns (radius/radius_lower[-1])**2

    Notes:
        This function gives the sqaure of the transit radius.
        If you would like to obtain the transit radius, take sqaure root of the output.
        If you would like to compute the transit depth, devide the output by the square of stellar radius

    Notes:
        We need the edge correction because the trapezoid integration with radius_lower lacks the edge point integration.
        i.e. the integration of the 0-th layer from radius_lower[0] to radius_top.
        We assume tau = 0 at the radius_top. then, the edge correction should be (1-T_0)*(delta r_0), but usually negligible though.

    """
    radius_midpoint = radius_lower + 0.5 * height
    _, Nnus = jnp.shape(dtau_chord_modpoint)
    f = 2.0 * (1.0 - jnp.exp(-dtau_chord_modpoint)) * radius_midpoint[:, None]
    f_lower = 2.0 * (1.0 - jnp.exp(-dtau_chord_lower)) * radius_lower[:, None]
    f_top = jnp.zeros(Nnus)
    deltaRp2 = simpson(f, f_lower, f_top, height)
    return deltaRp2 + radius_lower[-1] ** 2


@jit
def rtrun_emis_scat_lart_toonhm(
    dtau, single_scattering_albedo, asymmetric_parameter, source_matrix
):
    """Radiative Transfer for emission spectrum using flux-based two-stream scattering LART solver w/ Toon Hemispheric Mean with no surface.

    Args:
        dtau (2D array): Optical depth matrix, dtau (N_layer, N_nus)
        single_scattering_albedo (2D array): Single scattering albedo (N_layer, N_nus)
        asymmetric_parameter (2D array): Asymmetric parameter (N_layer, N_nus)
        source_matrix (2D array): Source matrix (N_layer, N_nus)

    Returns:
        tuple: A tuple containing:
            - spectrum (1D array): Emission spectrum in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
            - cumTtilde (2D array): Cumulative transmission function.
            - Qtilde (2D array): Scattering source function.
            - trans_coeff (2D array): Transmission coefficients.
            - scat_coeff (2D array): Scattering coefficients.
            - reduced_piB (2D array): Reduced source function.
    """
    toon_coeffs = setrt_toonhm_with_absorption(
        dtau, single_scattering_albedo, asymmetric_parameter, source_matrix
    )
    trans_coeff, scat_coeff, absorption_coeff, reduced_piB = toon_coeffs[:4]
    zeta_plus, zeta_minus, lambdan = toon_coeffs[4:]

    # avoids zero
    epsilon = 1.0e-8
    scat_coeff = scat_coeff + epsilon
    trans_coeff = trans_coeff + epsilon

    diagonal, lower_diagonal, upper_diagonal, vector = settridiag_toohm(
        dtau,
        zeta_plus,
        zeta_minus,
        lambdan,
        trans_coeff,
        scat_coeff,
        reduced_piB,
        absorption_coeff,
    )
    nlayer, Nnus = diagonal.shape
    cumTtilde, Qtilde, spectrum = solve_lart_twostream(
        diagonal, lower_diagonal, upper_diagonal, vector, jnp.zeros(Nnus)
    )

    return spectrum, cumTtilde, Qtilde, trans_coeff, scat_coeff, reduced_piB


@jit
def rtrun_emis_scat_lart_toonhm_surface(
    dtau, single_scattering_albedo, asymmetric_parameter, source_matrix, source_surface
):
    """Radiative Transfer for emission spectrum using flux-based two-stream scattering LART solver w/ Toon Hemispheric Mean with surface.

    Args:
        dtau (2D array): Optical depth matrix, dtau (N_layer, N_nus)
        single_scattering_albedo (2D array): Single scattering albedo (N_layer, N_nus)
        asymmetric_parameter (2D array): Asymmetric parameter (N_layer, N_nus)
        source_matrix (2D array): Source matrix (N_layer, N_nus)
        source_surface (1D array): Source from the surface (N_nus)

    Returns:
        tuple: A tuple containing:
            - spectrum (1D array): Emission spectrum in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
            - cumTtilde (2D array): Cumulative transmission function.
            - Qtilde (2D array): Scattering source function.
            - trans_coeff (2D array): Transmission coefficients.
            - scat_coeff (2D array): Scattering coefficients.
            - piB (2D array): Reduced source function.
    """
    toon_coeffs = setrt_toonhm_with_absorption(
        dtau, single_scattering_albedo, asymmetric_parameter, source_matrix
    )
    trans_coeff, scat_coeff, absorption_coeff, piB = toon_coeffs[:4]
    zeta_plus, zeta_minus, lambdan = toon_coeffs[4:]
    diagonal, lower_diagonal, upper_diagonal, vector = settridiag_toohm(
        dtau,
        zeta_plus,
        zeta_minus,
        lambdan,
        trans_coeff,
        scat_coeff,
        piB,
        absorption_coeff,
    )

    cumTtilde, Qtilde, spectrum = solve_lart_twostream(
        diagonal, lower_diagonal, upper_diagonal, vector, source_surface
    )

    return spectrum, cumTtilde, Qtilde, trans_coeff, scat_coeff, piB

@jit
def rtrun_reflect_fluxadding_toonhm(
    dtau,
    single_scattering_albedo,
    asymmetric_parameter,
    source_matrix,
    source_surface,
    reflectivity_surface,
    incoming_flux,
):
    """Radiative Transfer for reflected spectrum using the flux adding solver w/ Toon Hemispheric Mean with surface.

    Args:
        dtau (2D array): Layer optical depth (N_layer, N_nus)
        single_scattering_albedo (2D array): Single scattering albedo (N_layer, N_nus)
        asymmetric_parameter (2D array): Asymmetric parameter (N_layer, N_nus)
        source_matrix (2D array): Source term (N_layer, N_nus)
        source_surface (1D array): Source from the surface (N_nus)
        reflectivity_surface (1D array): Reflectivity from the surface (N_nus)
        incoming_flux (1D array): Incoming flux F_0^- (N_nus)

    Returns:
        1D array: Reflected spectrum in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    toon_coeffs = setrt_toonhm_with_absorption(
        dtau, single_scattering_albedo, asymmetric_parameter, source_matrix
    )
    trans_coeff, scat_coeff, absorption_coeff, reduced_piB = toon_coeffs[:4]
    
    Rphat, Sphat = solve_fluxadding_twostream(
        trans_coeff,
        scat_coeff,
        reduced_piB,
        reflectivity_surface,
        source_surface,
        absorption_coeff,
    )
    
    return Rphat * incoming_flux + Sphat


@jit
def rtrun_emis_scat_fluxadding_toonhm(
    dtau, single_scattering_albedo, asymmetric_parameter, source_matrix
):
    """Radiative Transfer for emission spectrum (w/ scattering) using flux-based two-stream scattering the flux adding solver w/ Toon Hemispheric Mean with surface.

    Args:
        dtau (2D array): Optical depth matrix, dtau (N_layer, N_nus)
        single_scattering_albedo (2D array): Single scattering albedo (N_layer, N_nus)
        asymmetric_parameter (2D array): Asymmetric parameter (N_layer, N_nus)
        source_matrix (2D array): Source matrix (N_layer, N_nus)

    Returns:
        1D array: Emission spectrum in the unit of [erg/cm2/s/cm-1] if using piBarr as a source function.
    """
    _, Nnus = dtau.shape
    source_surface = jnp.zeros(Nnus)
    reflectivity_surface = jnp.zeros(Nnus)

    toon_coeffs = setrt_toonhm_with_absorption(
        dtau, single_scattering_albedo, asymmetric_parameter, source_matrix
    )
    trans_coeff, scat_coeff, absorption_coeff, reduced_piB = toon_coeffs[:4]

    _, spectrum = solve_fluxadding_twostream(
        trans_coeff,
        scat_coeff,
        reduced_piB,
        reflectivity_surface,
        source_surface,
        absorption_coeff,
    )

    return spectrum


def _solve_sfm2st_layer_source(
    dtau,
    single_scattering_albedo,
    asymmetric_parameter,
    source_matrix,
    reflectivity_bottom,
    source_bottom,
    source_top=None,
):
    """Build the SFM-2st layer source from the two-stream flux solution."""
    toon_coeffs = setrt_toonhm_with_absorption(
        dtau, single_scattering_albedo, asymmetric_parameter, source_matrix
    )
    trans_coeff, scat_coeff, absorption_coeff, reduced_piB = toon_coeffs[:4]

    flux_plus, flux_minus = solve_fluxadding_twostream_fluxes(
        trans_coeff,
        scat_coeff,
        reduced_piB,
        reflectivity_bottom,
        source_bottom,
        source_top=source_top,
        absorption_coeff=absorption_coeff,
    )

    flux_plus_layer = 0.5 * (flux_plus[:-1] + flux_plus[1:])
    flux_minus_layer = 0.5 * (flux_minus[:-1] + flux_minus[1:])
    source_sfm = (1.0 - single_scattering_albedo) * source_matrix + (
        0.5
        * single_scattering_albedo
        * (
            (1.0 + asymmetric_parameter) * flux_plus_layer
            + (1.0 - asymmetric_parameter) * flux_minus_layer
        )
    )
    return source_sfm, flux_plus[-1]


def _rtrun_sfm2st_toonhm(
    dtau,
    single_scattering_albedo,
    asymmetric_parameter,
    source_matrix,
    source_surface,
    reflectivity_surface,
    incoming_flux,
    mus,
    weights,
):
    """Run the shared SFM-2st formal solution with boundary sources."""
    source_sfm, source_bottom = _solve_sfm2st_layer_source(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        source_matrix,
        reflectivity_surface,
        source_surface,
        incoming_flux,
    )

    intensity = rtrun_emis_pureabs_ibased_intensity_surface(
        dtau, source_sfm, source_bottom, mus
    )
    return rtrun_emis_pureabs_ibased_flux_from_intensity(intensity, mus, weights)


@jit
def rtrun_emis_scat_sfm2st_toonhm(
    dtau,
    single_scattering_albedo,
    asymmetric_parameter,
    source_matrix,
    mus,
    weights,
):
    """Radiative transfer for emission with scattering using SFM-2st.

    The two-stream fluxes are computed with Toon hemispheric mean and
    converted into layer source functions. The final intensity transfer
    reuses the isothermal-layer intensity-based pure absorption solver.

    Args:
        dtau (2D array): Optical depth matrix, dtau (N_layer, N_nus).
        single_scattering_albedo (2D array): Single scattering albedo.
        asymmetric_parameter (2D array): Asymmetric parameter.
        source_matrix (2D array): Source matrix in pi B scale.
        mus (1D array): Gaussian quadrature cosine angles.
        weights (1D array): Gaussian quadrature weights.

    Returns:
        1D array: Emission spectrum.
    """

    _, Nnus = dtau.shape
    source_surface = jnp.zeros(Nnus)
    reflectivity_surface = jnp.zeros(Nnus)

    source_sfm, _ = _solve_sfm2st_layer_source(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        source_matrix,
        reflectivity_surface,
        source_surface,
    )

    return rtrun_emis_pureabs_ibased(dtau, source_sfm, mus, weights)


@jit
def rtrun_emis_scat_sfm2st_toonhm_surface(
    dtau,
    single_scattering_albedo,
    asymmetric_parameter,
    source_matrix,
    source_surface,
    mus,
    weights,
):
    """Radiative transfer for SFM-2st emission with a lower thermal source.

    Args:
        dtau: Layer optical depths with shape ``(N_layer, N_nus)``.
        single_scattering_albedo: Single-scattering albedo.
        asymmetric_parameter: Scattering asymmetry parameter.
        source_matrix: Thermal layer source in pi B scale.
        source_surface: Isotropic lower-boundary source in pi B scale.
        mus: Positive ray-angle cosines.
        weights: Gaussian quadrature weights.

    Returns:
        Top-of-atmosphere emission flux.
    """
    _, Nnus = dtau.shape
    return _rtrun_sfm2st_toonhm(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        source_matrix,
        source_surface,
        jnp.zeros(Nnus),
        jnp.zeros(Nnus),
        mus,
        weights,
    )


@jit
def rtrun_reflect_sfm2st_toonhm(
    dtau,
    single_scattering_albedo,
    asymmetric_parameter,
    source_matrix,
    source_surface,
    reflectivity_surface,
    incoming_flux,
    mus,
    weights,
):
    """Radiative transfer for diffuse reflection using SFM-2st.

    Toon hemispheric-mean two-stream fluxes are converted into layer source
    functions. The final outgoing flux is obtained from an intensity-based
    formal solution. The incident radiation is a diffuse hemispheric flux at
    the top boundary.

    Args:
        dtau: Layer optical depths with shape ``(N_layer, N_nus)``.
        single_scattering_albedo: Single-scattering albedo.
        asymmetric_parameter: Scattering asymmetry parameter.
        source_matrix: Thermal layer source in pi B scale.
        source_surface: Emitting lower-boundary source.
        reflectivity_surface: Lambertian lower-boundary reflectivity.
        incoming_flux: Diffuse downward flux at the top boundary.
        mus: Positive ray-angle cosines.
        weights: Gaussian quadrature weights.

    Returns:
        Reflected and emitted top-of-atmosphere flux.
    """
    return _rtrun_sfm2st_toonhm(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        source_matrix,
        source_surface,
        reflectivity_surface,
        incoming_flux,
        mus,
        weights,
    )


@partial(jit, static_argnames=("phase_function",))
def rtrun_reflect_sfm2st_direct(
    dtau,
    single_scattering_albedo,
    reflectivity_surface,
    incoming_flux,
    mu_in,
    mu_out,
    relative_azimuth=0.0,
    phase_function="rayleigh",
):
    """Return specific intensity for direct illumination using two-stream SFM.

    ``incoming_flux`` is beam-normal irradiance; the horizontal incident flux
    is ``mu_in * incoming_flux``. Both positive direction cosines are scalar.
    ``relative_azimuth`` is the azimuth difference between the outward star
    and observer directions, in radians. The result is physical intensity
    (irradiance per steradian), rather than the pi-I scale used internally.

    Isotropic and Rayleigh single scattering are integrated analytically in
    each homogeneous layer. Multiple scattering uses Toon quadrature with
    g=0 and a Toon-style hemispheric, layer-averaged source reconstruction. This
    approximation does not retain Rayleigh's higher angular moments or
    polarization. Resolve the diffuse source by refining the optical-depth
    layers. Angular integration of this approximate intensity need not conserve
    the two-stream flux exactly, even after layer convergence. The lower
    boundary is Lambertian; the top has no diffuse source.
    No disk integration is performed.
    """
    if phase_function not in ("isotropic", "rayleigh"):
        raise ValueError("phase_function must be 'isotropic' or 'rayleigh'")

    gamma_1, gamma_2, gamma_3, _ = params_quadrature(
        single_scattering_albedo, jnp.zeros_like(dtau), mu_in
    )
    trans, scat, absorption = set_scat_trans_absorption_coeffs(
        gamma_1, gamma_2, dtau
    )
    source_plus, source_minus = _direct_layer_sources(
        dtau, gamma_1, gamma_2, single_scattering_albedo, mu_in, gamma_3
    )
    tau = jnp.concatenate((jnp.zeros_like(dtau[:1]), jnp.cumsum(dtau, axis=0)))
    beam = incoming_flux * jnp.exp(-tau / mu_in)
    surface = jnp.broadcast_to(reflectivity_surface, dtau.shape[1:])
    flux_plus, flux_minus = solve_fluxadding_twostream_fluxes(
        trans,
        scat,
        jnp.zeros_like(dtau),
        surface,
        surface * mu_in * beam[-1],
        absorption_coeff=absorption,
        source_plus=source_plus * beam[:-1],
        source_minus=source_minus * beam[:-1],
    )

    # F+ and F- contain scattered light only, so scattering this field adds
    # the multiple-scattering contribution without counting the beam twice.
    diffuse_source = 0.25 * single_scattering_albedo * (
        flux_plus[:-1] + flux_plus[1:] + flux_minus[:-1] + flux_minus[1:]
    )
    diffuse_pi_intensity = rtrun_emis_pureabs_ibased_intensity_surface(
        dtau, diffuse_source, flux_plus[-1], jnp.atleast_1d(mu_out)
    )[0]

    phase = 1.0
    if phase_function == "rayleigh":
        sine_product_squared = (1.0 - mu_in**2) * (1.0 - mu_out**2)
        # Keep derivatives finite when either direction is exactly normal.
        at_normal = sine_product_squared == 0.0
        sine_product = jnp.where(
            at_normal, 0.0, jnp.sqrt(jnp.where(at_normal, 1.0, sine_product_squared))
        )
        cos_scattering = -(
            mu_in * mu_out + sine_product * jnp.cos(relative_azimuth)
        )
        phase = 0.75 * (1.0 + cos_scattering**2)

    attenuation = jnp.exp(-tau[:-1] / mu_out) * -jnp.expm1(
        -dtau * (1.0 / mu_in + 1.0 / mu_out)
    )
    single_pi_intensity = jnp.sum(
        0.25 * single_scattering_albedo * phase * beam[:-1] * attenuation
        * mu_in / (mu_in + mu_out),
        axis=0,
    )
    return (diffuse_pi_intensity + single_pi_intensity) / jnp.pi


def setrt_toonhm(
    dtau,
    single_scattering_albedo,
    asymmetric_parameter,
    source_matrix,
):
    """Sets some coefficients for radiative transfer assuming Toon Hemispheric Mean.

    Args:
        dtau (2D array): Optical depth matrix, dtau (N_layer, N_nus)
        single_scattering_albedo (2D array): Single scattering albedo (N_layer, N_nus)
        asymmetric_parameter (2D array): Asymmetric parameter (N_layer, N_nus)
        source_matrix (2D array): Source matrix (N_layer, N_nus)

    Returns:
        tuple: Transmission, scattering, source, zeta, and lambda coefficients.
    """
    gamma_1, gamma_2, _ = params_hemispheric_mean(
        single_scattering_albedo, asymmetric_parameter
    )
    zeta_plus, zeta_minus, lambdan = zetalambda_coeffs(gamma_1, gamma_2)
    trans_coeff, scat_coeff = set_scat_trans_coeffs(gamma_1, gamma_2, dtau)

    return (
        trans_coeff,
        scat_coeff,
        source_matrix,
        zeta_plus,
        zeta_minus,
        lambdan,
    )


def setrt_toonhm_with_absorption(
    dtau, single_scattering_albedo, asymmetric_parameter, source_matrix
):
    """Sets Toon hemispheric-mean coefficients including absorption.

    The absorption coefficient is evaluated directly and returned after the
    scattering coefficient so that thin-layer thermal sources remain accurate.
    """
    gamma_1, gamma_2, mu1 = params_hemispheric_mean(
        single_scattering_albedo, asymmetric_parameter
    )
    zeta_plus, zeta_minus, lambdan = zetalambda_coeffs(gamma_1, gamma_2)
    trans_coeff, scat_coeff, absorption_coeff = set_scat_trans_absorption_coeffs(
        gamma_1, gamma_2, dtau
    )

    reduced_piB = source_matrix

    return (
        trans_coeff,
        scat_coeff,
        absorption_coeff,
        reduced_piB,
        zeta_plus,
        zeta_minus,
        lambdan,
    )


def settridiag_toohm(
    dtau,
    zeta_plus,
    zeta_minus,
    lambdan,
    trans_coeff,
    scat_coeff,
    reduced_piB,
    absorption_coeff=None,
):
    diagonal_top = 1.0 * jnp.ones_like(trans_coeff[0, :])  # setting b0=1
    upper_diagonal_top = trans_coeff[0, :]

    # emission (no reflection)
    if absorption_coeff is None:
        raw_absorption_coeff = (1.0 - trans_coeff) - scat_coeff
        absorption_coeff = jnp.where(
            raw_absorption_coeff < 0.0, 0.0, raw_absorption_coeff
        )
    vector_top = absorption_coeff[0, :] * reduced_piB[0, :]

    # tridiagonal elements
    (
        diagonal,
        lower_diagonal,
        upper_diagonal,
        vector,
    ) = compute_tridiag_diagonals_and_vector(
        scat_coeff,
        trans_coeff,
        reduced_piB,
        upper_diagonal_top,
        diagonal_top,
        vector_top,
        absorption_coeff,
    )

    return diagonal, lower_diagonal, upper_diagonal, vector
