"""Methods for Toon et al. 1989

- gamma_1, gamma_2, (gamma_3,) and mu1 are the fundamental parameters of Toon+89 (Table 1), called toon_params. 
- gamma_1, gamma_2 can be converted to the fundamental parameters of the twostream solver, zeta and lambda.
- one can choose Eddington, Quadrature, Hemispheric_Mean to compute toon_params from single_scattering_albedo and asymmetric_parameter (and mu0 for the former two cases)

"""

import jax.numpy as jnp

def toon_zetalambda_coeffs(gamma_1, gamma_2):
    """computes coupling coefficients zeta and lambda coefficients for Toon-type two stream approximation 

    Args:
        gamma_1 (_type_): Toon+89 gamma_1 coefficient
        gamma_2 (_type_): Toon+89 gamma_2 coefficient

    Returns:
        _type_: coupling zeta (+), coupling zeta (-), lambda coefficients
    """
    delta = jnp.sqrt((gamma_1 - gamma_2) / (gamma_1 + gamma_2))
    zeta_plus = 0.5 * (1.0 + delta)
    zeta_minus = 0.5 * (1.0 - delta)
    lambdan = jnp.sqrt(gamma_1**2 - gamma_2**2)
    return zeta_plus, zeta_minus, lambdan

def toon_params_eddington(single_scattering_albedo, asymmetric_parameter, mu0):
    gamma_1 = (7.0 - single_scattering_albedo*(4.0+3.0*asymmetric_parameter))/4.0
    gamma_2 = -(1.0 - single_scattering_albedo*(4.0-3.0*asymmetric_parameter))/4.0
    gamma_3 = (2.0 - 3.0*asymmetric_parameter*mu0)/4.0
    mu1 = 0.5
    return gamma_1, gamma_2, gamma_3, mu1

def toon_params_quadrature(single_scattering_albedo, asymmetric_parameter, mu0):
    s3 = jnp.sqrt(3.0)
    gamma_1 = s3*(2.0 - single_scattering_albedo*(1.0+asymmetric_parameter))/2.0
    gamma_2 = single_scattering_albedo*s3*(1.0 - asymmetric_parameter)/2.0
    gamma_3 = (1.0 - s3*asymmetric_parameter*mu0)/2.0
    mu1 = 1.0/s3
    return gamma_1, gamma_2, gamma_3, mu1

def toon_params_hemispheric_mean(single_scattering_albedo, asymmetric_parameter):
    gamma_1 = 2.0 - single_scattering_albedo*(1.0+asymmetric_parameter)
    gamma_2 = single_scattering_albedo*(1.0-asymmetric_parameter)
    mu1 = 0.5
    return gamma_1, gamma_2, mu1

