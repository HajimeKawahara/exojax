"""Limb darkening functions."""

import jax.numpy as jnp


def ld_kipping(q1, q2):
    """Uninformative prior conversion of the limb darkening by Kipping
    (arxiv:1308.0009)

    Args:
        q1: U(0,1)
        q2: U(0,1)

    Returns:
        u1: quadratic LD coefficient u1
        u2: quadratic LD coefficient u2
    """
    sqrtq1 = jnp.sqrt(q1)
    return 2.0 * sqrtq1 * q2, sqrtq1 * (1.0 - 2.0 * q2)


def quadratic_ld_from_intensity(mus, intensity, weights=None):
    """Fit quadratic limb-darkening coefficients from emergent intensities.

    Args:
        mus: cosine angle array (N_mu)
        intensity: emergent intensity array (N_mu, N_nu) or (N_mu)
        weights: Gaussian quadrature weights for mu integration. If given, the
            least-squares fit uses disk-area weights proportional to mu * weight.

    Returns:
        u1: quadratic limb-darkening coefficient
        u2: quadratic limb-darkening coefficient
    """

    if len(mus) < 3:
        raise ValueError("At least three mu points are required for quadratic LD.")

    mus = jnp.asarray(mus)
    intensity = jnp.asarray(intensity)
    squeeze = intensity.ndim == 1
    if squeeze:
        intensity = intensity[:, None]

    q = 1.0 - mus
    design = jnp.stack([jnp.ones_like(q), -q, -(q**2)], axis=1)
    if weights is None:
        sqrt_weights = jnp.ones_like(mus)
    else:
        sqrt_weights = jnp.sqrt(jnp.asarray(weights) * mus)

    design_weighted = design * sqrt_weights[:, None]
    intensity_weighted = intensity * sqrt_weights[:, None]
    coeff = jnp.linalg.solve(
        design_weighted.T @ design_weighted,
        design_weighted.T @ intensity_weighted,
    )
    u1 = coeff[1] / coeff[0]
    u2 = coeff[2] / coeff[0]
    if squeeze:
        return u1[0], u2[0]
    return u1, u2


def average_limb_darkening_coefficients(u1, u2, weights=None):
    """Average wavelength-dependent limb-darkening coefficients."""

    if weights is None:
        return jnp.mean(u1), jnp.mean(u2)
    norm = jnp.sum(weights)
    return jnp.sum(u1 * weights) / norm, jnp.sum(u2 * weights) / norm
