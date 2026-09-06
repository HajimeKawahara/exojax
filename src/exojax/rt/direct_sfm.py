"""Layer source coefficients for direct-beam two-stream reflection."""

import math

import jax.numpy as jnp


def _attenuation_average(x):
    """Return ``(1 - exp(-x)) / x``, including its differentiable limit."""
    small = jnp.abs(x) < 1.0e-3
    safe_x = jnp.where(small, 1.0, x)
    regular = -jnp.expm1(-safe_x) / safe_x
    series_x = jnp.where(small, x, 0.0)
    series = 1.0 + series_x * (-0.5 + series_x * (1.0 / 6.0 - series_x / 24.0))
    return jnp.where(small, series, regular)


def _beam_moments(x):
    """Return ``x * integral_0^1 s**n exp(-x*s) ds`` for n = 0,...,3."""
    small = x < 1.0
    safe_x = jnp.where(small, 1.0, x)
    exponential = jnp.exp(-safe_x)
    moments = [-jnp.expm1(-x)]
    for order in range(1, 4):
        regular = order * moments[-1] / safe_x - exponential
        # The series avoids cancellation in the incomplete-gamma recurrence.
        series = jnp.zeros_like(x)
        series_x = jnp.where(small, x, 0.0)
        for power in range(17, -1, -1):
            coefficient = (-1.0) ** power / (
                math.factorial(power) * (order + power + 1)
            )
            series = coefficient + series_x * series
        moments.append(jnp.where(small, x * series, regular))
    return moments


def _direct_layer_sources(dtau, gamma1, gamma2, omega, mu0, gamma3):
    """Return upward/downward layer sources for a unit incident direct beam.

    The incident irradiance is measured normal to the beam at the layer top.
    The diffuse incoming intensities at both boundaries are zero. The source
    coefficients include all two-stream scattering within each uniform layer.
    Green-function integrals avoid the particular-solution singularity at
    ``lambda * mu0 = 1``. A Taylor branch includes the conservative limit.
    """
    dtau = jnp.asarray(dtau)
    squared = (gamma1 - gamma2) * (gamma1 + gamma2) * dtau**2
    use_taylor = squared < jnp.sqrt(jnp.finfo(squared.dtype).eps)
    safe_squared = jnp.where(use_taylor, 1.0, squared)
    u = jnp.sqrt(safe_squared)
    safe_dtau = jnp.where(dtau == 0.0, 1.0, dtau)
    lam = u / safe_dtau
    k = 1.0 / mu0
    x = dtau * k
    exponential = jnp.exp(-u)

    integral_a = dtau * _attenuation_average(u + x)
    difference = x - u
    positive = difference >= 0.0
    absolute_difference = jnp.where(positive, difference, -difference)
    integral_b = (
        dtau
        * jnp.exp(-jnp.where(positive, u, x))
        * _attenuation_average(absolute_difference)
    )
    c_plus = 0.5 * (integral_a + exponential * integral_b)
    c_minus = 0.5 * (integral_b + exponential * integral_a)
    s_plus = 0.5 * (integral_a - exponential * integral_b) / lam
    s_minus = 0.5 * (integral_b - exponential * integral_a) / lam
    denominator = 0.5 * (1.0 + exponential**2) + gamma1 * dtau * _attenuation_average(
        2.0 * u
    )

    # Cancel the common exp(-u) factor before expanding at lambda = 0.
    # Retaining squared explicitly also preserves conservative-limit gradients.
    taylor_squared = jnp.where(use_taylor, squared, 0.0)
    m0, m1, m2, m3 = _beam_moments(x)
    c_plus_taylor = mu0 * (m0 + 0.5 * taylor_squared * (m0 - 2.0 * m1 + m2))
    c_minus_taylor = mu0 * (m0 + 0.5 * taylor_squared * m2)
    s_plus_taylor = (
        dtau * mu0 * (m0 - m1 + taylor_squared * (m0 - 3.0 * m1 + 3.0 * m2 - m3) / 6.0)
    )
    s_minus_taylor = dtau * mu0 * (m1 + taylor_squared * m3 / 6.0)
    denominator_taylor = (
        1.0 + 0.5 * taylor_squared + gamma1 * dtau * (1.0 + taylor_squared / 6.0)
    )
    c_plus = jnp.where(use_taylor, c_plus_taylor, c_plus)
    c_minus = jnp.where(use_taylor, c_minus_taylor, c_minus)
    s_plus = jnp.where(use_taylor, s_plus_taylor, s_plus)
    s_minus = jnp.where(use_taylor, s_minus_taylor, s_minus)
    denominator = jnp.where(use_taylor, denominator_taylor, denominator)
    gamma4 = 1.0 - gamma3
    return (
        omega
        * (gamma3 * c_plus + (gamma1 * gamma3 + gamma2 * gamma4) * s_plus)
        / denominator,
        omega
        * (gamma4 * c_minus + (gamma1 * gamma4 + gamma2 * gamma3) * s_minus)
        / denominator,
    )
