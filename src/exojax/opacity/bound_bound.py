"""Absorption and emission coefficients for bound-bound transitions."""

import jax.numpy as jnp

from exojax.utils.checkarray import require_ndim
from exojax.utils.constants import ccgs
from exojax.utils.constants import hcgs


def population_inversion_mask(
    number_density_lower,
    number_density_upper,
    g_lower,
    g_upper,
):
    """Return where a bound-bound transition has negative true absorption.

    Args:
        number_density_lower: Shape ``(N_layer, N_line)`` in cm-3.
        number_density_upper: Shape ``(N_layer, N_line)`` in cm-3.
        g_lower: Lower-state statistical weights, shape ``(N_line,)``.
        g_upper: Upper-state statistical weights, shape ``(N_line,)``.

    Returns:
        Boolean array with shape ``(N_layer, N_line)``.
    """
    number_density_lower = jnp.asarray(number_density_lower)
    number_density_upper = jnp.asarray(number_density_upper)
    g_lower = jnp.asarray(g_lower)
    g_upper = jnp.asarray(g_upper)

    require_ndim("number_density_lower", number_density_lower, 2)
    require_ndim("number_density_upper", number_density_upper, 2)
    require_ndim("g_lower", g_lower, 1)
    require_ndim("g_upper", g_upper, 1)
    if number_density_upper.shape != number_density_lower.shape:
        raise ValueError(
            "number_density_lower and number_density_upper must have the same shape."
        )
    nline = number_density_lower.shape[1]
    if g_lower.shape != (nline,) or g_upper.shape != (nline,):
        raise ValueError(
            f"g_lower and g_upper must have shape ({nline},), got "
            f"{g_lower.shape} and {g_upper.shape}."
        )

    return number_density_upper * g_lower[None, :] > (
        number_density_lower * g_upper[None, :]
    )


def bound_bound_absorption_emission(
    line_profile,
    nu_lines,
    einstein_a,
    g_lower,
    g_upper,
    number_density_lower,
    number_density_upper,
):
    """Compute bound-bound absorption and pi-scaled emissivity.

    The same profile is used for absorption, stimulated emission, and
    spontaneous emission. It must be normalized over wavenumber. Level
    populations are accepted directly; no LTE population model is applied.

    Args:
        line_profile: Normalized profile with shape
            ``(N_layer, N_line, N_wavenumber)``.
        nu_lines: Rest line centers in cm-1, shape ``(N_line,)``.
        einstein_a: Einstein A in s-1, shape ``(N_line,)``.
        g_lower: Lower-state statistical weights, shape ``(N_line,)``.
        g_upper: Upper-state statistical weights, shape ``(N_line,)``.
        number_density_lower: Lower-state number density in cm-3, shape
            ``(N_layer, N_line)``.
        number_density_upper: Upper-state number density in cm-3, shape
            ``(N_layer, N_line)``.

    Returns:
        Tuple ``(alpha_line, eta_pi_line)``. Both have shape
        ``(N_layer, N_wavenumber)``. ``alpha_line`` is in cm-1.
        ``eta_pi_line / alpha_line`` follows the pi-scaled ExoJAX source
        convention.

    Notes:
        Population inversion produces negative absorption. Callers targeting
        ordinary transfer should constrain or reject samples identified by
        :func:`population_inversion_mask`.
    """
    line_profile = jnp.asarray(line_profile)
    nu_lines = jnp.asarray(nu_lines)
    einstein_a = jnp.asarray(einstein_a)
    g_lower = jnp.asarray(g_lower)
    g_upper = jnp.asarray(g_upper)
    number_density_lower = jnp.asarray(number_density_lower)
    number_density_upper = jnp.asarray(number_density_upper)

    require_ndim("line_profile", line_profile, 3)
    nlayer, nline, _ = line_profile.shape
    for name, array in (
        ("nu_lines", nu_lines),
        ("einstein_a", einstein_a),
        ("g_lower", g_lower),
        ("g_upper", g_upper),
    ):
        require_ndim(name, array, 1)
        if array.shape[0] != nline:
            raise ValueError(
                f"{name} must have length {nline}, got shape {array.shape}."
            )
    expected_population_shape = (nlayer, nline)
    if number_density_lower.shape != expected_population_shape:
        raise ValueError(
            "number_density_lower must have shape "
            f"{expected_population_shape}, got {number_density_lower.shape}."
        )
    if number_density_upper.shape != expected_population_shape:
        raise ValueError(
            "number_density_upper must have shape "
            f"{expected_population_shape}, got {number_density_upper.shape}."
        )

    integrated_absorption = (
        einstein_a[None, :]
        / (8.0 * jnp.pi * ccgs * nu_lines[None, :] ** 2)
        * (
            g_upper[None, :] / g_lower[None, :] * number_density_lower
            - number_density_upper
        )
    )
    integrated_emissivity_pi = (
        hcgs
        * ccgs
        * nu_lines[None, :]
        / 4.0
        * einstein_a[None, :]
        * number_density_upper
    )

    alpha_line = jnp.einsum("ln,lnw->lw", integrated_absorption, line_profile)
    eta_pi_line = jnp.einsum("ln,lnw->lw", integrated_emissivity_pi, line_profile)
    return alpha_line, eta_pi_line
