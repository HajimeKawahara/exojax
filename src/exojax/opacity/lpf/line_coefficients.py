"""Line extinction and emissivity from supplied level populations."""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit

from exojax.utils.constants import ccgs, hcgs

__all__ = ["line_coefficients"]


@jit
def line_coefficients(
    profiles,
    nu_lines,
    einstein_a,
    g_lower,
    g_upper,
    lower_population,
    upper_population,
):
    """Compute extinction and emissivity without assuming LTE populations.

    Args:
        profiles: Profiles in cm with shape ``(..., Nline, Nnu)``, normalized
            to unit integral over wavenumber. Absorption and emission share
            the same profile; a finite spectral window may truncate its wings.
        nu_lines: Rest line centers in cm-1, shaped ``(Nline,)``.
        einstein_a: Spontaneous transition rates in s-1, shaped ``(Nline,)``.
        g_lower: Lower-level statistical weights, shaped ``(Nline,)``.
        g_upper: Upper-level statistical weights, shaped ``(Nline,)``.
        lower_population: Lower-level number densities in cm-3, shaped
            ``(..., Nline)``. These are populations, not quantum numbers.
        upper_population: Upper-level number densities with the same units
            and broadcast-compatible shape as ``lower_population``.

    Returns:
        Extinction in cm-1 and emissivity in erg s-1 cm-3 sr-1 per cm-1,
        each shaped ``(..., Nnu)``. Leading dimensions are broadcast.

    Notes:
        Extinction includes stimulated emission and is signed: population
        inversion produces negative extinction. No division by extinction or
        clipping is performed. The caller selects an appropriate RT solver.
        Transition prefactors use rest line centers (the narrow-line
        approximation), with no relativistic transformation of coefficients.
        For positive extinction, emissivity / extinction is the physical
        source intensity; multiply by pi for ExoJAX's piB flux convention.
    """
    profiles = jnp.asarray(profiles)
    nu_lines = jnp.asarray(nu_lines)
    einstein_a = jnp.asarray(einstein_a)
    g_lower, g_upper = jnp.asarray(g_lower), jnp.asarray(g_upper)
    lower_population = jnp.asarray(lower_population)
    upper_population = jnp.asarray(upper_population)

    if profiles.ndim < 2:
        raise ValueError("profiles must have shape (..., Nline, Nnu).")
    line_shape = (profiles.shape[-2],)
    for name, values in (
        ("nu_lines", nu_lines), ("einstein_a", einstein_a),
        ("g_lower", g_lower), ("g_upper", g_upper),
    ):
        if values.shape != line_shape:
            raise ValueError(f"{name} must have shape {line_shape}.")
    for name, values in (
        ("lower_population", lower_population),
        ("upper_population", upper_population),
    ):
        if values.ndim == 0 or values.shape[-1:] != line_shape:
            raise ValueError(f"{name} must have shape (..., {line_shape[0]}).")
    lower_population, upper_population = jnp.broadcast_arrays(
        lower_population, upper_population
    )

    absorption_weight = einstein_a / (8.0 * jnp.pi * ccgs * nu_lines**2) * (
        g_upper / g_lower * lower_population - upper_population
    )
    emission_weight = (
        hcgs * ccgs * nu_lines * einstein_a / (4.0 * jnp.pi) * upper_population
    )
    extinction = jnp.sum(absorption_weight[..., :, None] * profiles, axis=-2)
    emissivity = jnp.sum(emission_weight[..., :, None] * profiles, axis=-2)
    return extinction, emissivity
