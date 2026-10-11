"""Broadband sources and shared pure-absorption boundaries for the RCE example.

Spectral fluxes are per wavenumber in erg s^-1 cm^-2 (cm^-1)^-1, not per
Hz. Integrated fluxes are in erg s^-1 cm^-2. Arrays run from top to bottom.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from exojax.rt.flux import (
    direct_beam_fluxes,
    integrate_ckd_flux,
    reconstruct_boundary_temperature,
    rtrun_emis_pureabs_ibased_linsap_fluxes,
)
from exojax.rt.planck import piBarr


class Fluxes(NamedTuple):
    """Positive directional fluxes and net upward flux at every boundary."""

    upward: object
    downward: object
    incident: object
    net: object


def _validate_band_edges(band_edges):
    edges = np.asarray(band_edges, dtype=float)
    if (
        edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0
        or not np.all(np.isfinite(edges)) or np.any(edges <= 0)
        or np.any(edges[:, 1] <= edges[:, 0])
        or np.any(edges[1:, 0] < edges[:-1, 1])
    ):
        raise ValueError("Band edges must be positive, ordered, and nonoverlapping.")
    return edges


def make_band_quadrature(band_edges, order=8):
    """Return cm^-1 nodes (Nband, Nquad) and normalized within-band weights.

    This host-side setup accepts disjoint bands; the omitted intervals are not
    included in an integral. A broadband energy calculation needs full coverage.
    """
    edges = _validate_band_edges(band_edges)
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("Quadrature order must be a positive integer.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = edges[:, :1] + np.diff(edges, axis=1) * (nodes + 1.0) / 2.0
    return jnp.asarray(nodes), jnp.asarray(weights / 2.0)


def band_planck_flux(temperature, band_nodes, band_weights):
    """Average pi B over each band, preserving arbitrary temperature axes.

    Multiply these averages by band widths in cm^-1 to integrate them. The
    within-band quadrature is independent of the CKD probability quadrature.
    """
    temperature = jnp.asarray(temperature)
    nodes = jnp.asarray(band_nodes)
    source = piBarr(temperature.reshape(-1), nodes.reshape(-1))
    source = source.reshape(temperature.shape + nodes.shape)
    return jnp.sum(source * jnp.asarray(band_weights), axis=-1)


def band_stellar_flux(wavenumber, surface_flux, band_edges):
    """Conservatively average a tabulated stellar surface spectrum at setup.

    The positive, increasing wavenumbers must cover all bin edges. Integrate
    the piecewise-linear interpolant using every tabulated point inside each
    band, so narrow stellar features are not skipped by quadrature sampling.
    Flux must already be per cm^-1; no distance or angular factors are applied.
    """
    edges = _validate_band_edges(band_edges)
    nu = np.asarray(wavenumber, dtype=float)
    flux = np.asarray(surface_flux, dtype=float)
    if (
        nu.ndim != 1 or len(nu) < 2 or flux.shape != nu.shape
        or not np.all(np.isfinite(nu)) or np.any(nu <= 0)
        or np.any(np.diff(nu) <= 0)
        or not np.all(np.isfinite(flux)) or np.any(flux < 0)
    ):
        raise ValueError("Stellar flux must be finite and nonnegative on increasing wavenumbers.")
    if nu[0] > edges[0, 0] or nu[-1] < edges[-1, 1]:
        raise ValueError("Stellar spectrum must cover all band edges without extrapolation.")
    averages = []
    for lower, upper in edges:
        start, stop = np.searchsorted(nu, lower, side="right"), np.searchsorted(nu, upper)
        inner = nu[start:stop]
        points = np.concatenate(([lower], inner, [upper]))
        values = np.interp(points, nu, flux)
        integral = np.sum(np.diff(points) * (values[:-1] + values[1:]) / 2.0)
        averages.append(integral / (upper - lower))
    return jnp.asarray(averages)


def irradiation_flux(surface_flux, a_over_rstar, amplitude=1.0):
    """Apply f_irr (R_star/a)^2 to obtain the horizontal incident flux.

    ``amplitude`` is f_irr, already containing the incidence or redistribution
    convention. Only beam attenuation uses mu0; it must not multiply this flux
    again. For a local parallel beam, explicitly choose f_irr = mu0.
    """
    return jnp.asarray(surface_flux) * amplitude / a_over_rstar**2


def absorption_fluxes(
    dtau, temperature, bottom_temperature, pressure_bar,
    pressure_boundaries_bar, source_fn, mus, angular_weights,
    incident_flux=0.0, mu0=0.5,
):
    """Evaluate shared thermal and direct boundaries for CKD or resolved lines.

    ``source_fn(T)`` returns pi B with the spectral axes of ``dtau`` (or
    broadcastable axes). The lower boundary is black at bottom_temperature;
    the top has no diffuse thermal incidence. Any unabsorbed beam reaching the
    bottom remains explicit in ``incident[-1]`` and is not silently recycled.
    Opacity is held constant within each layer and the center source splits it
    at its pressure fraction. ``incident_flux`` is horizontal flux, while mu0
    changes only the slant optical depth. Downward thermal and direct fluxes
    are separate; ``net = upward - downward - incident``.
    """
    boundaries = jnp.asarray(pressure_boundaries_bar)
    pressure = jnp.asarray(pressure_bar)
    boundary_temperature = reconstruct_boundary_temperature(
        pressure, boundaries, temperature, bottom_temperature
    )
    upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
        dtau, source_fn(boundary_temperature), mus, angular_weights,
        source_center=source_fn(temperature),
        upper_fraction=(pressure - boundaries[:-1]) / jnp.diff(boundaries),
    )
    incident = direct_beam_fluxes(dtau, incident_flux, mu0)
    return Fluxes(upward, downward, incident, upward - downward - incident)


def ckd_absorption_fluxes(
    dtau, temperature, bottom_temperature, pressure_bar,
    pressure_boundaries_bar, band_nodes, band_weights, mus, angular_weights,
    incident_flux=0.0, mu0=0.5,
):
    """Evaluate spectral CKD fluxes with band-averaged Planck sources.

    dtau has shape (Nlayer, Ng, Nband), already mixed using RORR when needed.
    The return values have shape (Nlayer+1, Ng, Nband). Supply the stellar
    band averages as incident_flux; the same source applies to each g term.
    """
    def source_fn(values):
        return band_planck_flux(values, band_nodes, band_weights)[:, None, :]

    return absorption_fluxes(
        dtau, temperature, bottom_temperature, pressure_bar,
        pressure_boundaries_bar, source_fn, mus, angular_weights,
        incident_flux, mu0,
    )


def integrate_broadband_fluxes(fluxes, g_weights, band_widths):
    """Integrate every CKD directional and net boundary flux over g and cm^-1."""
    return Fluxes(*(
        integrate_ckd_flux(value, g_weights, band_widths) for value in fluxes
    ))


def bottom_leakage_fraction(fluxes):
    """Fraction of broadband incident stellar energy reaching the bottom."""
    top = fluxes.incident[0]
    denominator = jnp.where(top > 0.0, top, 1.0)
    return jnp.where(top > 0.0, fluxes.incident[-1] / denominator, 0.0)
