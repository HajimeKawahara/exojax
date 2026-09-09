"""Pure-absorption radiation at every atmospheric interface."""

import jax.numpy as jnp
from jax import jit, lax

from exojax.rt.rtransfer import coeffs_linsap


@jit
def rtrun_emis_pureabs_ibased_linsap_fluxes(
    dtau,
    source_boundary,
    mus,
    weights,
    incoming_top=0.0,
    outgoing_bottom=None,
    *,
    source_center=None,
    upper_fraction=0.5,
):
    """Compute upward and downward thermal fluxes with a linear source.

    Layers and interfaces are ordered from top to bottom. The source is linear
    in optical depth between supplied source points. Sources and boundary
    intensities use the ExoJAX ``pi * B`` convention, so angular integration uses
    ``2 * mu``.
    Boundary intensities are isotropic over their incoming hemisphere.

    Args:
        dtau: Nonnegative layer optical depths, shape ``(Nlayer, ...)``.
        source_boundary: Interface sources, broadcastable to ``(Nlayer+1, ...)``.
            For spectral fluxes, use ``piBarr`` in erg/s/cm2/(cm-1).
        mus: Positive hemisphere quadrature cosines, shape ``(Nangle,)``.
        weights: Quadrature weights on [0, 1], shape ``(Nangle,)``.
        incoming_top: Downward pi-scaled intensity at the top, broadcastable to
            the trailing spectral dimensions. Defaults to no thermal incidence.
        outgoing_bottom: Upward pi-scaled intensity at the bottom, broadcastable
            to the trailing spectral dimensions. ``None`` uses the last source,
            corresponding to a black surface at the bottom gas temperature.
        source_center: Optional layer-center sources, broadcastable to
            ``(Nlayer, ...)``. When supplied, use two linear source segments per
            layer, passing through the center source at ``upper_fraction`` of
            the layer optical depth. Returned fluxes remain at the original
            physical interfaces.
        upper_fraction: Fraction of each layer optical depth above its source
            center, in [0, 1]. A scalar or shape ``(Nlayer,)``. With opacity
            constant in each layer, use ``(Pcenter-Pupper)/(Plower-Pupper)``.
            Used only when ``source_center`` is supplied.

    Returns:
        Upward and downward flux arrays, each of shape ``(Nlayer+1, ...)``.
        Both are positive in their respective directions and have source units.
    """
    dtau = jnp.asarray(dtau)
    shape = (dtau.shape[0] + 1,) + dtau.shape[1:]
    source = jnp.broadcast_to(jnp.asarray(source_boundary), shape)
    if source_center is not None:
        center = jnp.broadcast_to(jnp.asarray(source_center), dtau.shape)
        fraction = jnp.asarray(upper_fraction)
        if fraction.ndim == 1:
            fraction = fraction.reshape((-1,) + (1,) * (dtau.ndim - 1))
        split_shape = (2 * dtau.shape[0],) + dtau.shape[1:]
        source = jnp.concatenate(
            (
                jnp.stack((source[:-1], center), axis=1).reshape(split_shape),
                source[-1:],
            ),
            axis=0,
        )
        dtau = jnp.stack(
            (fraction * dtau, (1.0 - fraction) * dtau), axis=1
        ).reshape(split_shape)
    if outgoing_bottom is None:
        outgoing_bottom = source[-1]
    dtype = jnp.result_type(
        dtau, source, mus, weights, incoming_top, outgoing_bottom
    )
    dtau = jnp.asarray(dtau, dtype=dtype)
    source = jnp.asarray(source, dtype=dtype)
    top = jnp.broadcast_to(jnp.asarray(incoming_top, dtype=dtype), dtau.shape[1:])
    bottom = jnp.broadcast_to(
        jnp.asarray(outgoing_bottom, dtype=dtype), dtau.shape[1:]
    )

    def integrate_angle(fluxes, mu_weight):
        mu, weight = mu_weight
        depth = dtau / mu
        transmission = jnp.exp(-depth)
        beta, gamma = coeffs_linsap(depth, transmission)
        emission_up = beta * source[:-1] + gamma * source[1:]
        emission_down = gamma * source[:-1] + beta * source[1:]

        def propagate(intensity, layer):
            trans, emission = layer
            intensity = trans * intensity + emission
            return intensity, intensity

        _, upward = lax.scan(
            propagate, bottom, (transmission, emission_up), reverse=True
        )
        _, downward = lax.scan(propagate, top, (transmission, emission_down))
        upward = jnp.concatenate((upward, bottom[None]), axis=0)
        downward = jnp.concatenate((top[None], downward), axis=0)
        factor = 2.0 * mu * weight
        return (fluxes[0] + factor * upward, fluxes[1] + factor * downward), None

    initial = (jnp.zeros_like(source), jnp.zeros_like(source))
    fluxes, _ = lax.scan(
        integrate_angle,
        initial,
        (jnp.asarray(mus, dtype=dtype), jnp.asarray(weights, dtype=dtype)),
    )
    if source_center is not None:
        return fluxes[0][::2], fluxes[1][::2]
    return fluxes


@jit
def direct_beam_fluxes(dtau, flux_top, mu0):
    """Attenuate a downward stellar beam at every interface.

    Args:
        dtau: Nonnegative layer absorption depths, shape ``(Nlayer, ...)``.
        flux_top: Incident flux through a horizontal surface, broadcastable to
            the trailing dimensions of ``dtau``. It already includes incidence
            geometry and must not be multiplied by ``mu0`` again.
        mu0: Positive incidence cosine, at most one.

    Returns:
        Positive downward flux, shape ``(Nlayer+1, ...)``, in ``flux_top`` units.
        The last element includes stellar energy reaching the lower boundary.
    """
    dtau = jnp.asarray(dtau)
    tau = jnp.concatenate(
        (jnp.zeros((1,) + dtau.shape[1:], dtype=dtau.dtype), jnp.cumsum(dtau, axis=0)),
        axis=0,
    )
    return jnp.asarray(flux_top) * jnp.exp(-tau / mu0)


@jit
def integrate_ckd_flux(flux, weights, band_widths):
    """Integrate CKD spectral flux over g ordinates and wavenumber bands.

    Args:
        flux: Spectral flux with trailing dimensions ``(Ng, Nband)``.
        weights: Normalized g quadrature weights, shape ``(Ng,)``.
        band_widths: Band widths in cm-1, shape ``(Nband,)``.

    Returns:
        Flux with the last two dimensions integrated out. Input in
        erg/s/cm2/(cm-1) gives output in erg/s/cm2.
    """
    return jnp.einsum("...gb,g,b->...", flux, weights, band_widths)
