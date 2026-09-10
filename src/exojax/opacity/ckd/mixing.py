"""Random overlap with resorting and rebinning of CKD optical depths."""

import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap


def validate_ckd_mixture_tables(opas):
    """Check the common spectral bands and quadrature of prepared OpaCKDs.

    Call this host-side function once, before evaluating a JIT-compiled model.
    Band centers, band edges, their order, and quadrature weights must agree
    exactly. Temperature and pressure grids may differ: each table is
    interpolated separately, within its own domain. The function does not
    resample tables or inspect their broadening assumptions.

    Args:
        opas: Nonempty iterable of prepared :class:`OpaCKD` calculators.

    Raises:
        ValueError: If a table is unprepared, malformed, or incompatible.
    """
    opas = tuple(opas)
    if not opas:
        raise ValueError("At least one CKD table is required.")

    reference = None
    for index, opa in enumerate(opas):
        info = opa.ckd_info
        if not opa.ready or info is None:
            raise ValueError(f"CKD table {index} is not prepared.")
        weights = np.asarray(info.weights)
        centers = np.asarray(info.nu_bands)
        edges = np.asarray(info.band_edges)
        if (
            weights.ndim != 1 or weights.size == 0
            or not np.all(np.isfinite(weights)) or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1.0, rtol=1.e-5, atol=1.e-8)
        ):
            raise ValueError("CKD weights must be positive, finite, and sum to one.")
        if (
            centers.ndim != 1 or centers.size == 0
            or not np.all(np.isfinite(centers)) or np.any(centers <= 0)
            or np.any(np.diff(centers) <= 0)
        ):
            raise ValueError("CKD nu_bands must be positive and strictly increasing.")
        if (
            edges.shape != (centers.size, 2)
            or not np.all(np.isfinite(edges)) or np.any(edges <= 0)
            or np.any(edges[:, 1] <= edges[:, 0])
            or np.any(centers < edges[:, 0]) or np.any(centers > edges[:, 1])
        ):
            raise ValueError("CKD band_edges must bound each band center.")
        expected_shape = (
            np.size(info.T_grid), np.size(info.P_grid), weights.size, centers.size
        )
        if (
            opa.Ng != weights.size or np.shape(info.ggrid) != weights.shape
            or np.ndim(info.T_grid) != 1 or np.ndim(info.P_grid) != 1
            or 0 in expected_shape or info.log_kggrid.shape != expected_shape
        ):
            raise ValueError("CKD table shape must match its T, P, g, and band axes.")
        if (
            not np.array_equal(opa.nu_bands, centers)
            or not np.array_equal(opa.band_edges, edges)
        ):
            raise ValueError("CKD calculator bands must match its table metadata.")
        current = (weights, centers, edges)
        if reference is not None:
            for name, actual, expected in zip(
                ("weights", "nu_bands", "band_edges"), current, reference
            ):
                if not np.array_equal(actual, expected):
                    raise ValueError(f"CKD mixture tables must share identical {name}.")
        reference = current


def _mix_two(a, b, weights):
    """Combine one band in one layer and average over probability intervals."""
    values = (a[:, None] + b[None, :]).ravel()
    probabilities = (weights[:, None] * weights[None, :]).ravel()
    order = jnp.argsort(values, stable=True)
    values, probabilities = values[order], probabilities[order]
    zero = jnp.zeros(1, dtype=weights.dtype)
    cdf = jnp.concatenate((zero, jnp.cumsum(probabilities)))
    # Integrating residuals preserves constant distributions and reduces
    # cancellation in the difference of cumulative areas.
    offset = values[0]
    area = jnp.concatenate((zero, jnp.cumsum(probabilities * (values - offset))))
    edges = jnp.concatenate((zero, jnp.cumsum(weights)))

    # Snap roundoff-sized boundary differences before interpolating the area.
    # This prevents an opaque sample from leaking into a transparent bin when
    # equivalent cumulative weights were summed in different orders. Only the
    # query is snapped, so no new duplicate interpolation knots are introduced.
    upper = jnp.clip(jnp.searchsorted(cdf, edges), 1, cdf.size - 1)
    lower = upper - 1
    nearest = jnp.where(edges - cdf[lower] <= cdf[upper] - edges, lower, upper)
    tolerance = 4 * jnp.finfo(weights.dtype).eps
    area_at_edges = jnp.where(
        jnp.abs(edges - cdf[nearest]) <= tolerance,
        area[nearest],
        jnp.interp(edges, cdf, area),
    )
    # Interpolate the integral of tau(g), not tau at quadrature nodes. This is
    # the weighted bin average without an (Ng, Ng**2) overlap matrix.
    return offset + jnp.diff(area_at_edges) / weights


@jit
def mix_ckd_rorr(dtau_species, weights):
    """Mix CKD optical depths with random overlap, resorting, and rebinning.

    Each layer and band is mixed independently. Two species produce all Ng**2
    sums with product probabilities; sorting and linear probability-bin
    averaging reduce them to Ng terms. Further species are added in input
    order. The returned terms are bin averages, not samples at the original
    Gauss nodes, and are integrated using the supplied weights.

    The function supports ``jit``, ``vmap``, and forward/reverse differentiation
    with respect to optical depths (including their abundance, temperature,
    and pressure dependencies). The underlying conservative rebinning is
    continuous and piecewise differentiable; derivatives can change at sorting
    ties. Species count, input order, and quadrature weights must remain fixed
    when differentiating a model.

    Args:
        dtau_species: Finite nonnegative, dimensionless optical depths, shape
            ``(Nspecies, Nlayer, Ng, Nband)``. Abundances must already be applied,
            for example using ``layer_optical_depth_ckd``. All axes are nonempty.
        weights: Common positive normalized quadrature weights, shape ``(Ng,)``.
            Normalization is reapplied internally to remove summation roundoff.

    Returns:
        Mixed optical depths with shape ``(Nlayer, Ng, Nband)``, suitable for
        the existing CKD radiative-transfer solvers and the same weights.

    Notes:
        Runtime validation checks shapes only. Use
        ``validate_ckd_mixture_tables`` during setup to check table compatibility.
        No logarithm, abundance normalization, or removal of zero species is
        performed. A single species is returned unchanged.

        Random overlap approximates spectral overlap. Rebinning preserves the
        weighted mean optical depth to floating-point precision, not exact
        transmission, and three or more species have order-dependent errors.
        Using corresponding mixed g terms across layers also assumes vertical
        rank correlation of the mixture. Validate these approximations against
        line-by-line calculations for the intended atmosphere. Per layer and
        band, each added species sorts Ng**2 terms; storage does not grow
        exponentially with species count. Use x64 for precision-sensitive
        mixtures with a large optical-depth range.

        See Amundsen et al. (2017), A&A 598, A97, section 3.2.2,
        https://doi.org/10.1051/0004-6361/201629322.
    """
    dtau_species = jnp.asarray(dtau_species)
    weights = jnp.asarray(weights)
    if dtau_species.ndim != 4 or 0 in dtau_species.shape:
        raise ValueError(
            "dtau_species must have nonempty shape (Nspecies, Nlayer, Ng, Nband)."
        )
    if weights.ndim != 1 or weights.shape[0] != dtau_species.shape[2]:
        raise ValueError("weights must be one-dimensional with length Ng.")
    dtype = jnp.result_type(dtau_species, weights, jnp.float32)
    dtau_species = dtau_species.astype(dtype)
    weights = weights.astype(dtype)
    if dtau_species.shape[0] == 1:
        return dtau_species[0]
    weights = weights / jnp.sum(weights)
    species = jnp.moveaxis(dtau_species, 2, -1)
    mix_layers = vmap(vmap(_mix_two, in_axes=(0, 0, None)), in_axes=(0, 0, None))

    def add_species(mixed, next_species):
        return mix_layers(mixed, next_species, weights), None

    mixed, _ = lax.scan(add_species, species[0], species[1:])
    return jnp.moveaxis(mixed, -1, 1)
