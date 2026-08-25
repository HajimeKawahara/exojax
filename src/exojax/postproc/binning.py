"""Spectral binning operators for fixed grids and bin edges.

The operators average in the supplied spectral coordinate.  They do not
convert between wavelength and wavenumber or apply an instrumental response.
Build an operator once, outside JIT-compiled model code, and reuse it for all
spectra on the same grid and target bins.  Configure JAX precision before
building an operator; its arrays are placed on the active JAX device.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectralBinningOperator:
    """Sparse linear operator for spectral bin averages.

    The array fields contain local, nonzero contributions.  ``input_size`` and
    ``output_size`` are static PyTree metadata, so changing only a spectrum
    does not retrace :func:`apply_bin_operator`.  Construct instances with
    the operator-building functions in this module.
    """

    weights: Any
    source_indices: Any
    target_indices: Any
    anchor_indices: Any
    input_size: int
    output_size: int

    def tree_flatten(self):
        """Flatten the operator as a JAX PyTree."""
        children = (
            self.weights,
            self.source_indices,
            self.target_indices,
            self.anchor_indices,
        )
        metadata = (self.input_size, self.output_size)
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        """Restore the operator from its JAX PyTree representation."""
        weights, source_indices, target_indices, anchor_indices = children
        input_size, output_size = metadata
        return cls(
            weights=weights,
            source_indices=source_indices,
            target_indices=target_indices,
            anchor_indices=anchor_indices,
            input_size=input_size,
            output_size=output_size,
        )


def _validated_edges(edges, name):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{name} must have shape (number_of_bins, 2).")
    if edges.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one bin.")
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(edges[:, 1] <= edges[:, 0]):
        raise ValueError(f"Every bin in {name} must have positive width.")
    return edges


def _append_weight(
    weights,
    source_indices,
    target_indices,
    weight,
    source_index,
    target_index,
):
    if weight <= 0.0:
        return
    if (
        weights
        and source_indices[-1] == source_index
        and target_indices[-1] == target_index
    ):
        weights[-1] += float(weight)
    else:
        weights.append(float(weight))
        source_indices.append(int(source_index))
        target_indices.append(int(target_index))


def _operator_from_lists(
    weights,
    source_indices,
    target_indices,
    input_size,
    output_size,
):
    target_indices = np.asarray(target_indices, dtype=np.int32)
    source_indices = np.asarray(source_indices, dtype=np.int32)
    first_contribution = np.searchsorted(
        target_indices,
        np.arange(output_size, dtype=np.int32),
    )
    return SpectralBinningOperator(
        weights=jnp.asarray(np.asarray(weights, dtype=float)),
        source_indices=jnp.asarray(source_indices),
        target_indices=jnp.asarray(target_indices),
        anchor_indices=jnp.asarray(source_indices[first_contribution]),
        input_size=int(input_size),
        output_size=int(output_size),
    )


def piecewise_linear_bin_operator(sample_coordinate, target_bin_edges):
    """Build exact top-hat averages of a piecewise-linear spectrum.

    Args:
        sample_coordinate: Strictly increasing sample coordinates, shape
            ``(number_of_samples,)``.
        target_bin_edges: Lower and upper target edges in the same coordinate
            and unit, shape ``(number_of_bins, 2)``.  Output bins preserve this
            input order.

    Returns:
        A :class:`SpectralBinningOperator`.  It is exact for the
        piecewise-linear reconstruction between the supplied samples.
    """

    coordinate = np.asarray(sample_coordinate, dtype=float)
    target_edges = _validated_edges(target_bin_edges, "target_bin_edges")
    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError("sample_coordinate must contain at least two points.")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("sample_coordinate must contain finite values.")
    if np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("sample_coordinate must be strictly increasing.")

    if np.any(target_edges[:, 0] < coordinate[0]) or np.any(
        target_edges[:, 1] > coordinate[-1]
    ):
        raise ValueError("Target bins must lie inside sample_coordinate.")

    interval_lower = coordinate[:-1]
    interval_upper = coordinate[1:]
    interval_width = interval_upper - interval_lower
    weights = []
    source_indices = []
    target_indices = []

    for target_index, (target_lower, target_upper) in enumerate(target_edges):
        first_interval = max(
            0,
            int(np.searchsorted(coordinate, target_lower, side="right")) - 1,
        )
        last_interval = min(
            coordinate.size - 1,
            int(np.searchsorted(coordinate, target_upper, side="left")),
        )
        active_indices = np.arange(first_interval, last_interval, dtype=int)
        overlap_lower = np.maximum(
            interval_lower[active_indices], target_lower
        )
        overlap_upper = np.minimum(
            interval_upper[active_indices], target_upper
        )
        overlap_width = overlap_upper - overlap_lower
        lower_fraction = (
            overlap_lower - interval_lower[active_indices]
        ) / interval_width[active_indices]
        upper_fraction = (
            overlap_upper - interval_lower[active_indices]
        ) / interval_width[active_indices]
        lower_fraction = np.clip(lower_fraction, 0.0, 1.0)
        upper_fraction = np.clip(upper_fraction, 0.0, 1.0)
        midpoint_fraction = 0.5 * (lower_fraction + upper_fraction)
        normalization = 1.0 / (target_upper - target_lower)
        left_weights = overlap_width * (1.0 - midpoint_fraction)
        right_weights = overlap_width * midpoint_fraction
        for source_index, left_weight, right_weight in zip(
            active_indices, left_weights, right_weights
        ):
            _append_weight(
                weights,
                source_indices,
                target_indices,
                left_weight * normalization,
                source_index,
                target_index,
            )
            _append_weight(
                weights,
                source_indices,
                target_indices,
                right_weight * normalization,
                source_index + 1,
                target_index,
            )

    return _operator_from_lists(
        weights,
        source_indices,
        target_indices,
        coordinate.size,
        target_edges.shape[0],
    )


def band_mean_bin_operator(source_band_edges, target_bin_edges):
    """Build overlap averages of finite-width source-band means.

    Args:
        source_band_edges: Ordered, non-overlapping source-band edges, shape
            ``(number_of_source_bands, 2)``.
        target_bin_edges: Target edges in the same coordinate and unit, shape
            ``(number_of_target_bins, 2)``.  Output bins preserve this input
            order.

    Returns:
        A :class:`SpectralBinningOperator`.  It is exact when each source value
        is treated as constant inside its finite band.
    """

    source_edges = _validated_edges(source_band_edges, "source_band_edges")
    target_edges = _validated_edges(target_bin_edges, "target_bin_edges")
    if np.any(np.diff(source_edges[:, 0]) < 0.0) or np.any(
        source_edges[1:, 0] < source_edges[:-1, 1]
    ):
        raise ValueError(
            "source_band_edges must be ordered and non-overlapping."
        )

    weights = []
    source_indices = []
    target_indices = []
    source_lower = source_edges[:, 0]
    source_upper = source_edges[:, 1]

    for target_index, (target_lower, target_upper) in enumerate(target_edges):
        first_band = int(
            np.searchsorted(source_upper, target_lower, side="right")
        )
        last_band = int(
            np.searchsorted(source_lower, target_upper, side="left")
        )
        active_indices = np.arange(first_band, last_band, dtype=int)
        overlap = np.maximum(
            0.0,
            np.minimum(source_upper[active_indices], target_upper)
            - np.maximum(source_lower[active_indices], target_lower),
        )
        requested_width = float(target_upper - target_lower)
        if not np.isclose(
            np.sum(overlap),
            requested_width,
            rtol=128.0 * np.finfo(float).eps,
            atol=0.0,
        ):
            raise ValueError(
                "Every target bin must be completely covered by source bands."
            )
        for local_index in np.flatnonzero(overlap > 0.0):
            source_index = active_indices[local_index]
            _append_weight(
                weights,
                source_indices,
                target_indices,
                overlap[local_index] / requested_width,
                source_index,
                target_index,
            )

    return _operator_from_lists(
        weights,
        source_indices,
        target_indices,
        source_edges.shape[0],
        target_edges.shape[0],
    )


@jax.jit
def apply_bin_operator(operator, spectrum):
    """Apply a spectral binning operator along the last array axis.

    Leading dimensions are treated as independent spectra.  The operation is
    differentiable with respect to ``spectrum``; bin geometry is static setup
    data represented by ``operator``.
    """

    spectrum = jnp.asarray(spectrum)
    if spectrum.ndim == 0 or spectrum.shape[-1] != operator.input_size:
        raise ValueError(
            "The last spectrum dimension must match operator.input_size."
        )
    flattened = spectrum.reshape((-1, operator.input_size))
    anchors = flattened[:, operator.anchor_indices]
    contributions = (
        flattened[:, operator.source_indices]
        - anchors[:, operator.target_indices]
    ) * operator.weights
    binned_offsets = jax.ops.segment_sum(
        contributions.T,
        operator.target_indices,
        num_segments=operator.output_size,
        indices_are_sorted=True,
    ).T
    binned = anchors + binned_offsets
    return binned.reshape(spectrum.shape[:-1] + (operator.output_size,))
