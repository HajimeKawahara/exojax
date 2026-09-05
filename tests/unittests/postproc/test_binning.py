import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.postproc.binning import SpectralBinningOperator
from exojax.postproc.binning import apply_bin_operator
from exojax.postproc.binning import band_mean_bin_operator
from exojax.postproc.binning import piecewise_linear_bin_operator
from exojax.postproc.response import sampling_band_integral


def test_piecewise_linear_bin_operator_integrates_affine_spectrum():
    coordinate = np.array([1.0, 2.0, 4.0])
    target_edges = np.array([[2.5, 3.5], [1.25, 1.75]])
    operator = piecewise_linear_bin_operator(coordinate, target_edges)
    spectrum = 2.0 * coordinate + 3.0

    binned = apply_bin_operator(operator, spectrum)

    expected = 2.0 * np.mean(target_edges, axis=1) + 3.0
    np.testing.assert_allclose(np.asarray(binned), expected, rtol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(apply_bin_operator(operator, np.ones_like(coordinate))),
        np.ones(target_edges.shape[0]),
        rtol=1.0e-6,
    )


def test_piecewise_linear_bin_operator_supports_batch_and_gradient():
    coordinate = np.array([1.0, 2.0, 4.0])
    operator = piecewise_linear_bin_operator(
        coordinate,
        np.array([[1.0, 2.0], [2.0, 4.0]]),
    )
    spectrum = jnp.asarray(
        np.stack((2.0 * coordinate + 3.0, -coordinate + 4.0))
    )

    binned = apply_bin_operator(operator, spectrum)
    gradient = jax.grad(
        lambda values: jnp.sum(apply_bin_operator(operator, values))
    )(spectrum[0])

    np.testing.assert_allclose(
        np.asarray(binned),
        [[6.0, 9.0], [2.5, 1.0]],
        rtol=1.0e-6,
    )
    expected_gradient = np.zeros(coordinate.size)
    np.add.at(
        expected_gradient,
        np.asarray(operator.source_indices),
        np.asarray(operator.weights),
    )
    np.testing.assert_allclose(
        np.asarray(gradient), expected_gradient, rtol=1.0e-6
    )


def test_piecewise_linear_bin_operator_matches_sampling_band_integral():
    wavelength_angstrom = np.linspace(24900.0, 25000.0, 1001)
    wavenumber = 1.0e8 / wavelength_angstrom[::-1]
    spectrum_wavenumber_order = (
        3.0
        + 0.1 * np.sin(wavelength_angstrom[::-1] / 3.0)
        + 0.02 * np.cos(wavelength_angstrom[::-1] / 0.7)
    )
    target_edges = np.array([[24920.5, 24933.4], [24970.2, 24991.7]])

    operator = piecewise_linear_bin_operator(wavelength_angstrom, target_edges)
    binned = apply_bin_operator(operator, spectrum_wavenumber_order[::-1])
    legacy = sampling_band_integral(
        wavenumber,
        spectrum_wavenumber_order,
        target_edges[:, 0],
        target_edges[:, 1],
    )

    np.testing.assert_allclose(
        np.asarray(binned), np.asarray(legacy), rtol=5.0e-6
    )


def test_piecewise_linear_bin_operator_is_stable_for_narrow_float32_bin():
    coordinate = np.linspace(0.5, 20.0, 100001)
    target_edges = np.array([[19.0001, 19.0002]])
    operator = piecewise_linear_bin_operator(coordinate, target_edges)
    spectrum = jnp.asarray(10000.0 + 2.0 * coordinate, dtype=jnp.float32)

    binned = apply_bin_operator(operator, spectrum)

    expected = 10000.0 + np.sum(target_edges)
    np.testing.assert_allclose(
        np.asarray(binned), [expected], atol=0.01, rtol=0.0
    )


def test_piecewise_linear_bin_operator_stabilizes_long_float32_bin():
    coordinate = np.linspace(0.5, 20.0, 100001)
    operator = piecewise_linear_bin_operator(
        coordinate,
        np.array([[coordinate[0], coordinate[-1]]]),
    )
    spectra = jnp.asarray(
        np.stack(
            (
                np.ones(coordinate.size),
                10000.0 + 2.0 * coordinate,
            )
        ),
        dtype=jnp.float32,
    )

    binned = apply_bin_operator(operator, spectra)

    np.testing.assert_array_equal(np.asarray(binned[0]), [1.0])
    expected_affine = 10000.0 + coordinate[0] + coordinate[-1]
    np.testing.assert_allclose(
        np.asarray(binned[1]), [expected_affine], atol=0.002, rtol=0.0
    )


@pytest.mark.parametrize(
    "coordinate, message",
    [
        ([[1.0, 2.0]], "at least two"),
        ([1.0], "at least two"),
        ([1.0, np.nan], "finite"),
        ([1.0, 1.0], "strictly increasing"),
        ([2.0, 1.0], "strictly increasing"),
    ],
)
def test_piecewise_linear_bin_operator_validates_coordinate(
    coordinate, message
):
    with pytest.raises(ValueError, match=message):
        piecewise_linear_bin_operator(coordinate, np.array([[1.0, 1.5]]))


@pytest.mark.parametrize(
    "target_edges, message",
    [
        ([1.0, 2.0], "shape"),
        ([], "shape"),
        ([[1.0, np.nan]], "finite"),
        ([[1.0, 1.0]], "positive width"),
        ([[0.9, 1.5]], "inside"),
        ([[1.5, 2.1]], "inside"),
    ],
)
def test_piecewise_linear_bin_operator_validates_target_edges(
    target_edges, message
):
    with pytest.raises(ValueError, match=message):
        piecewise_linear_bin_operator(np.array([1.0, 2.0]), target_edges)


def test_piecewise_linear_bin_operator_does_not_clip_large_coordinates():
    coordinate = np.array([1.0e12, 1.0e12 + 1.0])
    target_edges = np.array([[1.0e12 - 0.001, 1.0e12 + 0.001]])

    with pytest.raises(ValueError, match="inside"):
        piecewise_linear_bin_operator(coordinate, target_edges)


def test_band_mean_bin_operator_uses_finite_band_overlap():
    source_edges = np.array([[1.0, 1.5], [1.5, 2.0], [2.0, 2.5]])
    target_edges = np.array([[1.25, 1.75], [1.75, 2.25]])
    operator = band_mean_bin_operator(source_edges, target_edges)
    band_means = jnp.asarray([[10.0, 20.0, 40.0], [1.0, 1.0, 1.0]])

    binned = apply_bin_operator(operator, band_means)

    np.testing.assert_allclose(
        np.asarray(binned),
        [[15.0, 30.0], [1.0, 1.0]],
        rtol=1.0e-6,
    )


@pytest.mark.parametrize(
    "source_edges, target_edges, message",
    [
        (
            [[1.0, 1.6], [1.5, 2.0]],
            [[1.2, 1.8]],
            "non-overlapping",
        ),
        (
            [[1.0, 1.4], [1.6, 2.0]],
            [[1.2, 1.8]],
            "completely covered",
        ),
        (
            [[1.0, 1.5], [1.5, 2.0]],
            [[0.9, 1.1]],
            "completely covered",
        ),
        (
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.5, 2.5 + 1.0e-15]],
            "completely covered",
        ),
    ],
)
def test_band_mean_bin_operator_rejects_invalid_coverage(
    source_edges, target_edges, message
):
    with pytest.raises(ValueError, match=message):
        band_mean_bin_operator(source_edges, target_edges)


def test_apply_bin_operator_validates_spectral_axis():
    operator = piecewise_linear_bin_operator(
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0]]),
    )

    with pytest.raises(ValueError, match="last spectrum dimension"):
        apply_bin_operator(operator, np.ones(2))


def test_binning_operator_is_a_jax_pytree():
    operator = piecewise_linear_bin_operator(
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0]]),
    )

    leaves = jax.tree_util.tree_leaves(operator)

    assert isinstance(operator, SpectralBinningOperator)
    assert len(leaves) == 4
