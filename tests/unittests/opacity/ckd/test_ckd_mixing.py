"""Conservative CKD mixing, limiting cases, and abundance derivatives."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity.ckd.api import OpaCKD
from exojax.opacity.ckd.contracts import CKDTableInfo
from exojax.opacity.ckd.mixing import mix_ckd_rorr, validate_ckd_mixture_tables


def _dense_reference(species, weights):
    """Integrate a sorted discrete distribution over target probability bins."""
    mixed = species[0].copy()
    target_edges = np.r_[0.0, np.cumsum(weights)]
    for added in species[1:]:
        for layer in range(mixed.shape[0]):
            for band in range(mixed.shape[2]):
                values = (
                    mixed[layer, :, band, None] + added[layer, None, :, band]
                ).ravel()
                probabilities = np.outer(weights, weights).ravel()
                order = np.argsort(values, kind="stable")
                values, probabilities = values[order], probabilities[order]
                edges = np.r_[0.0, np.cumsum(probabilities)]
                overlap = np.maximum(
                    0.0,
                    np.minimum(target_edges[1:, None], edges[None, 1:])
                    - np.maximum(target_edges[:-1, None], edges[None, :-1]),
                )
                mixed[layer, :, band] = overlap @ values / weights
    return mixed


@pytest.mark.parametrize("nspecies", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_probability_overlap_reference_and_preserves_mean(nspecies, dtype):
    jax.config.update("jax_enable_x64", dtype == np.float64)
    weights = (np.polynomial.legendre.leggauss(4)[1] / 2).astype(dtype)
    species = np.sort(
        np.random.default_rng(27).uniform(0.01, 5.0, (nspecies, 2, 4, 3)), axis=2
    ).astype(dtype)

    actual = jax.jit(mix_ckd_rorr)(jnp.asarray(species), jnp.asarray(weights))
    expected = _dense_reference(species.astype(np.float64), weights.astype(np.float64))
    tolerance = 4e-6 if dtype == np.float32 else 2e-13

    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        np.einsum("lgb,g->lb", actual, weights),
        np.einsum("slgb,g->lb", species, weights),
        rtol=tolerance,
    )
    assert np.all(np.diff(actual, axis=1) >= -tolerance)
    if nspecies == 2:
        np.testing.assert_allclose(
            mix_ckd_rorr(species[::-1].copy(), weights), actual,
            rtol=tolerance, atol=tolerance,
        )
    else:
        reverse_order = mix_ckd_rorr(species[::-1].copy(), weights)
        np.testing.assert_allclose(
            reverse_order,
            _dense_reference(
                species[::-1].astype(np.float64), weights.astype(np.float64)
            ),
            rtol=tolerance, atol=tolerance,
        )
        assert not np.allclose(reverse_order, actual, rtol=1e-3)


def test_single_species_and_gray_absorber_limits():
    weights = jnp.array([0.2, 0.3, 0.5])
    species = jnp.array([0.03, 0.7, 9.0])[None, :, None]
    np.testing.assert_array_equal(mix_ckd_rorr(species[None], weights), species)
    for gray_value in [0.0, 0.4]:
        gray = jnp.full_like(species, gray_value)
        np.testing.assert_allclose(
            mix_ckd_rorr(jnp.stack([species, gray]), weights),
            species + gray_value,
            rtol=2e-13,
        )
    np.testing.assert_array_equal(
        mix_ckd_rorr(jnp.zeros((3, 2, 3, 2)), weights), np.zeros((2, 3, 2))
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_transparent_species_does_not_leak_opaque_tail(dtype):
    jax.config.update("jax_enable_x64", dtype == np.float64)
    weights = jnp.asarray(np.polynomial.legendre.leggauss(8)[1] / 2, dtype=dtype)
    absorbing = jnp.asarray([1.0] * 7 + [1e20], dtype=dtype)[None, :, None]
    actual = mix_ckd_rorr(jnp.stack([absorbing, jnp.zeros_like(absorbing)]), weights)
    tolerance = 2e-5 if dtype == np.float32 else 2e-13
    np.testing.assert_allclose(actual, absorbing, rtol=tolerance, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gray_mixtures_with_one_or_many_terms_and_batched_vmap(dtype):
    jax.config.update("jax_enable_x64", dtype == np.float64)
    for ng in [1, 64]:
        weights = jnp.asarray(np.polynomial.legendre.leggauss(ng)[1] / 2, dtype=dtype)
        species = jnp.broadcast_to(
            jnp.asarray([1.0, 2.0, 3.0], dtype=dtype)[:, None, None, None],
            (3, 1, ng, 1),
        )
        batches = jnp.stack([species, 2 * species])
        actual = jax.jit(jax.vmap(mix_ckd_rorr, in_axes=(0, None)))(batches, weights)
        expected = jnp.sum(batches, axis=1)
        np.testing.assert_array_equal(actual, expected)


def test_single_layer_transmission_against_full_random_overlap():
    weights = jnp.array([0.2, 0.3, 0.5])
    species = jnp.array([[0.01, 0.1, 0.2], [0.02, 0.07, 0.16]])
    mixed = mix_ckd_rorr(species[:, None, :, None], weights)[0, :, 0]
    cartesian_tau = species[0, :, None] + species[1, None, :]
    exact = jnp.sum(weights[:, None] * weights[None, :] * jnp.exp(-cartesian_tau))
    transmission = jnp.dot(weights, jnp.exp(-mixed))

    np.testing.assert_allclose(exact, jnp.prod(jnp.exp(-species) @ weights), rtol=1e-14)
    # Averaging optical depth lowers transmission by convexity of exp(-tau).
    assert transmission <= exact
    np.testing.assert_allclose(transmission, exact, rtol=1e-3)


def test_jitted_abundance_derivatives_match_finite_differences():
    weights = jnp.asarray(np.polynomial.legendre.leggauss(4)[1] / 2)
    species = jnp.array([[0.02, 0.3, 2.0, 20.0], [0.2, 1.0, 3.0, 12.0]])

    def transmission(abundance):
        dtau = (species * abundance[:, None])[:, None, :, None]
        return jnp.dot(weights, jnp.exp(-mix_ckd_rorr(dtau, weights)[0, :, 0]))

    abundance = jnp.array([0.8, 0.4])
    reverse = jax.jit(jax.grad(transmission))(abundance)
    forward = jax.jit(jax.jacfwd(transmission))(abundance)
    step = 1e-5
    finite_difference = jnp.array([
        (transmission(abundance + step * direction)
         - transmission(abundance - step * direction)) / (2 * step)
        for direction in jnp.eye(2)
    ])

    np.testing.assert_allclose(reverse, forward, rtol=1e-12)
    np.testing.assert_allclose(reverse, finite_difference, rtol=2e-7, atol=1e-10)


def test_zero_abundance_keeps_absorption_gradient():
    weights = jnp.array([0.2, 0.3, 0.5])
    species = jnp.array([[0.03, 0.7, 9.0], [0.1, 0.4, 2.0]])

    def transmission(abundance):
        dtau = (species * abundance[:, None])[:, None, :, None]
        return jnp.dot(weights, jnp.exp(-mix_ckd_rorr(dtau, weights)[0, :, 0]))

    derivative = jax.jit(jax.grad(transmission))
    np.testing.assert_allclose(
        derivative(jnp.zeros(2)), -(species @ weights), rtol=1e-13
    )
    abundance = jnp.array([0.7, 0.0])
    expected = -transmission(abundance) * jnp.dot(species[1], weights)
    np.testing.assert_allclose(derivative(abundance)[1], expected, rtol=1e-13)
    step = 1e-7
    one_sided = (transmission(abundance + jnp.array([0.0, step]))
                 - transmission(abundance)) / step
    np.testing.assert_allclose(derivative(abundance)[1], one_sided, rtol=2e-6)


def test_mixed_optical_depth_is_continuous_when_sort_order_changes():
    weights = jnp.array([0.3, 0.7])

    def crossing(scale):
        species = jnp.stack([jnp.array([0.0, 1.0]), jnp.array([0.0, scale])])
        return mix_ckd_rorr(species[:, None, :, None], weights)

    step = 1e-8
    np.testing.assert_allclose(
        crossing(1.0 - step), crossing(1.0 + step), atol=4 * step, rtol=0
    )


@pytest.mark.parametrize("shape,weight_shape", [
    ((2, 3, 4), (3,)),
    ((0, 1, 3, 1), (3,)),
    ((2, 0, 3, 1), (3,)),
    ((2, 1, 0, 1), (0,)),
    ((2, 1, 3, 0), (3,)),
    ((2, 1, 3, 1), (2,)),
    ((2, 1, 3, 1), (1, 3)),
])
def test_invalid_shapes_raise(shape, weight_shape):
    with pytest.raises(ValueError):
        mix_ckd_rorr(jnp.zeros(shape), jnp.ones(weight_shape))


def _table():
    opa = OpaCKD.load_only()
    opa.ready = True
    opa.Ng = 3
    opa.nu_bands = jnp.array([1000.0, 1100.0])
    opa.band_edges = jnp.array([[950.0, 1050.0], [1050.0, 1150.0]])
    opa.ckd_info = CKDTableInfo(
        log_kggrid=jnp.zeros((2, 2, 3, 2)),
        ggrid=jnp.array([0.1, 0.4, 0.8]),
        weights=jnp.array([0.2, 0.3, 0.5]),
        T_grid=jnp.array([500.0, 1000.0]),
        P_grid=jnp.array([0.1, 1.0]),
        nu_bands=opa.nu_bands,
        band_edges=opa.band_edges,
    )
    return opa


def test_table_validation_allows_independent_temperature_pressure_grids():
    first, second = _table(), _table()
    second.ckd_info = replace(
        second.ckd_info,
        T_grid=jnp.array([300.0, 700.0, 1500.0]),
        P_grid=jnp.array([0.01, 10.0]),
        log_kggrid=jnp.zeros((3, 2, 3, 2)),
    )
    assert validate_ckd_mixture_tables([first, second]) is None


@pytest.mark.parametrize("field,value", [
    ("band_edges", [[949.0, 1050.0], [1050.0, 1151.0]]),
    ("band_edges", [950.0, 1050.0, 1150.0]),
    ("nu_bands", [1001.0, 1100.0]),
    ("nu_bands", [1100.0, 1000.0]),
    ("weights", [0.3, 0.2, 0.5]),
    ("weights", [0.0, 0.5, 0.5]),
    ("weights", [0.2, 0.3, 0.4]),
    ("weights", [0.2, np.nan, 0.8]),
    ("ggrid", [0.1, 0.4]),
    ("log_kggrid", np.zeros((2, 2, 4, 2))),
])
def test_incompatible_tables_raise(field, value):
    first, second = _table(), _table()
    second.ckd_info = replace(second.ckd_info, **{field: jnp.asarray(value)})
    if field in {"band_edges", "nu_bands"}:
        setattr(second, field, jnp.asarray(value))
    with pytest.raises(ValueError):
        validate_ckd_mixture_tables([first, second])


def test_table_validation_rejects_empty_or_unprepared_input():
    with pytest.raises(ValueError):
        validate_ckd_mixture_tables([])
    with pytest.raises((ValueError, RuntimeError)):
        validate_ckd_mixture_tables([OpaCKD.load_only()])
    malformed = _table()
    malformed.Ng = 4
    with pytest.raises(ValueError):
        validate_ckd_mixture_tables([malformed])
    inconsistent = _table()
    inconsistent.band_edges = inconsistent.band_edges + 1.0
    with pytest.raises(ValueError):
        validate_ckd_mixture_tables([inconsistent])
