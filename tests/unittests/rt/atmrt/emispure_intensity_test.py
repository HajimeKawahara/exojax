import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad, jacfwd, jit

from exojax.rt import ArtEmisPure
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import (
    coeffs_linsap,
    initialize_gaussian_quadrature,
    rtrun_emis_pureabs_ibased,
    rtrun_emis_pureabs_ibased_flux_from_intensity,
    rtrun_emis_pureabs_ibased_intensity,
    rtrun_emis_pureabs_ibased_intensity_surface,
)


@pytest.mark.parametrize("optical_depth", [0.0, 1.0e-9, 1.0e-3, 1.0])
def test_ibased_float32_slab_flux_and_gradient(optical_depth):
    mus, weights = initialize_gaussian_quadrature(8)
    mus = jnp.asarray(mus, dtype=jnp.float32)
    weights = jnp.asarray(weights, dtype=jnp.float32)

    def flux(depth):
        dtau = depth.reshape((1, 1))
        return rtrun_emis_pureabs_ibased(
            dtau, jnp.ones_like(dtau), mus, weights
        )[0]

    depth = jnp.float32(optical_depth)
    mu64 = np.asarray(mus, dtype=np.float64)
    weight64 = np.asarray(weights, dtype=np.float64)
    expected = np.sum(2.0 * mu64 * weight64 * -np.expm1(-float(depth) / mu64))
    expected_gradient = np.sum(2.0 * weight64 * np.exp(-float(depth) / mu64))

    assert flux(depth) == pytest.approx(expected, rel=1.0e-6, abs=0.0)
    assert grad(flux)(depth) == pytest.approx(expected_gradient, rel=1.0e-6)


def test_ibased_thin_float32_layer_below_absorbing_layer():
    dtau = jnp.array([[1.0], [1.0e-9]], dtype=jnp.float32)
    source = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
    mus = jnp.array([0.5], dtype=jnp.float32)

    intensity = rtrun_emis_pureabs_ibased_intensity(dtau, source, mus)
    expected = np.exp(-2.0) * -np.expm1(-2.0e-9)

    assert intensity == pytest.approx(np.array([[expected]]), rel=1.0e-6, abs=0.0)


def test_linsap_coefficients_at_small_optical_depth():
    x = jnp.array([0.0, 1.0e-8], dtype=jnp.float32)
    expected = jnp.array([0.0, 5.0e-9], dtype=jnp.float32)

    beta, gamma = jit(coeffs_linsap)(x, jnp.exp(-x))

    assert beta == pytest.approx(expected, rel=1.0e-6, abs=0.0)
    assert gamma == pytest.approx(expected, rel=1.0e-6, abs=0.0)

    derivatives = jacfwd(
        lambda value: jnp.stack(coeffs_linsap(value, jnp.exp(-value)))
    )(jnp.float32(0.0))
    assert derivatives == pytest.approx(jnp.full(2, 0.5), rel=1.0e-6)


def test_ibased_intensity_reconstructs_flux():
    nlayer = 4
    nnus = 3
    dtau = jnp.full((nlayer, nnus), 0.1)
    source_matrix = jnp.arange(1, nlayer * nnus + 1, dtype=float).reshape(nlayer, nnus)
    art = ArtEmisPure(nlayer=nlayer, nu_grid=jnp.arange(nnus) + 1000.0, nstream=8)

    intensity = rtrun_emis_pureabs_ibased_intensity(dtau, source_matrix, art.mus)
    flux_from_intensity = rtrun_emis_pureabs_ibased_flux_from_intensity(
        intensity, art.mus, art.weights
    )
    flux = rtrun_emis_pureabs_ibased(dtau, source_matrix, art.mus, art.weights)

    assert flux_from_intensity == pytest.approx(flux)


def test_ibased_intensity_surface_adds_attenuated_lower_boundary():
    dtau = jnp.array([[0.2, 0.4], [0.3, 0.1]])
    source_matrix = jnp.zeros_like(dtau)
    source_surface = jnp.array([2.0, 3.0])
    mus = jnp.array([0.25, 0.75])

    actual = rtrun_emis_pureabs_ibased_intensity_surface(
        dtau, source_matrix, source_surface, mus
    )
    expected = source_surface[None, :] * jnp.exp(
        -jnp.sum(dtau, axis=0)[None, :] / mus[:, None]
    )

    assert actual == pytest.approx(expected)


def test_artemispure_run_with_limb_darkening_reuses_intensity():
    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau = jnp.full((nlayer, len(nu_grid)), 0.1)
    art = ArtEmisPure(nlayer=nlayer, nu_grid=nu_grid, nstream=8)

    flux, u1, u2 = art.run_with_limb_darkening(dtau, temperature)
    intensity = art.run_intensity(dtau, temperature)
    flux_from_intensity = rtrun_emis_pureabs_ibased_flux_from_intensity(
        intensity, art.mus, art.weights
    )

    assert flux == pytest.approx(flux_from_intensity)
    assert jnp.shape(u1) == jnp.shape(nu_grid)
    assert jnp.shape(u2) == jnp.shape(nu_grid)


def test_artemispure_run_intensity_matches_low_level():
    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau = jnp.full((nlayer, len(nu_grid)), 0.1)
    art = ArtEmisPure(nlayer=nlayer, nu_grid=nu_grid, nstream=8)

    source_matrix = piBarr(temperature, nu_grid)
    intensity = art.run_intensity(dtau, temperature)
    expected = rtrun_emis_pureabs_ibased_intensity(dtau, source_matrix, art.mus)

    assert intensity == pytest.approx(expected)


def test_artemispure_run_ckd_with_limb_darkening_reuses_intensity():
    nlayer = 4
    ng = 3
    nbands = 2
    nu_bands = jnp.array([1000.0, 1005.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau_ckd = jnp.full((nlayer, ng, nbands), 0.1)
    weights_ckd = jnp.array([0.2, 0.3, 0.5])
    art = ArtEmisPure(nlayer=nlayer, nu_grid=nu_bands, nstream=8)

    flux_ckd, u1, u2 = art.run_ckd_with_limb_darkening(
        dtau_ckd, temperature, weights_ckd, nu_bands
    )
    intensity_ckd = art.run_ckd_intensity(
        dtau_ckd, temperature, weights_ckd, nu_bands
    )
    flux_from_intensity = rtrun_emis_pureabs_ibased_flux_from_intensity(
        intensity_ckd, art.mus, art.weights
    )

    assert flux_ckd == pytest.approx(flux_from_intensity)
    assert flux_ckd == pytest.approx(
        art.run_ckd(dtau_ckd, temperature, weights_ckd, nu_bands)
    )
    assert jnp.shape(u1) == jnp.shape(nu_bands)
    assert jnp.shape(u2) == jnp.shape(nu_bands)


def test_artemispure_run_ckd_with_reduced_limb_darkening():
    nlayer = 4
    ng = 3
    nbands = 2
    nu_bands = jnp.array([1000.0, 1005.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau_ckd = jnp.full((nlayer, ng, nbands), 0.1)
    weights_ckd = jnp.array([0.2, 0.3, 0.5])
    art = ArtEmisPure(nlayer=nlayer, nu_grid=nu_bands, nstream=8)

    flux_ckd, u1, u2 = art.run_ckd_with_limb_darkening(
        dtau_ckd, temperature, weights_ckd, nu_bands, reduce_ld=True
    )

    assert jnp.shape(flux_ckd) == jnp.shape(nu_bands)
    assert jnp.ndim(u1) == 0
    assert jnp.ndim(u2) == 0
