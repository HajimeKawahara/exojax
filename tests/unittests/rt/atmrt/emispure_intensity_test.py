import jax.numpy as jnp
import pytest

from exojax.rt import ArtEmisPure
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import (
    rtrun_emis_pureabs_ibased,
    rtrun_emis_pureabs_ibased_flux_from_intensity,
    rtrun_emis_pureabs_ibased_intensity,
)


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
