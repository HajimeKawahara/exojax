import jax.numpy as jnp
import pytest

from exojax.rt import ArtEmisPure, ArtEmisScat
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased


def test_artemisscat_sfm2st_reduces_to_pure_absorption_ibased():
    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau = jnp.full((nlayer, len(nu_grid)), 0.1)
    single_scattering_albedo = jnp.zeros_like(dtau)
    asymmetric_parameter = jnp.zeros_like(dtau)

    art_pure = ArtEmisPure(nlayer=nlayer, nu_grid=nu_grid, rtsolver="ibased", nstream=8)
    art_scat = ArtEmisScat(
        nlayer=nlayer,
        nu_grid=nu_grid,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )

    expected = art_pure.run(dtau, temperature)
    actual = art_scat.run(
        dtau, single_scattering_albedo, asymmetric_parameter, temperature
    )

    assert actual == pytest.approx(expected)


def test_rtrun_emis_scat_sfm2st_reduces_to_pure_absorption_ibased():
    from exojax.rt.rtransfer import rtrun_emis_scat_sfm2st_toonhm

    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau = jnp.full((nlayer, len(nu_grid)), 0.1)
    single_scattering_albedo = jnp.zeros_like(dtau)
    asymmetric_parameter = jnp.zeros_like(dtau)
    art_pure = ArtEmisPure(nlayer=nlayer, nu_grid=nu_grid, rtsolver="ibased", nstream=8)
    source_matrix = piBarr(temperature, nu_grid)

    expected = rtrun_emis_pureabs_ibased(
        dtau, source_matrix, art_pure.mus, art_pure.weights
    )
    actual = rtrun_emis_scat_sfm2st_toonhm(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        source_matrix,
        art_pure.mus,
        art_pure.weights,
    )

    assert actual == pytest.approx(expected)


def test_rtrun_emis_scat_sfm2st_returns_finite_flux_with_scattering():
    from exojax.rt.rtransfer import rtrun_emis_scat_sfm2st_toonhm

    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau = jnp.full((nlayer, len(nu_grid)), 0.1)
    single_scattering_albedo = jnp.full_like(dtau, 0.5)
    asymmetric_parameter = jnp.full_like(dtau, 0.1)
    art_pure = ArtEmisPure(nlayer=nlayer, nu_grid=nu_grid, rtsolver="ibased", nstream=8)
    source_matrix = piBarr(temperature, nu_grid)

    spectrum = rtrun_emis_scat_sfm2st_toonhm(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        source_matrix,
        art_pure.mus,
        art_pure.weights,
    )

    assert jnp.shape(spectrum) == jnp.shape(nu_grid)
    assert jnp.all(jnp.isfinite(spectrum))
    assert jnp.all(spectrum > 0.0)


def test_artemisscat_sfm2st_ckd_reduces_to_pure_absorption_ibased_ckd():
    nlayer = 4
    ng = 3
    nbands = 2
    nu_bands = jnp.array([1000.0, 1005.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau_ckd = jnp.full((nlayer, ng, nbands), 0.1)
    weights_ckd = jnp.array([0.2, 0.3, 0.5])
    single_scattering_albedo = jnp.zeros((nlayer, nbands))
    asymmetric_parameter = jnp.zeros((nlayer, nbands))

    art_pure = ArtEmisPure(
        nlayer=nlayer, nu_grid=nu_bands, rtsolver="ibased", nstream=8
    )
    art_scat = ArtEmisScat(
        nlayer=nlayer,
        nu_grid=nu_bands,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )

    expected = art_pure.run_ckd(dtau_ckd, temperature, weights_ckd, nu_bands)
    actual = art_scat.run_ckd(
        dtau_ckd,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        weights_ckd,
        nu_bands,
    )

    assert actual == pytest.approx(expected)
