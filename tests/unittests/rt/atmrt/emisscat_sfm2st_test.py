import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.rt import ArtEmisPure, ArtEmisScat, ArtReflectEmis
from exojax.rt.planck import piB, piBarr
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


def test_rtrun_emis_scat_sfm2st_surface_pure_absorption_boundary_term():
    from exojax.rt.rtransfer import (
        initialize_gaussian_quadrature,
        rtrun_emis_scat_sfm2st_toonhm,
        rtrun_emis_scat_sfm2st_toonhm_surface,
    )

    dtau = jnp.array(
        [
            [0.10, 0.20, 0.30],
            [0.05, 0.10, 0.20],
            [0.20, 0.05, 0.10],
        ]
    )
    source_matrix = jnp.ones_like(dtau)
    single_scattering_albedo = jnp.zeros_like(dtau)
    asymmetric_parameter = jnp.zeros_like(dtau)
    source_surface = jnp.array([2.0, 3.0, 4.0])
    mus, weights = initialize_gaussian_quadrature(8)

    without_surface = rtrun_emis_scat_sfm2st_toonhm(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        source_matrix,
        mus,
        weights,
    )
    with_surface = rtrun_emis_scat_sfm2st_toonhm_surface(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        source_matrix,
        source_surface,
        mus,
        weights,
    )

    tau_bottom = jnp.sum(dtau, axis=0)
    surface_transmission = jnp.sum(
        2.0
        * mus[:, None]
        * weights[:, None]
        * jnp.exp(-tau_bottom[None, :] / mus[:, None]),
        axis=0,
    )
    expected_boundary_term = source_surface * surface_transmission
    np.testing.assert_allclose(
        with_surface - without_surface,
        expected_boundary_term,
        rtol=1.0e-6,
    )


def test_artemisscat_sfm2st_run_with_surface_matches_existing_boundary_paths():
    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau = jnp.full((nlayer, len(nu_grid)), 0.1)
    single_scattering_albedo = jnp.full_like(dtau, 0.5)
    asymmetric_parameter = jnp.full_like(dtau, 0.1)
    art = ArtEmisScat(
        nlayer=nlayer,
        nu_grid=nu_grid,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )
    art_reflect_emis = ArtReflectEmis(
        nlayer=nlayer,
        nu_grid=nu_grid,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )

    expected = art.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
    )
    zero_surface = art.run_with_surface(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        jnp.zeros_like(nu_grid),
    )
    source_surface = piB(1300.0, nu_grid)
    with_surface = art.run_with_surface(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        source_surface,
    )
    expected_with_surface = art_reflect_emis.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        source_surface,
        jnp.zeros_like(nu_grid),
        jnp.zeros_like(nu_grid),
    )

    np.testing.assert_array_equal(zero_surface, expected)
    np.testing.assert_array_equal(with_surface, expected_with_surface)


def test_artemisscat_sfm2st_run_with_surface_transparent_limit_and_grad():
    nlayer = 2
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.array([800.0, 1000.0])
    dtau = jnp.zeros((nlayer, len(nu_grid)))
    single_scattering_albedo = jnp.full_like(dtau, 0.5)
    asymmetric_parameter = jnp.full_like(dtau, 0.1)
    source_surface = jnp.array([2.0, 3.0, 4.0])
    art = ArtEmisScat(
        nlayer=nlayer,
        nu_grid=nu_grid,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )

    def summed_flux(surface_scale):
        spectrum = art.run_with_surface(
            dtau,
            single_scattering_albedo,
            asymmetric_parameter,
            temperature,
            surface_scale * source_surface,
        )
        return jnp.sum(spectrum)

    actual = jax.jit(art.run_with_surface)(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        source_surface,
    )
    derivative = jax.grad(summed_flux)(1.0)

    np.testing.assert_allclose(actual, source_surface, rtol=1.0e-6)
    np.testing.assert_allclose(derivative, jnp.sum(source_surface), rtol=1.0e-6)


def test_artemisscat_run_with_surface_rejects_non_sfm_solver():
    art = ArtEmisScat(
        nlayer=2,
        nu_grid=jnp.array([1000.0]),
        rtsolver="fluxadding_toon_hemispheric_mean",
    )
    layer = jnp.ones((2, 1))

    with pytest.raises(ValueError, match="currently supports"):
        art.run_with_surface(
            layer,
            layer,
            layer,
            jnp.array([900.0, 1000.0]),
            jnp.ones(1),
        )


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
