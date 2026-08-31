import jax.numpy as jnp
import numpy as np

from exojax.rt import ArtEmisScat, ArtReflectEmis, ArtReflectPure
from exojax.rt.rtransfer import rtrun_reflect_sfm2st_toonhm


def _reflection_inputs(nlayer=4, nnus=3):
    dtau = jnp.full((nlayer, nnus), 0.1)
    single_scattering_albedo = jnp.full_like(dtau, 0.5)
    asymmetric_parameter = jnp.full_like(dtau, 0.1)
    reflectivity_surface = jnp.zeros(nnus)
    incoming_flux = jnp.ones(nnus)
    return (
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        reflectivity_surface,
        incoming_flux,
    )


def test_reflect_sfm2st_transparent_lambert_surface():
    nlayer = 2
    nnus = 3
    art = ArtReflectPure(
        nlayer=nlayer,
        nu_grid=jnp.arange(nnus) + 1000.0,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )
    dtau = jnp.zeros((nlayer, nnus))
    zeros = jnp.zeros_like(dtau)
    reflectivity_surface = jnp.array([0.2, 0.4, 0.6])
    incoming_flux = jnp.array([1.0, 2.0, 3.0])

    actual = art.run(
        dtau,
        zeros,
        zeros,
        reflectivity_surface,
        incoming_flux,
    )

    np.testing.assert_allclose(
        actual, reflectivity_surface * incoming_flux, rtol=1.0e-6
    )


def test_reflect_sfm2st_black_atmosphere_and_surface_returns_zero():
    nlayer = 4
    nnus = 3
    art = ArtReflectPure(
        nlayer=nlayer,
        nu_grid=jnp.arange(nnus) + 1000.0,
        rtsolver="sfm2st_toon_hemispheric_mean",
    )
    dtau = jnp.full((nlayer, nnus), 0.2)
    zeros = jnp.zeros_like(dtau)

    actual = art.run(
        dtau,
        zeros,
        zeros,
        jnp.zeros(nnus),
        jnp.ones(nnus),
    )

    np.testing.assert_array_equal(actual, jnp.zeros(nnus))


def test_reflect_sfm2st_absorbing_atmosphere_attenuates_surface_reflection():
    nlayer = 2
    nnus = 3
    art = ArtReflectPure(
        nlayer=nlayer,
        nu_grid=jnp.arange(nnus) + 1000.0,
        rtsolver="sfm2st_toon_hemispheric_mean",
    )
    dtau = jnp.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]])
    zeros = jnp.zeros_like(dtau)
    reflectivity_surface = jnp.array([0.2, 0.4, 0.6])
    incoming_flux = jnp.array([1.0, 2.0, 3.0])

    actual = art.run(
        dtau,
        zeros,
        zeros,
        reflectivity_surface,
        incoming_flux,
    )
    tau_bottom = jnp.sum(dtau, axis=0)
    downward_surface_flux = (
        reflectivity_surface * incoming_flux * jnp.exp(-2.0 * tau_bottom)
    )
    upward_transmission = jnp.sum(
        2.0
        * art.mus[:, None]
        * art.weights[:, None]
        * jnp.exp(-tau_bottom[None, :] / art.mus[:, None]),
        axis=0,
    )

    np.testing.assert_allclose(
        actual, downward_surface_flux * upward_transmission, rtol=1.0e-6
    )


def test_reflect_sfm2st_returns_finite_nonnegative_flux():
    inputs = _reflection_inputs()
    art = ArtReflectPure(
        nlayer=inputs[0].shape[0],
        nu_grid=jnp.arange(inputs[0].shape[1]) + 1000.0,
        rtsolver="sfm2st_toon_hemispheric_mean",
    )

    spectrum = art.run(*inputs)

    assert jnp.all(jnp.isfinite(spectrum))
    assert jnp.all(spectrum >= 0.0)
    assert jnp.all(spectrum <= inputs[-1])


def test_reflect_sfm2st_conservative_scattering_is_energy_bounded():
    nlayer = 100
    nnus = 2
    art = ArtReflectPure(
        nlayer=nlayer,
        nu_grid=jnp.arange(nnus) + 1000.0,
        rtsolver="sfm2st_toon_hemispheric_mean",
    )
    dtau = jnp.full((nlayer, nnus), 0.1)
    single_scattering_albedo = jnp.ones_like(dtau)
    asymmetric_parameter = jnp.zeros_like(dtau)
    incoming_flux = jnp.ones(nnus)

    spectrum = art.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        jnp.zeros(nnus),
        incoming_flux,
    )

    assert jnp.all(jnp.isfinite(spectrum))
    assert jnp.all(spectrum >= 0.0)
    assert jnp.all(spectrum <= incoming_flux)


def test_artreflectpure_sfm2st_matches_low_level_solver():
    inputs = _reflection_inputs()
    dtau, single_scattering_albedo, asymmetric_parameter, _, _ = inputs
    art = ArtReflectPure(
        nlayer=dtau.shape[0],
        nu_grid=jnp.arange(dtau.shape[1]) + 1000.0,
        rtsolver="sfm2st_toon_hemispheric_mean",
    )

    actual = art.run(*inputs)
    expected = rtrun_reflect_sfm2st_toonhm(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        jnp.zeros_like(dtau),
        jnp.zeros(dtau.shape[1]),
        inputs[-2],
        inputs[-1],
        art.mus,
        art.weights,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)


def test_artreflectemis_sfm2st_reduces_to_emission_sfm2st():
    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau = jnp.full((nlayer, len(nu_grid)), 0.1)
    single_scattering_albedo = jnp.full_like(dtau, 0.5)
    asymmetric_parameter = jnp.full_like(dtau, 0.1)
    art_emis = ArtEmisScat(
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

    expected = art_emis.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
    )
    actual = art_reflect_emis.run(
        dtau,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        jnp.zeros(len(nu_grid)),
        jnp.zeros(len(nu_grid)),
        jnp.zeros(len(nu_grid)),
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)


def test_artreflectpure_sfm2st_ckd_transparent_lambert_surface():
    nlayer = 2
    ng = 3
    nbands = 2
    art = ArtReflectPure(
        nlayer=nlayer,
        nu_grid=jnp.arange(nbands) + 1000.0,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )
    dtau_ckd = jnp.zeros((nlayer, ng, nbands))
    zeros = jnp.zeros((nlayer, nbands))
    reflectivity_surface = jnp.array([0.25, 0.5])
    incoming_flux = jnp.array([2.0, 3.0])
    weights = jnp.array([0.2, 0.3, 0.5])

    actual = art.run_ckd(
        dtau_ckd,
        zeros,
        zeros,
        reflectivity_surface,
        incoming_flux,
        weights,
    )

    np.testing.assert_allclose(
        actual, reflectivity_surface * incoming_flux, rtol=1.0e-6
    )


def test_artreflectemis_sfm2st_ckd_reduces_to_emission_sfm2st():
    nlayer = 4
    ng = 3
    nbands = 2
    nu_bands = jnp.array([1000.0, 1005.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau_ckd = jnp.full((nlayer, ng, nbands), 0.1)
    weights = jnp.array([0.2, 0.3, 0.5])
    single_scattering_albedo = jnp.full((nlayer, nbands), 0.5)
    asymmetric_parameter = jnp.full((nlayer, nbands), 0.1)
    art_emis = ArtEmisScat(
        nlayer=nlayer,
        nu_grid=nu_bands,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )
    art_reflect_emis = ArtReflectEmis(
        nlayer=nlayer,
        nu_grid=nu_bands,
        rtsolver="sfm2st_toon_hemispheric_mean",
        nstream=8,
    )

    expected = art_emis.run_ckd(
        dtau_ckd,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        weights,
        nu_bands,
    )
    actual = art_reflect_emis.run_ckd(
        dtau_ckd,
        single_scattering_albedo,
        asymmetric_parameter,
        temperature,
        jnp.zeros(nbands),
        jnp.zeros(nbands),
        jnp.zeros(nbands),
        weights,
        nu_bands,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)
