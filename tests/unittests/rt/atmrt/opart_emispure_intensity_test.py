import jax.numpy as jnp
import pytest

from exojax.rt import ArtEmisPure, OpartEmisPure


class ConstantOpaLayer:
    def __init__(self, nu_grid, dtau_layer):
        self.nu_grid = nu_grid
        self.dtau_layer = dtau_layer

    def __call__(self, params):
        return self.dtau_layer


def test_opart_run_intensity_matches_artemispure():
    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau_layer = jnp.array([0.1, 0.2, 0.3])
    dtau = jnp.tile(dtau_layer, (nlayer, 1))

    art = ArtEmisPure(nlayer=nlayer, nu_grid=nu_grid, nstream=8)
    opart = OpartEmisPure(
        ConstantOpaLayer(nu_grid, dtau_layer),
        nlayer=nlayer,
        nstream=8,
    )

    def layer_update_function(carry_tauintensity, params):
        carry_tauintensity = opart.update_layer_intensity(carry_tauintensity, params)
        return carry_tauintensity, None

    layer_params = [temperature]
    intensity = opart.run_intensity(layer_params, layer_update_function)
    expected = art.run_intensity(dtau, temperature)

    assert intensity == pytest.approx(expected)


def test_opart_run_with_limb_darkening_matches_artemispure():
    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau_layer = jnp.array([0.1, 0.2, 0.3])
    dtau = jnp.tile(dtau_layer, (nlayer, 1))

    art = ArtEmisPure(nlayer=nlayer, nu_grid=nu_grid, nstream=8)
    opart = OpartEmisPure(
        ConstantOpaLayer(nu_grid, dtau_layer),
        nlayer=nlayer,
        nstream=8,
    )

    def layer_update_function(carry_tauintensity, params):
        carry_tauintensity = opart.update_layer_intensity(carry_tauintensity, params)
        return carry_tauintensity, None

    layer_params = [temperature]
    flux, u1, u2 = opart.run_with_limb_darkening(
        layer_params, layer_update_function
    )
    expected_flux, expected_u1, expected_u2 = art.run_with_limb_darkening(
        dtau, temperature
    )

    assert flux == pytest.approx(expected_flux)
    assert u1 == pytest.approx(expected_u1, rel=1.0e-5, abs=1.0e-6)
    # The float32 least-squares fit amplifies sub-ppm intensity differences.
    assert u2 == pytest.approx(expected_u2, rel=2.0e-5, abs=1.0e-6)


def test_opart_run_with_reduced_limb_darkening():
    nlayer = 4
    nu_grid = jnp.array([1000.0, 1001.0, 1002.0])
    temperature = jnp.linspace(800.0, 1200.0, nlayer)
    dtau_layer = jnp.array([0.1, 0.2, 0.3])

    opart = OpartEmisPure(
        ConstantOpaLayer(nu_grid, dtau_layer),
        nlayer=nlayer,
        nstream=8,
    )

    def layer_update_function(carry_tauintensity, params):
        carry_tauintensity = opart.update_layer_intensity(carry_tauintensity, params)
        return carry_tauintensity, None

    layer_params = [temperature]
    flux, u1, u2 = opart.run_with_limb_darkening(
        layer_params, layer_update_function, reduce_ld=True
    )

    assert jnp.shape(flux) == jnp.shape(nu_grid)
    assert jnp.ndim(u1) == 0
    assert jnp.ndim(u2) == 0
