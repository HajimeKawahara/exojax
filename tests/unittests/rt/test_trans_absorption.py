import pytest
from exojax.rt import ArtAbsPure
import jax.numpy as jnp


def _constant_opacity_artabs_case():
    art = ArtAbsPure(
        pressure_top=1.0e-3,
        pressure_btm=1.0e1,
        nlayer=5,
        nu_grid=jnp.array([1.0]),
    )
    dtau = jnp.asarray(art.dParr)[:, jnp.newaxis]
    incoming_flux = jnp.ones(1)
    return art, dtau, incoming_flux


def test_artabs_run_at_toa():
    """Test constant-opacity attenuation for an observer at the TOA."""
    art, dtau, incoming_flux = _constant_opacity_artabs_case()
    ps = 1.0e0  # bar
    spectrum = art.run(
        dtau,
        pressure_surface=ps,
        incoming_flux=incoming_flux,
        mu_in=1.0,
        mu_out=1.0,
    )
    expected_tau = ps - art.pressure_boundary[0]

    assert spectrum[0] == pytest.approx(jnp.exp(-2.0 * expected_tau))


def test_artabs_run_at_ground():
    art, dtau, incoming_flux = _constant_opacity_artabs_case()
    deltalogp = 0.3
    ps = 10 ** (deltalogp)  # bar
    spectrum = art.run(
        dtau,
        pressure_surface=ps,
        incoming_flux=incoming_flux,
        mu_in=1.0,
        mu_out=None,
    )
    expected_tau = ps - art.pressure_boundary[0]

    assert spectrum[0] == pytest.approx(jnp.exp(-expected_tau))


@pytest.mark.parametrize(
    "pressure_surface",
    [
        1.0,
        10**-0.5,
        1.0e-3,
    ],
)
def test_artabs_run_partial_layer(pressure_surface):
    art, dtau, incoming_flux = _constant_opacity_artabs_case()

    spectrum = art.run(
        dtau,
        pressure_surface=pressure_surface,
        incoming_flux=incoming_flux,
        mu_in=1.0,
        mu_out=None,
    )
    expected_tau = pressure_surface - art.pressure_boundary[0]

    assert spectrum[0] == pytest.approx(jnp.exp(-expected_tau))


def test_artabs_run_ckd_partial_layer():
    art, dtau, incoming_flux = _constant_opacity_artabs_case()
    dtau_ckd = jnp.stack([dtau, 2.0 * dtau], axis=1)
    weights = jnp.array([0.25, 0.75])

    spectrum = art.run_ckd(
        dtau_ckd,
        pressure_surface=1.0,
        incoming_flux=incoming_flux,
        mu_in=1.0,
        mu_out=None,
        weights=weights,
    )

    expected_tau = 1.0 - art.pressure_boundary[0]
    expected = 0.25 * jnp.exp(-expected_tau) + 0.75 * jnp.exp(-2.0 * expected_tau)
    assert spectrum[0] == pytest.approx(expected)
