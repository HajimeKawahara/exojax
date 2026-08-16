import pytest
from exojax.rt import ArtAbsPure
from exojax.utils.grids import wavenumber_grid
import jax.numpy as jnp


def _toy_artabs_case():
    art = ArtAbsPure(
        pressure_top=1.0e-3,
        pressure_btm=1.0e1,
        nlayer=5,
        nu_grid=jnp.array([1.0]),
    )
    dtau = jnp.array([[1.0], [2.0], [4.0], [8.0], [16.0]])
    incoming_flux = jnp.ones(1)
    return art, dtau, incoming_flux


def test_artabs_run_at_toa():
    """Test the run method of ArtAbsPure at TOA
    Note:
        why the answer is exp(-3.5*2)? This test assumes the d(log10 P) = 1
        and pressure = [-3,-2,-1,0,1] this is (log) center of the layers.
        We integrate the dtau over d log10 P = -3.5 to 0 with dtau = 1 for each layer.
        3 layers + 0.5 layer = 3.5 layers. we get tau = 3.5,
        but the observer is lcocated at the top of the atmosphere.
        then we need to multiply 2.

    """
    nu_grid, wav, res = wavenumber_grid(22990.0, 23000.0, 2, xsmode="premodit")
    art = ArtAbsPure(pressure_top=1.0e-3, pressure_btm=1.0e1, nlayer=5, nu_grid=nu_grid)
    dtau = jnp.ones((art.nlayer, len(nu_grid)))
    incf = jnp.ones_like(nu_grid)
    ps = 1.0e0  # bar
    f = art.run(dtau, pressure_surface=ps, incoming_flux=incf, mu_in=1.0, mu_out=1.0)

    assert f[0] == pytest.approx(jnp.exp(-3.5 * 2))


def test_artabs_run_at_ground():
    nu_grid, wav, res = wavenumber_grid(22990.0, 23000.0, 2, xsmode="premodit")
    art = ArtAbsPure(pressure_top=1.0e-3, pressure_btm=1.0e1, nlayer=5, nu_grid=nu_grid)
    print(art.pressure)
    dtau = jnp.ones((art.nlayer, len(nu_grid)))
    incf = jnp.ones_like(nu_grid)
    deltalogp = 0.3
    ps = 10 ** (deltalogp)  # bar
    f = art.run(dtau, pressure_surface=ps, incoming_flux=incf, mu_in=1.0, mu_out=None)

    assert f[0] == pytest.approx(jnp.exp(-(3.5 + deltalogp)))


@pytest.mark.parametrize(
    ("pressure_surface", "expected_tau"),
    [
        (1.0, 11.0),
        (10**-0.5, 7.0),
        (1.0e-3, 0.5),
    ],
)
def test_artabs_run_partial_layer(pressure_surface, expected_tau):
    art, dtau, incoming_flux = _toy_artabs_case()

    spectrum = art.run(
        dtau,
        pressure_surface=pressure_surface,
        incoming_flux=incoming_flux,
        mu_in=1.0,
        mu_out=None,
    )

    assert spectrum[0] == pytest.approx(jnp.exp(-expected_tau))


def test_artabs_run_ckd_partial_layer():
    art, dtau, incoming_flux = _toy_artabs_case()
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

    expected = 0.25 * jnp.exp(-11.0) + 0.75 * jnp.exp(-22.0)
    assert spectrum[0] == pytest.approx(expected)


if __name__ == "__main__":
    test_artabs_run_at_toa()
    test_artabs_run_at_ground()
