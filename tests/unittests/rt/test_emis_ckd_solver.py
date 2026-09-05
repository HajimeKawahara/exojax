import jax.numpy as jnp
import pytest

from exojax.rt import ArtEmisPure


@pytest.mark.parametrize(
    "rtsolver,nstream", [("ibased", 8), ("fbased2st", 2), ("ibased_linsap", 8)]
)
@pytest.mark.parametrize("ng", [1, 3])
def test_ckd_matches_weighted_monochromatic_solver(rtsolver, nstream, ng):
    nu_bands = jnp.array([1000.0, 1500.0])
    ntemperature = 3 if rtsolver == "ibased_linsap" else 2
    temperature = jnp.linspace(500.0, 1000.0, ntemperature)
    art = ArtEmisPure(nlayer=2, nu_grid=nu_bands, rtsolver=rtsolver, nstream=nstream)
    dtau = jnp.array([[0.1, 0.3], [1.0, 3.0]])
    dtau_ckd = dtau[:, None, :] * jnp.arange(1, ng + 1)[None, :, None]
    weights = jnp.arange(1, ng + 1, dtype=float)
    weights = weights / weights.sum()
    expected = sum(
        weights[g] * art.run(dtau_ckd[:, g, :], temperature) for g in range(ng)
    )

    actual = art.run_ckd(dtau_ckd, temperature, weights, nu_bands)

    assert actual == pytest.approx(expected, rel=1.0e-6)
