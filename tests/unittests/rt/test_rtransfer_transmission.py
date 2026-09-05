import jax.numpy as jnp

from exojax.rt.rtransfer import rtrun_trans_pureabs_trapezoid


def test_opaque_transmission_radius_is_bounded_by_top_layer():
    Nlayer = 5
    Nnu = 2
    dtau_chord = jnp.ones((Nlayer, Nnu)) * jnp.inf
    radius_lower = jnp.array([1.4, 1.3, 1.2, 1.1, 1.0])
    radius_top = 1.5
    Rp2 = rtrun_trans_pureabs_trapezoid(dtau_chord, radius_lower, radius_top)
    assert Rp2[0] > radius_lower[0]**2
    assert Rp2[0] < radius_top**2
