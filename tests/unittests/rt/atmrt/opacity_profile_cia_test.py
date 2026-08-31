"""Tests for CIA optical-depth profile broadcasting."""

import jax.numpy as jnp
import numpy as np

from exojax.rt import ArtTransPure


def test_opacity_profile_cia_accepts_one_dimensional_profiles():
    """CIA opacity should accept 1D mean molecular weight and gravity profiles."""
    nlayer = 4
    nnu = 6
    art = ArtTransPure(
        pressure_top=1.0e-5,
        pressure_btm=1.0,
        nlayer=nlayer,
        warn_no_nu_grid=False,
    )
    logacia_matrix = jnp.full((nlayer, nnu), -46.0)
    temperature = jnp.linspace(800.0, 1100.0, nlayer)
    vmr_h2 = jnp.full(nlayer, 0.8)
    vmr_he = jnp.full(nlayer, 0.2)
    mmw = jnp.linspace(2.2, 2.4, nlayer)
    gravity = jnp.linspace(900.0, 1200.0, nlayer)

    dtau_1d = art.opacity_profile_cia(
        logacia_matrix, temperature, vmr_h2, vmr_he, mmw, gravity
    )
    dtau_2d = art.opacity_profile_cia(
        logacia_matrix, temperature, vmr_h2, vmr_he, mmw[:, None], gravity[:, None]
    )

    assert dtau_1d.shape == (nlayer, nnu)
    np.testing.assert_allclose(np.asarray(dtau_1d), np.asarray(dtau_2d))


def test_opacity_profile_xs_accepts_one_dimensional_profiles():
    """Cross-section opacity should accept 1D mass and gravity profiles."""
    nlayer = 4
    nnu = 6
    art = ArtTransPure(
        pressure_top=1.0e-5,
        pressure_btm=1.0,
        nlayer=nlayer,
        warn_no_nu_grid=False,
    )
    xsmatrix = jnp.full((nlayer, nnu), 1.0e-24)
    mixing_ratio = jnp.linspace(1.0e-5, 4.0e-5, nlayer)
    mass = jnp.linspace(2.2, 2.4, nlayer)
    gravity = jnp.linspace(900.0, 1200.0, nlayer)

    dtau_1d = art.opacity_profile_xs(xsmatrix, mixing_ratio, mass, gravity)
    dtau_2d = art.opacity_profile_xs(
        xsmatrix, mixing_ratio, mass[:, None], gravity[:, None]
    )

    assert dtau_1d.shape == (nlayer, nnu)
    np.testing.assert_allclose(np.asarray(dtau_1d), np.asarray(dtau_2d))
