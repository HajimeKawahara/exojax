"""Transmission CKD post-processing smoke tests."""

import jax.numpy as jnp
import numpy as np

from exojax.postproc.ckd import sample_ckd_bands_at_wavelengths
from exojax.rt import ArtTransPure


def test_transpure_ckd_can_be_sampled_at_observed_wavelengths():
    """CKD transmission output should be finite after wavelength sampling."""
    nlayer = 8
    ng = 3
    nband = 5
    art = ArtTransPure(
        pressure_top=1.0e-6,
        pressure_btm=1.0,
        nlayer=nlayer,
        warn_no_nu_grid=False,
    )

    temperature = jnp.linspace(900.0, 1300.0, nlayer)
    mean_molecular_weight = jnp.linspace(2.30, 2.35, nlayer)
    radius_btm = 7.0e9
    gravity_btm = 1200.0
    weights = jnp.array([0.25, 0.5, 0.25])

    layer_scale = jnp.linspace(0.1, 1.0, nlayer)[:, None, None]
    g_scale = jnp.array([0.5, 1.0, 2.0])[None, :, None]
    band_scale = jnp.linspace(0.7, 1.3, nband)[None, None, :]
    dtau_ckd = 0.01 * layer_scale * g_scale * band_scale

    nu_bands = jnp.array([2000.0, 2500.0, 3000.0, 3500.0, 4000.0])
    wavelength_nm = jnp.array([5000.0, 3333.3333333333, 2500.0])

    rp2_bands = art.run_ckd(
        dtau_ckd,
        temperature,
        mean_molecular_weight,
        radius_btm,
        gravity_btm,
        weights,
    )
    rp2_sampled = sample_ckd_bands_at_wavelengths(
        nu_bands, rp2_bands, wavelength_nm, unit="nm"
    )
    rp_over_rs = jnp.sqrt(rp2_sampled) * (radius_btm / 6.8e10)

    assert rp2_bands.shape == (nband,)
    assert rp2_sampled.shape == wavelength_nm.shape
    assert rp_over_rs.shape == wavelength_nm.shape
    assert np.all(np.isfinite(np.asarray(rp_over_rs)))
    assert np.all(np.asarray(rp_over_rs) > 0.0)
