import pytest
import jax.numpy as jnp
import numpy as np
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.database.core.broadening import normalized_doppler_sigma
from exojax.opacity._common.ditkernel import fold_voigt_kernel_logst
from jax import config


def test_fold_voigt_kernel_logst():

    config.update("jax_enable_x64", True)
    nu_grid, wav, resolution = mock_wavenumber_grid()
    Ttest = 1200.0
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0
    molmass = 28.0101  # mdb.molmass
    nsigmaD = normalized_doppler_sigma(Ttest, molmass, resolution)
    ngammaL_grid = jnp.array([16.23074604, 18.25429762, 20.53013342])
    log_ngammaL_grid = jnp.log(ngammaL_grid)
    Ng_nu = len(nu_grid)

    vk = fold_voigt_kernel_logst(
        jnp.fft.rfftfreq(2 * Ng_nu, 1),
        nsigmaD,
        log_ngammaL_grid,
        Ng_nu,
        pmarray,
    )

    ref = [362.08269655, 326.81894669, 294.27354816]  # feb 5th 2025
    assert vk.shape == (len(nu_grid) + 1, len(log_ngammaL_grid))
    assert jnp.all(jnp.sum(vk, axis=0) == pytest.approx(ref))
