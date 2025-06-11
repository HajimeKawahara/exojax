import pytest
import jax.numpy as jnp
import numpy as np
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.database.hitran  import normalized_doppler_sigma
from exojax.opacity.ditkernel import fold_voigt_kernel_logst
from jax import config


def test_fold_voigt_kernel_logst():

    config.update("jax_enable_x64", True)
    nu_grid, wav, resolution = mock_wavenumber_grid()
    Ttest = 1200.0
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0
    #### molmass and ngammaL_grid are based on the following code ###
    # mdb = mock_mdbExomol()
    # gammaL = gamma_exomol(P, Ttest, mdb.n_Texp, mdb.alpha_ref)
    # dv_lines = mdb.nu_lines / resolution
    # ngammaL = gammaL / dv_lines
    # ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.2)
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
    return vk


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    vk = test_fold_voigt_kernel_logst()
    plt.plot(vk[:, 0], label="0", alpha=0.3)
    plt.plot(vk[:, 1], label="1", alpha=0.2)
    plt.plot(vk[:, 2], label="2", alpha=0.1)
    plt.yscale("log")
    plt.legend()
    plt.savefig("fold_voigt_kernel_logst.png")
