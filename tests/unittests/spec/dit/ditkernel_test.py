import pytest
import jax.numpy as jnp
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.set_ditgrid import ditgrid_log_interval
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import normalized_doppler_sigma
from exojax.spec.ditkernel import fold_voigt_kernel_logst
from exojax.spec.initspec import init_modit
from jax import config


def test_fold_voigt_kernel_logst(fig=False):
    
    config.update("jax_enable_x64", True)
    mdb = mock_mdbExomol()
    nu_grid, wav, res = mock_wavenumber_grid()
    Ttest = 1200.0
    P = 1.0
    # MODIT manual
    cont, index, R, pmarray = init_modit(mdb.nu_lines, nu_grid)
    nsigmaD = normalized_doppler_sigma(Ttest, mdb.molmass, R)
    # log ngamma grid construction
    gammaL = gamma_exomol(P, Ttest, mdb.n_Texp, mdb.alpha_ref)
    dv_lines = mdb.nu_lines / R
    ngammaL = gammaL / dv_lines
    ngammaL_grid = ditgrid_log_interval(ngammaL, dit_grid_resolution=0.2)
    log_ngammaL_grid = jnp.log(ngammaL_grid)
    _, _, R, pmarray = init_modit(mdb.nu_lines, nu_grid)
    Ng_nu = len(nu_grid)
    vk = fold_voigt_kernel_logst(
        jnp.fft.rfftfreq(2 * Ng_nu, 1),
        jnp.log(nsigmaD),
        log_ngammaL_grid,
        Ng_nu,
        pmarray,
    )
    ref = [362.08269655,326.81894669,294.27354816] # feb 5th 2025

    if fig:
        import matplotlib.pyplot as plt
        plt.plot(vk[:,0],label="0")
        plt.plot(vk[:,1],label="1")
        plt.plot(vk[:,2],label="2")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.savefig("fold_voigt_kernel_logst.png")

    assert vk.shape == (len(nu_grid)+1, len(log_ngammaL_grid))
    assert jnp.all(jnp.sum(vk, axis=0) == pytest.approx(ref))

if __name__ == "__main__":
    test_fold_voigt_kernel_logst(fig=True)