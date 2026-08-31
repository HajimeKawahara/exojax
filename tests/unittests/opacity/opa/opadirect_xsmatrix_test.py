from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from exojax.opacity import OpaDirect


def test_opadirect_atomic_xsmatrix_with_different_line_and_layer_counts():
    nline = 3
    mdb = SimpleNamespace(
        dbtype="vald",
        nu_lines=np.array([1000.0, 1001.0, 1002.0]),
        T_gQT=jnp.array([300.0, 3000.0]),
        gQT_284species=jnp.array([[1.0, 2.0], [2.0, 4.0]]),
        QTmask=jnp.array([0, 1, 0]),
        QTref_284=jnp.array([1.0, 2.0]),
        vmrH=0.0,
        vmrHe=0.16,
        vmrHH=0.84,
        ielem=jnp.full(nline, 26),
        iion=jnp.ones(nline),
        dev_nu_lines=jnp.array([1000.0, 1001.0, 1002.0]),
        elower=jnp.array([0.0, 10.0, 20.0]),
        eupper=jnp.array([1000.0, 1011.0, 1022.0]),
        atomicmass=jnp.full(nline, 55.845),
        ionE=jnp.full(nline, 7.9),
        gamRad=jnp.full(nline, 8.0),
        gamSta=jnp.zeros(nline),
        vdWdamp=jnp.full(nline, -7.0),
        logsij0=jnp.log(jnp.array([1.0e-20, 2.0e-20, 3.0e-20])),
    )
    nu_grid = np.linspace(999.0, 1003.0, 5)
    opa = OpaDirect(mdb, nu_grid=nu_grid)

    xsmatrix = opa.xsmatrix(
        jnp.array([1000.0, 1500.0]), jnp.array([0.1, 0.01])
    )

    assert xsmatrix.shape == (2, len(nu_grid))
    assert jnp.all(jnp.isfinite(xsmatrix))
