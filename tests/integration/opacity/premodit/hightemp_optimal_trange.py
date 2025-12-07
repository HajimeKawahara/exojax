from jax import config
import pytest
config.update("jax_enable_x64", True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def test_high_temp_trange():
    import numpy as np
    from exojax.database.hitemp.api import MdbHitemp
    nu_start = 11353.636363636364
    nu_end = 11774.70588235294
    mdb = MdbHitemp("CH4", nurange=[nu_start,nu_end], elower_max=4000.0)
    from exojax.opacity import OpaPremodit
    from exojax.utils.grids import wavenumber_grid
    N = 10000
    nus, wav, res = wavenumber_grid(nu_start, nu_end, N, xsmode="premodit") 

    #version = 2 works for 80K
    opa = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True, auto_trange=[1000.0, 6500.0], diffmode=0, version_auto_trange=2)

    #version = 1 does not work for 80K
    with pytest.raises(ValueError):
        opa = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True, auto_trange=[1000.0, 6500.0], diffmode=0, version_auto_trange=1)


if __name__ == "__main__":
    test_high_temp_trange()