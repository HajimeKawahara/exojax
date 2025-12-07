from jax import config

config.update("jax_enable_x64", True)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def test_hitemp_elower_max():
    import numpy as np
    from exojax.database.hitemp.api import MdbHitemp
    nu_start = 11353.636363636364
    nu_end = 11774.70588235294
    mdb = MdbHitemp("CH4", nurange=[nu_start,nu_end], elower_max=4000.0)
    print(np.max(mdb.elower))
    assert np.max(mdb.elower) < 4000.0