""" unit test for ArtEmisPure with OpaPremodit

Note:
    The original file was from integration/unittests_long/premodit/premodit_spectrum_test.py

"""

import pytest
import pkg_resources
from jax import config
import pandas as pd
import numpy as np
from exojax.test.emulate_mdb import mock_mdb
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
from exojax.spec.opacalc import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.atmrt import ArtEmisPure

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "db, diffmode",
    [
        ("exomol", 0),
        ("hitemp", 0),
    ],
)
def test_rt_for_single_broadening_parameters(db, diffmode, fig=False):

    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid, rtsolver="fbased2st", nstream=2
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_mmr_profile(0.1)
    gravity = 2478.57
    # gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    # mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    # mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        diffmode=diffmode,
        auto_trange=[art.Tlow, art.Thigh],
        broadening_resolution={"mode": "single", "value": None},
    )
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity)
    F0 = art.run(dtau, Tarr)

    if db == "hitemp":
        filename = pkg_resources.resource_filename(
            "exojax", "data/testdata/" + TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
        )
    elif db == "exomol":
        filename = pkg_resources.resource_filename(
            "exojax", "data/testdata/" + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
        )

    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(F0 / dat["flux"].values - 1.0)
    print(np.max(residual))
    # assert np.all(residual < 0.05)
    return nu_grid, F0, dat["flux"].values


if __name__ == "__main__":
    nu, F, Fref = test_rt_for_single_broadening_parameters("exomol", 0)
    import matplotlib.pyplot as plt
    plt.plot(nu, F, label="F")
    plt.plot(nu, Fref, label="Fref")
    plt.show()
    
    test_rt_for_single_broadening_parameters("hitemp", 0)
