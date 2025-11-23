"""unit test for ArtEmisPure with OpaPremodit

Note:
    The original file was from integration/unittests_long/premodit/premodit_spectrum_test.py
    These tests takes relatively long time to run. So, one of the database is tested at a time (when pytest).

"""

import pytest
from importlib.resources import files
from jax import config
import pandas as pd
import numpy as np
from exojax.test.emulate_mdb import mock_mdb
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtEmisPure

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "db, diffmode",
    [
        ("hitemp", 0),
    ],
)
def test_rt_for_single_broadening_parameters(db, diffmode, fig=False):
    """compares PreMODIT+single broadening with MODIT, so difference is should be small but not very small, 0.0322

    Args:
        db: exomol or hitemp
        diffmode: 0, 1, or 2
        fig: True or False

    Returns:
        nu_grid, F0, reference F0

    """

    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.1)
    gravity = 2478.57

    mdb = mock_mdb(db)
    opa = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        diffmode=diffmode,
        auto_trange=[art.Tlow, art.Thigh],
        broadening_resolution={"mode": "single", "value": None},
    )
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, mdb.molmass, gravity)
    F0 = art.run(dtau, Tarr)

    if db == "hitemp":
        filename = files("exojax").joinpath(
            "data/testdata/" + TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
        )
    elif db == "exomol":
        filename = files("exojax").joinpath(
            "data/testdata/" + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
        )

    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(F0 / dat["flux"].values - 1.0)
    assert np.all(residual < 0.033)


@pytest.mark.parametrize("db, diffmode", [("exomol", 1)])
def test_consistency_ArtEmisPure(db, diffmode, fig=False):
    """compares PreMODIT with MODIT, so difference is very small, 0.005

    Args:
        db: exomol or hitemp
        diffmode: 0, 1, or 2
        fig: True or False

    Returns:
        nu_grid, F0, reference F0

    """
    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtEmisPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100, nu_grid=nu_grid
    )
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_profile(0.1)
    gravity = 2478.57

    mdb = mock_mdb(db)
    opa = OpaPremodit(
        mdb=mdb, nu_grid=nu_grid, diffmode=diffmode, auto_trange=[art.Tlow, art.Thigh]
    )

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, mdb.molmass, gravity)

    F0 = art.run(dtau, Tarr)

    if db == "hitemp":
        filename = files("exojax").joinpath(
            "data/testdata/" + TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
        )
    elif db == "exomol":
        filename = files("exojax").joinpath(
            "data/testdata/" + TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
        )

    dat = pd.read_csv(filename, delimiter=",", names=("nus", "flux"))
    residual = np.abs(F0 / dat["flux"].values - 1.0)
    assert np.all(residual < 0.005)
