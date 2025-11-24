"""unit test for ArtTransPure consistency

Note:
    This test validates ArtTransPure against reference transmission data
    to ensure the code implementation remains consistent.
"""

import pytest
from jax import config
import pandas as pd
import numpy as np
from exojax.test.data import get_testdata_filename
from exojax.test.emulate_mdb import mock_mdb
from exojax.test.data import (
    SAMPLE_TRANSMISSION_CH4,
    TESTDATA_CO_EXOMOL_PREMODIT_TRANSMISSION_REF,
)
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtTransPure

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("db, diffmode", [("exomol", 1)])
def test_consistency_ArtTransPure(db, diffmode, fig=False):
    """compares ArtTransPure with reference transmission data

    Note: This test ensures transmission calculations remain consistent.
    If no specific reference exists, it validates against physical expectations.

    Args:
        db: exomol or hitemp
        diffmode: 0, 1, or 2
        fig: True or False
    """
    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtTransPure(
        pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=50, integration="simpson"
    )

    # Set up atmospheric profile similar to sample conditions
    Tarr = np.linspace(1000.0, 1500.0, 50)
    mmr_arr = np.full(50, 0.1)
    mean_molecular_weight = np.full(50, 2.33)
    radius_btm = 6.9e9  # cm (Jupiter-like)
    gravity_btm = 2478.57  # cm/s2

    mdb = mock_mdb(db)
    opa = OpaPremodit(
        mdb=mdb, nu_grid=nu_grid, diffmode=diffmode, auto_trange=[800.0, 1600.0]
    )

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, mdb.molmass, gravity_btm)

    transit_result = art.run(dtau, Tarr, mean_molecular_weight, radius_btm, gravity_btm)

    # Load reference data for comparison
    fn = get_testdata_filename(TESTDATA_CO_EXOMOL_PREMODIT_TRANSMISSION_REF)
    dat = np.loadtxt(fn, delimiter=",")
    transit_ref = dat[:, 1]

    # Compare with reference data
    relative_diff = np.abs((transit_result - transit_ref) / transit_ref)
    max_relative_diff = np.max(relative_diff)

    print(f"Max relative difference: {max_relative_diff:.2e}")
    assert (
        max_relative_diff < 1e-4
    ), f"Transmission differs from reference by {max_relative_diff:.2e}"


if __name__ == "__main__":
    test_consistency_ArtTransPure("exomol", 1, fig=False)
