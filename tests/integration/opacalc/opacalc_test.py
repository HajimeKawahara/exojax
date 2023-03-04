"""
    Notes:
         These tests are classified as the integration test because it 
         sometimes ends up jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXAUST due to limited resources.

"""


import pytest
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.test.emulate_mdb import mock_mdb

from exojax.spec.opacalc import OpaPremodit
from exojax.spec.opacalc import OpaModit
from exojax.spec.opacalc import OpaDirect
import numpy as np


@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_OpaModit(db):
    mdb = mock_mdb(db)
    nu_grid, wav, res = mock_wavenumber_grid()
    opa = OpaModit(mdb=mdb, nu_grid=nu_grid)
    assert opa.method == "modit"


@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_OpaDirect(db):
    mdb = mock_mdb(db)
    nu_grid, wav, res = mock_wavenumber_grid()
    opa = OpaDirect(mdb=mdb, nu_grid=nu_grid)
    #check opainfo, should provide nu_matrix
    nmshape = (len(mdb.nu_lines), len(nu_grid))
    assert np.shape(opa.opainfo) == nmshape


@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_OpaPremodit_manual(db):
    mdb = mock_mdb(db)
    nu_grid, wav, res = mock_wavenumber_grid()
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid)
    Tref = 500.0
    Twt = 1000.0
    dE = 100.0
    opa.manual_setting(Twt=Twt, Tref=Tref, dE=dE)

    assert opa.Tref == Tref
    assert opa.Twt == Twt
    assert opa.dE == dE


@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_OpaPremodit_manual_params(db):
    mdb = mock_mdb(db)
    nu_grid, wav, res = mock_wavenumber_grid()
    #change Tref
    Tref = 500.0
    Twt = 1000.0
    dE = 100.0
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, manual_params=[dE, Tref, Twt])
    assert opa.Tref == Tref
    assert opa.Twt == Twt
    assert opa.dE == dE


@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_OpaPremodit_auto(db):
    mdb = mock_mdb(db)
    nu_grid, wav, res = mock_wavenumber_grid()
    Tl = 500.0
    Tu = 1200.0
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, auto_trange=[Tl, Tu])
    assert opa.Tref == pytest.approx(1153.6267095763965)
    assert opa.Twt == pytest.approx(554.1714566743503)
    assert opa.dE == pytest.approx(2250.0)


if __name__ == "__main__":
    #db = "exomol"
    db = "hitemp"

    test_OpaModit(db)
    test_OpaDirect(db)
    test_OpaPremodit_manual(db)
    test_OpaPremodit_manual_params(db)
    test_OpaPremodit_auto(db)
