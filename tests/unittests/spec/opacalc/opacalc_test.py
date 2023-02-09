import pytest
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.opacalc import OpaDirect
import numpy as np


def _select_db(db):
    """data base selector

    Args:
        db (_type_): db name = "exomol", "hitemp"

    Raises:
        ValueError: _description_

    Returns:
        _type_: mdb object
    """
    if db == "exomol":
        mdb = mock_mdbExomol()
    elif db == "hitemp":
        mdb = mock_mdbHitemp()
    else:
        raise ValueError("no exisiting dbname.")
    return mdb



@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_OpaDirect(db):
    mdb = _select_db(db)
    nu_grid, wav, res = mock_wavenumber_grid()
    opa = OpaDirect(mdb=mdb, nu_grid=nu_grid)
    #check opainfo, should provide nu_matrix
    nmshape = (len(mdb.nu_lines),len(nu_grid))
    assert np.shape(opa.opainfo) == nmshape

@pytest.mark.parametrize("db", ["exomol", "hitemp"])
def test_OpaPremodit_manual(db):
    mdb = _select_db(db)
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
    mdb = _select_db(db)
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
    mdb = _select_db(db)
    nu_grid, wav, res = mock_wavenumber_grid()
    Tl = 500.0
    Tu = 1200.0
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, auto_trange=[Tl, Tu])
    assert opa.Tref == pytest.approx(1153.6267095763965)
    assert opa.Twt == pytest.approx(554.1714566743503)
    assert opa.dE == pytest.approx(2250.0)

if __name__ == "__main__":
    db = "exomol"
    test_OpaDirect(db)
    #test_OpaPremodit_manual(db)
    #test_OpaPremodit_manual_params(db)
    #test_OpaPremodit_auto(db)