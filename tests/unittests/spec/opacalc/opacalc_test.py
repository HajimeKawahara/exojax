import pytest
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.spec.opacalc import OpaPremodit
from exojax.spec.opacalc import OpaDirect
from exojax.utils.grids import wavenumber_grid

#def test_OpaCalc():
#    mdb = mock_mdbExomol()
#    opc = OpaCalc()
#    assert opc.opainfo is None


def test_OpaDirect():
    mdb = mock_mdbExomol()
    Nx = 5000
    nu_grid, wav, res = mock_wavenumber_grid()
    opa = OpaDirect(mdb=mdb, nu_grid=nu_grid)
    

def test_OpaPremodit_manual():
    mdb = mock_mdbExomol()
    Nx = 5000
    nu_grid, wav, res = mock_wavenumber_grid()
    #change Tref
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid)
    Tref = 500.0
    Twt = 1000.0
    dE = 100.0
    opa.manual_setting(Twt=Twt, Tref=Tref, dE=dE)

    assert opa.Tref == Tref
    assert opa.Twt == Twt
    assert opa.dE == dE


def test_OpaPremodit_manual_params():
    mdb = mock_mdbExomol()
    Nx = 5000
    nu_grid, wav, res = mock_wavenumber_grid()
    #change Tref
    Tref = 500.0
    Twt = 1000.0
    dE = 100.0
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, manual_params=[dE, Tref, Twt])
    assert opa.Tref == Tref
    assert opa.Twt == Twt
    assert opa.dE == dE


def test_OpaPremodit_auto():
    mdb = mock_mdbExomol()
    Nx = 5000
    nu_grid, wav, res = mock_wavenumber_grid()
    Tl = 500.0
    Tu = 1200.0
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid, auto_trange=[Tl, Tu])
    assert opa.Tref == pytest.approx(1153.6267095763965)
    assert opa.Twt == pytest.approx(554.1714566743503)
    assert opa.dE == pytest.approx(2250.0)

if __name__ == "__main__":
    test_OpaPremodit_manual()
    #test_OpaPremodit_manual_params()
    #test_OpaPremodit_auto()