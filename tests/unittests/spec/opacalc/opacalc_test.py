import pytest
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.spec.opacalc import OpaCalc
from exojax.spec.opacalc import OpaPremodit
from exojax.utils.grids import wavenumber_grid

#def test_OpaCalc():
#    mdb = mock_mdbExomol()
#    opc = OpaCalc()
#    assert opc.opainfo is None


def test_OpaPremodit_manual():
    mdb = mock_mdbExomol()
    Nx = 5000
    nu_grid, wav, res = wavenumber_grid(22800.0,
                                        23100.0,
                                        Nx,
                                        unit='AA',
                                        xsmode="premodit")
    #change Tref
    opa = OpaPremodit(mdb=mdb, nu_grid=nu_grid)
    Tref = 500.0
    Twt = 1000.0
    dE = 100.0
    opa.manual_setting(Twt=Twt, Tref=Tref, dE=dE)

    assert opa.Tref == Tref
    assert opa.Twt == Twt
    assert opa.dE == dE


if __name__ == "__main__":
    test_OpaPremodit_manual()
