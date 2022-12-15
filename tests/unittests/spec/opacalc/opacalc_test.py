import pytest
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.spec.opacalc import OpaCalc
from exojax.spec.opacalc import OpaPremodit

def test_OpaCalc():
    mdb = mock_mdbExomol()
    opc = OpaCalc()
    assert opc.opainfo is None


def test_OpaPremodit():
    mdb = mock_mdbExomol()
    opc = OpaPremodit()
    