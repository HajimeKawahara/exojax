"""
See Issue #510, #515
"""

import copy
from exojax.test.emulate_mdb import mock_mdbHitemp, mock_mdbExomol
from exojax.utils.grids import wavenumber_grid
from exojax.opacity import OpaPremodit
import copy


def test_sideeffect_call():
    nus, wav, res = wavenumber_grid(
        22920.0, 23100.0, 20000, unit="AA", xsmode="premodit"
    )
    mdb = mock_mdbHitemp(multi_isotope=True)
    opa1 = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True, auto_trange=[500.0, 1000])
    opa1_orig = copy.deepcopy(opa1)
    opa2 = OpaPremodit(
        mdb, nu_grid=nus, allow_32bit=True, auto_trange=[500.0, 1200]
    )  # used the same mdb used in opa1
    print(opa1 == opa1_orig)
    assert opa1 == opa1_orig

def test_sideeffect_call_exomol():
    nus, wav, res = wavenumber_grid(
        22920.0, 23100.0, 20000, unit="AA", xsmode="premodit"
    )
    mdb = mock_mdbExomol()
    opa1 = OpaPremodit(mdb, nu_grid=nus, allow_32bit=True, auto_trange=[500.0, 1000])
    opa1_orig = copy.deepcopy(opa1)
    opa2 = OpaPremodit(
        mdb, nu_grid=nus, allow_32bit=True, auto_trange=[500.0, 1200]
    )  # used the same mdb used in opa1
    print(opa1 == opa1_orig)
    assert opa1 == opa1_orig


if __name__ == "__main__":
    test_sideeffect_call()
    test_sideeffect_call_exomol()