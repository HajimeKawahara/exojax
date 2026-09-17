"""Reusing a database must not change an existing calculator (Issues #510 and #515)."""

import copy

import pytest

from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_mdbExomol, mock_mdbHitemp
from exojax.utils.grids import wavenumber_grid


@pytest.mark.parametrize("db", ["hitemp", "exomol"])
def test_reusing_mdb_preserves_existing_calculator(db):
    nu_grid, _, _ = wavenumber_grid(
        22920.0, 23100.0, 64, unit="AA", xsmode="premodit"
    )
    mdb = mock_mdbHitemp(multi_isotope=True) if db == "hitemp" else mock_mdbExomol()
    opa = OpaPremodit(mdb, nu_grid=nu_grid, auto_trange=[500.0, 1000.0])
    original = copy.deepcopy(opa)
    OpaPremodit(mdb, nu_grid=nu_grid, auto_trange=[500.0, 1200.0])
    assert opa == original
