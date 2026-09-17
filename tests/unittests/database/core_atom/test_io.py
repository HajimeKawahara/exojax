"""test for loading atomic data."""

from exojax.database.core_atom.io import load_atomicdata, load_pf_Barklem2016


def test_loadatom():
    data = load_atomicdata()
    assert data['ionizationE1'][0] == 13.595


def test_barklem():
    data = load_pf_Barklem2016()
    assert data[1]['1.00000e-05'][0] == 2.0
