"""test for loading atomic data."""

from exojax.database.core_atom.io import load_atomicdata, load_pf_Barklem2016
from exojax.utils.zsol import nsol


def test_loadatom():
    data = load_atomicdata()
    assert data['ionizationE1'][0] == 13.595


def test_barklem():
    data = load_pf_Barklem2016()
    assert data[1]['1.00000e-05'][0] == 2.0


def test_nsol():
    nsun = nsol()
    assert nsun['Mg'] == 3.275853193332226e-05


if __name__ == '__main__':
    test_loadatom()
    test_barklem()
    test_nsol()
