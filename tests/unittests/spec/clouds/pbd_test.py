"""unit tests for pardb
"""
from exojax.test.emulate_pdb import mock_PdbPlouds
import numpy as np
import pytest

def test_pdb_clouds():
    pdb = mock_PdbPlouds()
    pdb.load_miegrid()

def test_pdb_clouds_nurange_initialize():
    pdb = mock_PdbPlouds(nurange=[12000.0, 15000.0])
    pdb.load_miegrid()

    assert pdb.refraction_index_wavenumber[0] == pytest.approx(11990.407673860911)
    assert np.all(pdb.refraction_index_wavenumber == pytest.approx(1.e7/pdb.refraction_index_wavelength_nm))


def test_pdb_clouds_nurange_redefine():
    pdb = mock_PdbPlouds()
    pdb.load_miegrid()
    pdb.nurange = [12000.0, 15000.0]
    
    pdb.reset_miegrid_for_nurange()
    
    assert pdb.refraction_index_wavenumber[0] == pytest.approx(11990.407673860911)
    assert np.all(pdb.refraction_index_wavenumber == pytest.approx(1.e7/pdb.refraction_index_wavelength_nm))
    

if __name__ == "__main__":
    test_pdb_clouds_nurange_initialize()
    #test_pdb_clouds_nurange_redefine()
