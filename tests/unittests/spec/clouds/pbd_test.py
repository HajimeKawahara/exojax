from exojax.test.emulate_pdb import mock_PdbPlouds
import numpy as np
def test_pdb_clouds():
    pdb = mock_PdbPlouds()
    
def test_pdb_clouds_nurange():
    pdb = mock_PdbPlouds()
    
    pdb.nurange = [15000.0,16000.0]
    irange = np.searchsorted(pdb.refraction_index_wavenumber, pdb.nurange)
        
    

    #irange=pdb.get_indices_nurage()
    print(irange[0])
    istart = int(irange[0])
    
    print(pdb.refraction_index_wavenumber[istart])


if __name__ == "__main__":
    test_pdb_clouds_nurange()