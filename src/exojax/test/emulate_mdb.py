"""emulate mdb class for unittest
"""
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_CO_HITEMP
from exojax.test.data import TESTDATA_moldb_CO_HITEMP_SINGLE_ISOTOPE
from exojax.test.data import TESTDATA_moldb_VALD
import os
import shutil
from exojax.spec import api
from exojax.utils.grids import wavenumber_grid


def mock_mdbExomol():
    """default mock mdb of the ExoMol form for unit test   
    Returns:
        mdbExomol instance  
    """
    dirname = pkg_resources.resource_filename(
        'exojax', 'data/testdata/CO')
    target_dir = os.getcwd()+"/CO"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(dirname,target_dir)
    path="CO/12C-16O/SAMPLE"
    Nx = 10000
    lambda0 = 22920.0
    lambda1 = 24000.0
    nus, wav, res = wavenumber_grid(lambda0, lambda1, Nx, unit='AA')    
    mdb = api.MdbExomol(str(path),
                              nus,
                              crit=1e-35,
                              Ttyp=296.0,
                              inherit_dataframe=True,
                              gpu_transfer=True)
    
    return mdb


def mock_mdbHitemp(multi_isotope=False):
    """default mock mdb of the Hitemp form for unit test   
    
    Args:
        multi isotope: if True, use multi isotope mdb
    
    Returns:
        mdbHitemp instance  
    """
    if multi_isotope:
        filename = TESTDATA_moldb_CO_HITEMP
    else:
        filename = TESTDATA_moldb_CO_HITEMP_SINGLE_ISOTOPE

    filename = pkg_resources.resource_filename('exojax',
                                               'data/testdata/' + filename)
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
    return mdb


def mock_mdbVALD():
    """default mock mdb of the VALD form for unit test
    Returns:
        AdbVald instance
    """
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_moldb_VALD)
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
    return mdb


if __name__ == "__main__":
    mdb = mock_mdbExomol()
    print(mdb.df)