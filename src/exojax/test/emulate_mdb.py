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
    dirname = pkg_resources.resource_filename('exojax', 'data/testdata/CO')
    target_dir = os.getcwd() + "/CO"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(dirname, target_dir)
    path = "CO/12C-16O/SAMPLE"
    Nx = 20000
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(lambda0, lambda1, Nx, unit='AA', xsmode="modit")
    mdb = api.MdbExomol(str(path),
                        nus,
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
        isotope = 0
    else:
        isotope = 1

    from exojax.test.data import TESTDATA_CO_HITEMP_PARFILE
    parfile = pkg_resources.resource_filename(
        'exojax', 'data/testdata/CO/' + TESTDATA_CO_HITEMP_PARFILE)
    Nx = 20000
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(lambda0,
                                    lambda1,
                                    Nx,
                                    unit='AA',
                                    xsmode="modit")
    mdb = api.MdbHitemp('CO',
                        nus,
                        isotope=isotope,
                        parfile=parfile,
                        inherit_dataframe=True,
                        gpu_transfer=True)
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
    #mdb = mock_mdbHitemp()
    print(mdb.df)