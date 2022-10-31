"""emulate mdb class for unittest
"""
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_CO_EXOMOL
from exojax.test.data import TESTDATA_moldb_CO_HITEMP
from exojax.test.data import TESTDATA_moldb_CO_HITEMP_SINGLE_ISOTOPE
from exojax.test.data import TESTDATA_moldb_VALD


def mock_mdbExomol():
    """default mock mdb of the ExoMol form for unit test   
    Returns:
        mdbExomol instance  
    """
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + TESTDATA_moldb_CO_EXOMOL)
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
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
