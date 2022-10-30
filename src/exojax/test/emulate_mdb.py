"""emulate mdb class for unittest
"""
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_CO_EXOMOL
from exojax.test.data import TESTDATA_moldb_CO_HITEMP
from exojax.test.data import TESTDATA_moldb_VALD

def mock_mdbExomol():
    """default mock mdb of the ExoMol form for unit test   
    Returns:
        mdbExomol instance  
    """
    filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_CO_EXOMOL)
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
    return mdb

def mock_mdbHitemp():
    """default mock mdb of the Hitemp form for unit test   
    Returns:
        mdbHitemp instance  
    """
    filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_CO_HITEMP)
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
    return mdb


def mock_mdbVALD():
    """default mock mdb of the VALD form for unit test
    Returns:
        AdbVald instance
    """
    filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_VALD)
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
    return mdb

