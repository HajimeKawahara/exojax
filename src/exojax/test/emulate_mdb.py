"""emulate mdb class for unittest
"""
import pickle
import pkg_resources
from exojax.test.data import TESTDATA_moldb_CO_EXOMOL

def mock_mdbExoMol():
    """default mock mdb of the ExoMol form for unit test   
    Returns:
        mdbExoMol instance  
    """
    filename = pkg_resources.resource_filename('exojax', 'data/testdata/'+TESTDATA_moldb_CO_EXOMOL)
    with open(filename, 'rb') as f:
        mdb = pickle.load(f)
    return mdb