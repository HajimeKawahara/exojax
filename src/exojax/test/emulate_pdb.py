"""emulate mdb class for unittest
"""
import pickle
import pkg_resources
import os
import shutil
from exojax.spec import api
from exojax.test.data import TESTDATA_pardb_NH3


def mock_pdb_clouds_NH3():
    """default mock pdb clouds
    Returns:
        PdbClouds instance  
    """
    dirname = pkg_resources.resource_filename('exojax', 'data/testdata')
    
    return pdb
