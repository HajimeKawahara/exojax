"""emulate mdb class for unittest
"""

import pickle
import os
import shutil

from exojax.spec import api
from exojax.test.data import TESTDATA_moldb_VALD
from exojax.test.data import get_testdata_filename
from exojax.utils.grids import wavenumber_grid


def mock_mdb(db):
    """data base selector

    Args:
        db (_type_): db name = "exomol", "hitemp"

    Raises:
        ValueError: _description_

    Returns:
        _type_: mdb object
    """
    if db == "exomol":
        mdb = mock_mdbExomol()
    elif db == "hitemp":
        mdb = mock_mdbHitemp()
    else:
        raise ValueError("no exisiting dbname.")
    return mdb


def mock_wavenumber_grid():
    Nx = 20000
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(
        lambda0, lambda1, Nx, unit="AA", xsmode="modit", wavelength_order="ascending"
    )
    return nus, wav, res


def mock_mdbExomol(molecule="CO", crit=0.0):
    """default mock mdb of the ExoMol form for unit test
    Args:
        molecule (str, optional): "CO" or "H2O". Defaults to "CO".
        crit (float, optional): line strength criterion. Defaults to 0.

    Returns:
        mdbExomol instance
    """
    
    dirname = get_testdata_filename(molecule)
    target_dir = os.getcwd() + "/"+molecule
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(dirname, target_dir)
    
    path_dict = {
        "CO": "CO/12C-16O/SAMPLE",
        "H2O": "H2O/1H2-16O/SAMPLE",
    }
    path = path_dict[molecule]
    nus, wav, res = mock_wavenumber_grid()
    mdb = api.MdbExomol(
        str(path),
        nus,
        crit=crit,
        inherit_dataframe=True,
        gpu_transfer=True,
        broadf_download=False,
    )
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

    parfile = get_testdata_filename(TESTDATA_CO_HITEMP_PARFILE)
    nus, wav, res = mock_wavenumber_grid()
    mdb = api.MdbHitemp(
        "CO",
        nus,
        isotope=isotope,
        parfile=parfile,
        inherit_dataframe=True,
        gpu_transfer=True,
    )
    return mdb


def mock_mdbVALD():
    """default mock mdb of the VALD form for unit test
    Returns:
        AdbVald instance
    """
    filename = get_testdata_filename(TESTDATA_moldb_VALD)
    with open(filename, "rb") as f:
        mdb = pickle.load(f)
    return mdb


if __name__ == "__main__":
    mdb = mock_mdbExomol()
    mdb = mock_mdbExomol("H2O")
#    mdb = mock_mdbHitemp()
#    print(mdb.df)
