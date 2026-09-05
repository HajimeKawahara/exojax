"""emulate mdb class for unittest"""

import pickle
import shutil
from pathlib import Path
from tempfile import mkdtemp

from exojax.database.exomol.api import MdbExomol
from exojax.database.hitemp.api import MdbHitemp
from exojax.database.hargreaves.api import MdbHargreaves
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
    elif db == "hargreaves":
        mdb = mock_mdbHargreaves()
    else:
        raise ValueError("no exisiting dbname.")
    return mdb


def mock_wavenumber_grid(lambda0=22920.0, lambda1=23100.0, Nx=20000):
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
    # Keep both source data and other database instances intact. Tests run in
    # a temporary working directory that owns these copies and their caches.
    root = Path(mkdtemp(prefix="exojax-exomol-", dir=Path.cwd()))
    target_dir = root / molecule
    shutil.copytree(dirname, target_dir)

    path_dict = {
        "CO": "CO/12C-16O/SAMPLE",
        "H2O": "H2O/1H2-16O/SAMPLE",
    }
    path = root / path_dict[molecule]
    nus, wav, res = mock_wavenumber_grid()
    mdb = MdbExomol(
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

    source_parfile = Path(get_testdata_filename(TESTDATA_CO_HITEMP_PARFILE))
    root = Path(mkdtemp(prefix="exojax-hitemp-", dir=Path.cwd()))
    parfile = root / source_parfile.name
    shutil.copy2(source_parfile, parfile)
    nus, _, _ = mock_wavenumber_grid()
    mdb = MdbHitemp(
        "CO",
        nus,
        isotope=isotope,
        parfile=str(parfile),
        inherit_dataframe=True,
        gpu_transfer=True,
    )
    return mdb


def mock_mdbVALD():
    """default mock mdb of the VALD form for unit test
    Returns:
        AdbVald instance
    """
    class _AdbValdUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "exojax.spec.moldb" and name == "AdbVald":
                from exojax.database.vald.api import AdbVald

                return AdbVald
            if module == "pandas.core.internals.blocks" and name == "new_block":
                from pandas._libs.internals import BlockPlacement
                from pandas.core.internals.blocks import new_block

                def new_block_compat(values, placement, ndim=None):
                    if isinstance(placement, slice):
                        placement = BlockPlacement(placement)
                    return new_block(values, placement=placement, ndim=ndim)

                return new_block_compat
            return super().find_class(module, name)

    filename = get_testdata_filename(TESTDATA_moldb_VALD)
    with open(filename, "rb") as f:
        mdb = _AdbValdUnpickler(f).load()
    if not hasattr(mdb, "Tref"):
        from exojax.utils.constants import Tref_original

        mdb.Tref = Tref_original
    return mdb


def mock_mdbHargreaves():
    """default mock mdb of the Hargreaves 2010 form for unit test
    Returns:
        MdbHargreaves instance
    """

    path = "FeH/SAMPLE"
    nus, _, _ = mock_wavenumber_grid(lambda0=15820.0, lambda1=20040.0)
    mdb = MdbHargreaves(
        str(path),
        nus,
    )
    return mdb


if __name__ == "__main__":
    #    mdb = mock_mdbExomol()
    #    mdb = mock_mdbExomol("H2O")
    #    mdb = mock_mdbHitemp()
    #    print(mdb.df)
    mdb = mock_mdbHargreaves()
