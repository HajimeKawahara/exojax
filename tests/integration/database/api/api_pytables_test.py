from shutil import copyfile

from exojax.database.hitemp.api import MdbHitemp
from exojax.database.exomol.api import MdbExomol
from exojax.test.data import TESTDATA_CO_HITEMP_PARFILE, get_testdata_filename


def test_mdb_exomol_pytables():
    mdb = MdbExomol(
        "CO/12C-16O/Li2015", nurange=[4000.0, 4100.0], engine="pytables"
    )
    assert mdb.engine == "pytables"


def test_mdb_hitemp_pytables(tmp_path):
    parfile = tmp_path / "CO_HITEMP_SAMPLE.par"
    copyfile(get_testdata_filename(TESTDATA_CO_HITEMP_PARFILE), parfile)
    mdb = MdbHitemp(
        "CO",
        nurange=[4200.0, 4300.0],
        isotope=1,
        elower_max=3300.0,
        engine="pytables",
        parfile=parfile,
    )
    assert mdb.engine == "pytables"
