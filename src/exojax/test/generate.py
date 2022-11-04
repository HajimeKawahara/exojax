"""generate test data

"""
from exojax.spec import api
from exojax.utils.grids import wavenumber_grid
import pickle


def gendata_moldb(database):
    """generate test data for CO exomol
    """
    Nx = 10000
    nus, wav, res = wavenumber_grid(22920.0, 24000.0, Nx, unit='AA')
    if database == "exomol":
        from exojax.test.data import TESTDATA_moldb_CO_EXOMOL as filename
        mdbCO = api.MdbExomol('.database/CO/12C-16O/Li2015',
                              nus,
                              crit=1e-35,
                              Ttyp=296.0,
                              gpu_transfer=True)
    elif database == "hitemp":
        from exojax.test.data import TESTDATA_moldb_CO_HITEMP as filename
        mdbCO = api.MdbHitemp('CO',
                              nus,
                              crit=1e-35,
                              Ttyp=296.0,
                              gpu_transfer=True)
    elif database == "hitemp_isotope":
        from exojax.test.data import TESTDATA_moldb_CO_HITEMP_SINGLE_ISOTOPE as filename
        mdbCO = api.MdbHitemp('CO',
                              nus,
                              crit=1e-35,
                              Ttyp=296.0,
                              isotope=1,
                              gpu_transfer=True)
    with open(filename, 'wb') as f:
        pickle.dump(mdbCO, f)


if __name__ == "__main__":
    gendata_moldb("exomol")
    gendata_moldb("hitemp")
    gendata_moldb("hitemp_isotope")

    print(
        "to include the generated files in the package, move pickles to exojax/src/data/testdata/"
    )
