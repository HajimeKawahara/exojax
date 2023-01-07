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
                              inherit_dataframe=False,
                              gpu_transfer=True)
    elif database == "hitemp":
        from exojax.test.data import TESTDATA_moldb_CO_HITEMP as filename
        mdbCO = api.MdbHitemp('CO',
                              nus,
                              crit=1e-35,
                              Ttyp=296.0,
                              isotope=None,
                              inherit_dataframe=False,
                              gpu_transfer=True)
    elif database == "hitemp_isotope":
        from exojax.test.data import TESTDATA_moldb_CO_HITEMP_SINGLE_ISOTOPE as filename
        mdbCO = api.MdbHitemp('CO',
                              nus,
                              crit=1e-35,
                              Ttyp=296.0,
                              isotope=1,
                              inherit_dataframe=False,                              
                              gpu_transfer=True)
    with open(filename, 'wb') as f:
        pickle.dump(mdbCO, f)


def gendata_moldb_H2O():
    from exojax.test.data import TESTDATA_moldb_H2O_EXOMOL
    filename = TESTDATA_moldb_H2O_EXOMOL
    wls, wll, wavenumber_grid_res = 15540, 15550, 0.05
    nus, wav, reso = wavenumber_grid(wls,
                                     wll,
                                     int((wll - wls) / wavenumber_grid_res),
                                     unit="AA",
                                     xsmode="modit")
    mdb = api.MdbExomol('.database/H2O/1H2-16O/POKAZATEL',
                        nus,
                        inherit_dataframe=False,
                        crit=1.e-29,
                        gpu_transfer=True)
    with open(filename, 'wb') as f:
        pickle.dump(mdb, f)


if __name__ == "__main__":
    gendata_moldb("exomol")
    gendata_moldb("hitemp")
    gendata_moldb("hitemp_isotope")
    gendata_moldb_H2O()
    print(
        "to include the generated files in the package, move pickles to exojax/src/exojax/data/testdata/"
    )
