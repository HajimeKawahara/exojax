"""generate test data

"""
from exojax.spec import api
from exojax.utils.grids import wavenumber_grid
import pickle
import dill

from jax.config import config

config.update("jax_enable_x64", True)


def gendata_moldb(database):
    """generate test data for CO exomol
    """

    Nx = 20000
    nus, wav, reso = wavenumber_grid(22900.0,
                                     23100.0,
                                     Nx,
                                     unit='AA',
                                     xsmode="modit")

    if database == "exomol":
        from exojax.test.data import TESTDATA_moldb_CO_EXOMOL as filename
        mdbCO = api.MdbExomol('.database/CO/12C-16O/Li2015',
                              nus,
                              inherit_dataframe=False,
                              gpu_transfer=True)
    elif database == "hitemp":
        from exojax.test.data import TESTDATA_moldb_CO_HITEMP as filename
        mdbCO = api.MdbHitemp('CO',
                              nus,
                              isotope=None,
                              inherit_dataframe=False,
                              gpu_transfer=True)
    elif database == "hitemp_isotope":
        from exojax.test.data import TESTDATA_moldb_CO_HITEMP_SINGLE_ISOTOPE as filename
        mdbCO = api.MdbHitemp('CO', nus, gpu_transfer=True, isotope=1)

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
