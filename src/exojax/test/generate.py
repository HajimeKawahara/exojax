"""generate test data

"""
from exojax.spec import api
from exojax.utils.grids import wavenumber_grid
<<<<<<< HEAD
import pickle
from jax.config import config
config.update("jax_enable_x64", True)
=======
import numpy as np
import pathlib
import os
>>>>>>> 0ca6550b8cf91a76230b71cf5995c32140a6f17d


def gendata_moldb(database):
    """generate test data for CO exomol
    """
<<<<<<< HEAD

    Nx = 20000
    nus, wav, reso = wavenumber_grid(22900.0,
                                     23100.0,
                                     Nx,
                                     unit='AA',
                                     xsmode="modit")

=======
    Nx = 10000
    lambda0 = 22920.0
    lambda1 = 24000.0
    nus, wav, res = wavenumber_grid(lambda0, lambda1, Nx, unit='AA')
>>>>>>> 0ca6550b8cf91a76230b71cf5995c32140a6f17d
    if database == "exomol":
        mdbCO = api.MdbExomol('.database/CO/12C-16O/Li2015',
                              nus,
<<<<<<< HEAD
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
=======
                              crit=1e-35,
                              Ttyp=296.0,
                              inherit_dataframe=True,
                              gpu_transfer=True)

        trans_filename = "12C-16O__SAMPLE.trans"
        mask = (mdbCO.df["nu_lines"] <= nus[-1]) * (mdbCO.df["nu_lines"] >=
                                                    nus[0])
        maskeddf = mdbCO.df[mask]
        directory_name = "CO/12C-16O/SAMPLE"
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        cdir = os.getcwd()
        os.chdir(directory_name)
        save_trans(trans_filename, maskeddf)
        os.chdir(cdir)



def make_hdf():
    path = "CO/12C-16O/SAMPLE"
    path = pathlib.Path(path).expanduser()
    print(path)
    Nx = 10000
    lambda0 = 22920.0
    lambda1 = 24000.0
    nus, wav, res = wavenumber_grid(lambda0, lambda1, Nx, unit='AA')
    mdbCO = api.MdbExomol(str(path),
                          nus,
                          crit=1e-35,
                          Ttyp=296.0,
                          inherit_dataframe=True,
                          gpu_transfer=True)
>>>>>>> 0ca6550b8cf91a76230b71cf5995c32140a6f17d


def save_trans(trans_filename, maskeddf):
    maskeddf.export_csv(trans_filename,
                        columns=["i_upper", "i_lower", "A", "nu_lines"],
                        sep="\t",
                        header=False)
    import bz2
    with open(trans_filename, 'rb') as f_in:
        with bz2.open(trans_filename + ".bz2", 'wb') as f_out:
            f_out.writelines(f_in)


if __name__ == "__main__":
    gendata_moldb("exomol")
<<<<<<< HEAD
    gendata_moldb("hitemp")
    gendata_moldb("hitemp_isotope")
    gendata_moldb_H2O()
    print(
        "to include the generated files in the package, move pickles to exojax/src/exojax/data/testdata/"
    )
=======
    #print("cp some files from data/testdata/CO/12C-16O/SAMPLE/ to CO/12C-16O/SAMPLE/")
    #make_hdf()
 
>>>>>>> 0ca6550b8cf91a76230b71cf5995c32140a6f17d
