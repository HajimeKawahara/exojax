"""generate test data

"""
from exojax.spec import api
from exojax.utils.grids import wavenumber_grid
import numpy as np
import pathlib
import os


def gendata_moldb(database):
    """generate test data for CO exomol
    """
    Nx = 10000
    lambda0 = 22920.0
    lambda1 = 24000.0
    nus, wav, res = wavenumber_grid(lambda0, lambda1, Nx, unit='AA')
    if database == "exomol":
        mdbCO = api.MdbExomol('.database/CO/12C-16O/Li2015',
                              nus,
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

    elif database == "hitemp_isotope":
        parfile = '/home/kawahara/exojax/tests/integration/moldb/05_HITEMP_SAMPLE.par'
        mdbCO = api.MdbHitemp('CO',
                              nus,
                              isotope=1,
                              parfile=parfile,
                              inherit_dataframe=True,
                              gpu_transfer=True)
        print(mdbCO.df)


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
    #gendata_moldb("exomol")
    #print("cp some files from data/testdata/CO/12C-16O/SAMPLE/ to CO/12C-16O/SAMPLE/")
    #make_hdf()
    #gendata_moldb("hitemp")
    gendata_moldb("hitemp_isotope")
