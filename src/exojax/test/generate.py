"""generate test data for exomol

"""
from exojax.spec import api
from exojax.test.emulate_mdb import mock_wavenumber_grid
import pathlib
import os
from jax.config import config

config.update("jax_enable_x64", True)


def gendata_moldb():
    """generate test data for CO exomol
    """
    nus, wav, res = mock_wavenumber_grid()
    mdbCO = api.MdbExomol('.database/CO/12C-16O/Li2015',
                          nus,
                          inherit_dataframe=True,
                          gpu_transfer=True)

    trans_filename = "12C-16O__SAMPLE.trans"
    mask = (mdbCO.df["nu_lines"] <= nus[-1]) * (mdbCO.df["nu_lines"] >= nus[0])
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
    nus, wav, res = mock_wavenumber_grid()
    mdbCO = api.MdbExomol(str(path),
                          nus,
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
    # regenerate hdf for exomol
    gendata_moldb()
    print(
        "cp some files from data/testdata/CO/12C-16O/SAMPLE/ to CO/12C-16O/SAMPLE/, then do make_hdf"
    )
    make_hdf()
    print(">cp -rf CO ../data/testdata/")