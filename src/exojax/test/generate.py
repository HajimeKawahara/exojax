"""generate test data for exomol

"""

from exojax.spec import api
from exojax.test.emulate_mdb import mock_wavenumber_grid
import pathlib
import numpy as np
import os
from jax import config

config.update("jax_enable_x64", True)

sample_directory_name = {"CO": "CO/12C-16O/SAMPLE", "H2O": "H2O/1H2-16O/SAMPLE"}


def gendata_moldb(molecules="CO"):
    """generate test data for CO exomol"""
    nus, wav, res = mock_wavenumber_grid()

    exomoldb = {
        "CO": ".database/CO/12C-16O/Li2015",
        "H2O": ".database/H2O/1H2-16O/POKAZATEL",
    }
    line_strength_criterion = {"CO": 0.0, "H2O": 1.0e-30}
    mdb = api.MdbExomol(
        exomoldb[molecules], nus, inherit_dataframe=True, gpu_transfer=True
    )
    line_mask = mdb.df["Sij0"] > line_strength_criterion[molecules]
    trans_filename = {"CO": "12C-16O__SAMPLE.trans", "H2O": "1H2-16O__SAMPLE.trans"}
    mask = (mdb.df["nu_lines"] <= nus[-1]) * (mdb.df["nu_lines"] >= nus[0]) * line_mask
    maskeddf = mdb.df[mask]
    if not os.path.exists(sample_directory_name[molecules]):
        os.makedirs(sample_directory_name[molecules])
    cdir = os.getcwd()
    os.chdir(sample_directory_name[molecules])
    save_trans(trans_filename[molecules], maskeddf)
    os.chdir(cdir)


def make_hdf(molecule):
    path = pathlib.Path(sample_directory_name[molecule]).expanduser()
    nus, wav, res = mock_wavenumber_grid()
    mdb = api.MdbExomol(str(path), nus, inherit_dataframe=True, gpu_transfer=True)


def save_trans(trans_filename, maskeddf):
    maskeddf.export_csv(
        trans_filename,
        columns=["i_upper", "i_lower", "A", "nu_lines"],
        sep="\t",
        header=False,
    )
    import bz2

    with open(trans_filename, "rb") as f_in:
        with bz2.open(trans_filename + ".bz2", "wb") as f_out:
            f_out.writelines(f_in)


if __name__ == "__main__":
    # regenerate hdf for exomol

    molecule = "H2O"
    gendata_moldb(molecule)
    print(
        "cp some files from data/testdata/"
        + sample_directory_name[molecule]
        + " to "
        + sample_directory_name[molecule]
        + ", then do make_hdf"
    )
    make_hdf(molecule)
    print(">cp -rf " + molecule + " ../data/testdata/")
