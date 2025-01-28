"""generate test miegrid

"""

from importlib.resources import files
from exojax.spec import pardb
from exojax.test.data import TESTDATA_refrind


def gendata_miegrid():
    """generates miegrid for test.refrind

    Warnings:
        this is just for testdata. Not for real use!

    """
    refrind_path = files("exojax").joinpath("data/testdata/" + TESTDATA_refrind)
    pdb_nh3 = pardb.PdbCloud("test", download=False, refrind_path=refrind_path)
    if True:
        pdb_nh3.generate_miegrid(
            sigmagmin=1.01,
            sigmagmax=4.0,
            Nsigmag=10,
            log_rg_min=-7.0,
            log_rg_max=-4.0,
            Nrg=4,
        )


if __name__ == "__main__":
    gendata_miegrid()
