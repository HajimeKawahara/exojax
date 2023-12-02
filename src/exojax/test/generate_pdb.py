"""generate test pdb

"""
from exojax.spec import pardb
import pickle


def gendata_pardb():
    """generate test data for ..."""
    from exojax.test.data import TESTDATA_pardb_NH3 as filename

    pdb_nh3 = pardb.PdbCloud("NH3",path="./")
    pdb_nh3.load_miegrid()
    with open(filename, "wb") as f:
        pickle.dump(pdb_nh3, f)


if __name__ == "__main__":
    if True:
        pdb_nh3 = pardb.PdbCloud("NH3")
        pdb_nh3.generate_miegrid(
            sigmagmin=-1.0,
            sigmagmax=1.0,
            Nsigmag=4,
            rg_max=-4.0,
            Nrg=4,
        )
    gendata_pardb()
    print("move pickle file to ../data/testdata/")