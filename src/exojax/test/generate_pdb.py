"""generate test miegrid

"""
import pkg_resources
from exojax.spec import pardb
from exojax.test.data import TESTDATA_refrind

def gendata_miegrid():
    """
    generates miegrid for test.refrind
    """
    refrind_path = pkg_resources.resource_filename(
        "exojax", "data/testdata/"+TESTDATA_refrind
    )

    pdb_nh3 = pardb.PdbCloud("test", download=False, refrind_path=refrind_path)
    if True:
        pdb_nh3.generate_miegrid(
            log_sigmagmin=-1.0,
            log_sigmagmax=1.0,
            Nsigmag=4,
            log_rg_min=-7.0,
            log_rg_max=-4.0,
            Nrg=4,
        )


if __name__ == "__main__":
    gendata_miegrid()
