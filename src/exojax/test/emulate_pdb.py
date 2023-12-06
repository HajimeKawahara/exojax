"""emulate mdb class for unittest
"""
import pkg_resources
from exojax.test.data import TESTDATA_refrind
from exojax.spec import pardb


def mock_PdbPlouds(nurange=None):
    """default mock pdb clouds with miegrid file

    Note:
        The refrind file is data/testdata/test.refrind
        The migrid file is data/testdata/miegrid_lognorm_test.mg.npz
        These files should be copied in exojax installed directory when ExoJAX is installed

    Returns:
        PdbClouds instance
    """
    refrind_path = pkg_resources.resource_filename(
        "exojax", "data/testdata/" + TESTDATA_refrind
    )
    path = pkg_resources.resource_filename("exojax", "data/testdata/")
    pdb_nh3 = pardb.PdbCloud(
        "test", download=False, refrind_path=refrind_path, path=path, nurange=nurange
    )
    return pdb_nh3


if __name__ == "__main__":
    mock_PdbPlouds()
