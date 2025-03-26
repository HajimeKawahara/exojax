"""emulate mdb class for unittest
"""
from exojax.test.data import TESTDATA_refrind
from exojax.test.data import get_testdata_filename

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
    refrind_path = get_testdata_filename(TESTDATA_refrind)
    path = get_testdata_filename("")
    pdb_nh3 = pardb.PdbCloud(
        "test", download=False, refrind_path=refrind_path, path=path, nurange=nurange
    )
    return pdb_nh3


if __name__ == "__main__":
    mock_PdbPlouds()
