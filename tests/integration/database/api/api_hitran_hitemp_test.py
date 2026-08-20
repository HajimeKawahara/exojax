from shutil import copyfile

from exojax.database.hitran.api import MdbHitran
from exojax.database.hitemp.api import MdbHitemp
from exojax.test.data import TESTDATA_CO_HITEMP_PARFILE, get_testdata_filename
from exojax.utils.grids import wavenumber_grid
import pytest


@pytest.fixture
def hitemp_parfile(tmp_path):
    target = tmp_path / "CO_HITEMP_SAMPLE.par"
    copyfile(get_testdata_filename(TESTDATA_CO_HITEMP_PARFILE), target)
    return target


def test_Hitran_nonair():
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(lambda0,
                                    lambda1,
                                    100000,
                                    unit='AA',
                                    xsmode="modit")
    mdb = MdbHitran("CO",nus, nonair_broadening=True)
    
def test_Hitemp(hitemp_parfile):
    nus,wav,res=wavenumber_grid(23000.,23010.0,100000,unit='AA',xsmode="premodit")
    mdb = MdbHitemp("CO", nus, parfile=hitemp_parfile)
    assert not hasattr(mdb, "ierr")

def test_Hitran():
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(lambda0,
                                    lambda1,
                                    100000,
                                    unit='AA',
                                    xsmode="modit")
    mdb = MdbHitran("CO",nus)



def test_noline_Hitran():
    nus,wav,res=wavenumber_grid(6910,6990,100000,unit='cm-1',xsmode="premodit")
    with pytest.raises(ValueError):
        mdb = MdbHitran("CO",nus)

def test_Hitran_with_error():
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(lambda0,
                                    lambda1,
                                    100000,
                                    unit='AA',
                                    xsmode="premodit")
    mdb = MdbHitran("CO",nus, with_error=True)

def test_Hitemp_with_error(hitemp_parfile):
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(lambda0,
                                    lambda1,
                                    100000,
                                    unit='AA',
                                    xsmode="premodit")
    mdb = MdbHitemp("CO", nus, with_error=True, parfile=hitemp_parfile)
    assert len(mdb.ierr) == len(mdb.nu_lines) > 0
    assert mdb.ierr.dtype.kind == "i"


if __name__ == "__main__":
    #test_Hitemp()
    #test_Hitran_nonair()
    #test_Hitran()
    test_Hitran_with_error()
    #test_Hitemp_with_error()
    
    #test_noline_Hitran()
