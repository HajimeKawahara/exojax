from exojax.database.hitran.api import MdbHitran
from exojax.utils.grids import wavenumber_grid
import pytest


def test_Hitran_nonair():
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(lambda0,
                                    lambda1,
                                    100000,
                                    unit='AA',
                                    xsmode="modit")
    mdb = MdbHitran("CO",nus, nonair_broadening=True)
    print(mdb.n_h2)

def test_Hitemp():
    nus,wav,res=wavenumber_grid(23000.,23010.0,100000,unit='AA',xsmode="premodit")
    mdb = MdbHitemp("CO",nus)

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

def test_Hitemp_with_error():
    lambda0 = 22920.0
    lambda1 = 23100.0
    nus, wav, res = wavenumber_grid(lambda0,
                                    lambda1,
                                    100000,
                                    unit='AA',
                                    xsmode="premodit")
    mdb = MdbHitemp("CO",nus, with_error=True)


if __name__ == "__main__":
    #test_Hitemp()
    test_Hitran_nonair()
    #test_Hitran()
    #test_noline_Hitran()