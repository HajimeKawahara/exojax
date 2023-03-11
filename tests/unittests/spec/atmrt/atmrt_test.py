from exojax.spec.atmrt import ArtCommon
from exojax.spec.atmrt import ArtEmisPure
from exojax.spec.atmrt import ArtTransPure
from exojax.test.emulate_mdb import mock_wavenumber_grid

def test_ArtCommon():
    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtCommon(nu_grid, pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)

def test_ArtEmisPure():
    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisPure(nu_grid,pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)

#def test_ArtTransPure()
#    nu_grid, wav, res = mock_wavenumber_grid()
#    art = ArtTransPure(nu_grid,pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)

if __name__ == "__main__":
    test_ArtCommon()
    test_ArtEmisPure()
