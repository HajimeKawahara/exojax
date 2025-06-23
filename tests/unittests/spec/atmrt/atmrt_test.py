from exojax.rt.common import ArtCommon
from exojax.rt import ArtEmisPure
from exojax.rt import ArtTransPure
from exojax.test.emulate_mdb import mock_wavenumber_grid

def test_ArtCommon():
    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtCommon(pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100, nu_grid=nu_grid)

def test_ArtEmisPure():
    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisPure(pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100, nu_grid=nu_grid)

def test_ArtTransPure():
    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtTransPure(pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)

if __name__ == "__main__":
    test_ArtCommon()
    test_ArtEmisPure()
    test_ArtTransPure()