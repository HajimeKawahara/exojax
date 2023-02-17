import pytest
from exojax.spec.atmrt import ArtCommon
from exojax.spec.atmrt import ArtEmisPure
from exojax.test.emulate_mdb import mock_wavenumber_grid

def test_ArtCommon():
    nu_grid, wav, res = mock_wavenumber_grid()
    pressure_layer_params = [1.e2, 1.e-8, 100]
    art = ArtCommon(nu_grid,pressure_layer_params)
    assert art.k == pytest.approx(1.2618568830660204)

def test_ArtEmisPure():
    nu_grid, wav, res = mock_wavenumber_grid()
    pressure_layer_params = [1.e2, 1.e-8,  100]
    art = ArtEmisPure(nu_grid,pressure_layer_params)
    assert art.k == pytest.approx(1.2618568830660204)
    assert art.method == "emission_with_pure_absorption"

if __name__ == "__main__":
    test_ArtCommon()
    test_ArtEmisPure()