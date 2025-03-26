import numpy as np
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.utils.instfunc import resolution_eslin
from exojax.utils.instfunc import resolution_eslog
from exojax.utils.instfunc import nx_even_from_resolution_eslog
import pytest

from exojax.utils.grids import wavenumber_grid





def test_nx_from_resolution_eslog():
    nu0 = 4000.0
    nu1 = 4500.0
    resolution = 849010.2113833647
    Nx = nx_even_from_resolution_eslog(nu0, nu1, resolution)
    
    assert Nx == 100000


def test_resolution_to_gaussian_std():
    resolution = 10**5
    beta = resolution_to_gaussian_std(resolution)
    assert beta == pytest.approx(1.2731013507066515)


def test_resolution_eslin():
    nus = np.linspace(1000, 2000, 1000)
    ref = (999.0000000000146, 1500.0, 1998.000000000029)
    assert np.all(resolution_eslin(nus) == pytest.approx(ref))


def test_resolution_eslog():
    nu0 = 4000.0
    nu1 = 4500.0
    Nx = 100000
    nus = np.logspace(np.log10(nu0), np.log10(nu1), Nx)
    assert resolution_eslog(nus) == pytest.approx(849010.2113833647)


if __name__ == "__main__":
    test_resolution_to_gaussian_std()
    test_resolution_eslin()
    test_resolution_eslog()
    test_nx_from_resolution_eslog()