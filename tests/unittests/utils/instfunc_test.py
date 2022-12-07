import numpy as np
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.utils.instfunc import resolution_eslin
from exojax.utils.instfunc import resolution_eslog
import pytest


def test_resolution_to_gaussian_std():
    resolution = 10**5
    beta = resolution_to_gaussian_std(resolution)
    assert beta == pytest.approx(1.2731013507066515)

def test_resolution_eslin():
    nus = np.linspace(1000, 2000, 1000)
    ref = (999.0000000000146, 1500.0, 1998.000000000029)
    assert np.all(resolution_eslin(nus) == pytest.approx(ref))

def test_resolution_eslog():
    nus = np.linspace(3, 4, 10000)
    assert resolution_eslog(nus) == pytest.approx(34757.1189083253)

if __name__ == "__main__":
    test_resolution_to_gaussian_std()
    test_resolution_eslin()
    test_resolution_eslog()