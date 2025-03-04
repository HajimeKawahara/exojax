from exojax.utils.photometry import apparent_magnitude_isothermal_sphere
from exojax.utils.constants import Rs, RJ
from exojax.test.emulate_filter import mock_filter
import pytest

def test_apparent_magnitude_isothermal_sphere_sun():
    #from jax import config
    #config.update("jax_enable_x64", True)
    _, f0, nu_ref, transmission_ref = mock_filter()

    mag = apparent_magnitude_isothermal_sphere(5772.0, Rs/RJ, 10.0, nu_ref, transmission_ref, f0)
    print(mag)
    assert mag == pytest.approx(5.33265, rel=1e-4)

if __name__ == "__main__":
    test_apparent_magnitude_isothermal_sphere_sun()