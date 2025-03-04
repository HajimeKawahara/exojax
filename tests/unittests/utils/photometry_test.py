from exojax.utils.photometry import apparent_magnitude_isothermal_sphere
from exojax.utils.constants import Rs, RJ
import pytest
import pandas as pd

def test_apparent_magnitude_isothermal_sphere_sun(download=False):
    #from jax import config
    #config.update("jax_enable_x64", True)

    if download:
        from exojax.utils.photometry import download_filter_from_svo
        from exojax.utils.photometry import download_zero_magnitude_flux_from_svo
        filter_name = "SLOAN/SDSS.g"
        nu_ref, transmission_ref = download_filter_from_svo(filter_name)
        pd.DataFrame({"nu": nu_ref, "transmission": transmission_ref}).to_csv("filter_sdss_g.csv")
        _, f0 = download_zero_magnitude_flux_from_svo(filter_name, unit="cm-1")
    else:
        dat = pd.read_csv("filter_sdss_g.csv")
        nu_ref = dat["nu"].values
        transmission_ref = dat["transmission"].values
        f0 = 1.20623691665283e-09
        
    mag = apparent_magnitude_isothermal_sphere(5772.0, Rs/RJ, 10.0, nu_ref, transmission_ref, f0)
    
    assert mag == pytest.approx(5.33265, rel=1e-4)

if __name__ == "__main__":
    test_apparent_magnitude_isothermal_sphere_sun()