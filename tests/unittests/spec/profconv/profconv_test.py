from exojax.spec.profconv import calc_xsection_from_lsd_zeroscan
from exojax.spec.profconv import calc_xsection_from_lsd_scanfft
import jax.numpy as jnp

def test_basic_convolution_calc_xsection_from_lsd_zeroscan():

    Nsignal = 6
    Ngamma = 2
    Slsd = jnp.zeros((Nsignal, Ngamma))
    
    # these are used for log-linear conversuion, so assume identical convolution in this test
    R = 1.0
    nu_grid = jnp.ones(Nsignal)
    pmarray = 

    xsv = calc_xsection_from_lsd_zeroscan(
    Slsd, R, pmarray, nsigmaD, nu_grid, log_ngammaL_grid)
