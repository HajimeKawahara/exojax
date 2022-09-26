import numpy as np

from exojax.signal.ola import optimal_div_length
from exojax.utils.instfunc import resolution_eslog
from exojax.spec.setrt import gen_wavenumber_grid


def reshape_line_density(lbd, nu_grid, filter_length):
    #spectral_resolution = resolution_eslog(nu_grid)
    input_length = len(nu_grid)
    opt_div_length = optimal_div_length(filter_length)
    n_block = int(input_length / opt_div_length) + 1
    
    return 

def test_reshape_line_density():

    filter_length = 25001
    N = 1000000
    nu_grid = gen_wavenumber_grid(22000.0,
                                  23000.0,
                                  N,
                                  unit="AA",
                                  xsmode="premodit")
    reshape_line_density(nu_grid, filter_length)


if __name__ == "__main__":
    test_reshape_line_density()
