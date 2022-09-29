import numpy as np

from exojax.signal.ola import optimal_div_length
from exojax.utils.instfunc import resolution_eslog
from exojax.spec.setrt import gen_wavenumber_grid


def mini_batches_number(nu_grid, filter_length):
    input_length = len(nu_grid)
    opt_div_length = optimal_div_length(filter_length)
    n_block = int(input_length / opt_div_length) + 1
    return n_block

def reshape_line_density(biased_lsd, nu_grid, filter_length):
    #spectral_resolution = resolution_eslog(nu_grid)
    n_mini_batches = mini_batches_number(nu_grid, filter_length)
    biased_lsd.reshape(, int(Nx / ndiv))
    return

def example_filter():
    filter_length = 50001
    N = 2000000
    nu_grid, wav, resolution = gen_wavenumber_grid(3000.0,
                                                   5000.0,
                                                   N,
                                                   unit="cm-1",
                                                   xsmode="presolar")
    return filter_length, nu_grid

def test_mini_batches_number():
    filter_length, nu_grid = example_filter()
    n_mini_batches = mini_batches_number(nu_grid, filter_length)
    assert n_mini_batches == 3


def test_reshape_line_density():
    filter_length, nu_grid = example_filter()
    n_E_h = 10
    n_broadening_k = 19
    lbd = np.zeros((nu_grid,n_broadening_k,n_E_h))
    reshaped_lbd = reshape_line_density(lbd, nu_grid, filter_length)
    
    
if __name__ == "__main__":
    test_mini_batches_number()
    test_reshape_line_density()