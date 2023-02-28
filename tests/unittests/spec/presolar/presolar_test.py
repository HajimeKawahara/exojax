"""unit tests for presolar basic functions

    * See presolar_xsection_test.py for the cross section test
    * See presolar_spectrum_test.py for the spectrum test

"""

import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.spec.presolar import optimal_mini_batch
from exojax.spec.presolar import lbd_olaform
from exojax.spec.presolar import _reshape_lbd
from exojax.spec.presolar import shapefilter_olaform
from exojax.utils.constants import Tref_original

def _example_filter(N, filter_length):
    nu_grid, wav, resolution = wavenumber_grid(3000.0,
                                                   5000.0,
                                                   N,
                                                   unit="cm-1",
                                                   xsmode="presolar")
    return filter_length, nu_grid


def _example_lbd_and_filter():
    filter_length, nu_grid = _example_filter(2000000, 50001)
    n_E_h = 10
    n_broadening_k = 19
    lbd = np.zeros((len(nu_grid), n_broadening_k, n_E_h))
    return lbd, filter_length


def _simple_example_lbd():
    n_E_h = 2

    n_broadening_k = 3
    input_length = 13
    N = input_length * n_broadening_k * n_E_h
    lbd = np.array(list(range(N)))
    lbd = lbd.reshape((input_length, n_broadening_k, n_E_h))
    return lbd


def test_reshape_lbd_simple():
    lbd = _simple_example_lbd()
    input_length = np.shape(lbd)[0]
    ndiv, div_length = 3, 5
    rlbd = _reshape_lbd(lbd, ndiv, div_length)
    residual = ndiv * div_length - input_length
    res = np.sum(1.0 / rlbd[-1, div_length -
                            residual:, :, :])  #summation of 1/padding elements
    assert res == 0.0


def test_reshape_lbd():
    lbd, filter_length = _example_lbd_and_filter()
    input_length = np.shape(lbd)[0]
    ndiv, div_length = optimal_mini_batch(input_length, filter_length)
    rlbd = _reshape_lbd(lbd, ndiv, div_length)
    assert np.shape(rlbd) == (3, 712048, 19, 10)


def test_optimal_mini_batch():
    filter_length, nu_grid = _example_filter(2000000, 50001)
    ndiv, opt_div_length = optimal_mini_batch(len(nu_grid), filter_length)
    assert ndiv == 3
    assert opt_div_length == 712048
    assert len(nu_grid) < ndiv * opt_div_length


def test_lbd_olaform_simple():
    lbd = _simple_example_lbd()
    input_length = np.shape(lbd)[0]
    ndiv, div_length = 3, 5
    filter_length = 3
    hat_lbd = lbd_olaform(lbd, ndiv, div_length, filter_length)
    residual = ndiv * div_length - input_length
    res = np.sum(1.0 /
                 hat_lbd[-1, div_length -
                         residual:, :, :])  #summation of 1/padding elements
    assert res == 0.0
    margin = filter_length - 1
    res2 = np.sum(1.0 /
                  hat_lbd[:, -margin:, :, :])  #summation of 1/padding elements
    assert res2 == 0.0


def test_lbd_olaform():
    lbd, filter_length = _example_lbd_and_filter()
    input_length = np.shape(lbd)[0]
    ndiv, div_length = optimal_mini_batch(input_length, filter_length)
    hat_lbd = lbd_olaform(lbd, ndiv, div_length, filter_length)
    assert np.shape(hat_lbd) == (3, 762048, 19, 10)



import jax.numpy as jnp
from exojax.spec.premodit import logf_bias, g_bias


from exojax.spec.premodit import unbiased_lsd_zeroth
from jax import vmap 
def vmap_unbiased_lsd(hat_lbd, T, nu_grid, elower_grid, qt):
    """ unbias the biased LSD

    Args:
        lbd_biased: log biased hat LSD
        T: temperature for unbiasing in Kelvin
        nu_grid: wavenumber grid in cm-1
        elower_grid: Elower grid in cm-1
        qt: partition function ratio Q(T)/Q(Tref)

    Returns:
        LSD, shape = (number_of_wavenumber_bin, number_of_broadening_parameters)
        
    """
    vmapped_unbiased_lsd = vmap(unbiased_lsd_zeroth,(0,None,None,None,None,None),0)
    return vmapped_unbiased_lsd(hat_lbd, T, Tref_original, nu_grid, elower_grid, qt)
    

def test_unbiased_lsd_simple():
    lbd = _simple_example_lbd()
    input_length, n_broadening_grid, n_elower_grid = np.shape(lbd)
    ndiv, div_length = 3, 5
    filter_length = 3
    nu_grid = np.array(range(input_length))
    hat_lbd = lbd_olaform(lbd, ndiv, div_length, filter_length)
    T = 1000.0  
    elower_grid = np.ones(n_elower_grid)
    qt = 1.0
    lsd = unbiased_lsd_zeroth(lbd, T, Tref_original, nu_grid, elower_grid, qt)
    assert np.all(np.shape(lsd) == (13,3))

def test_shapefilter_olaform():
    N = 10
    shapefilter = np.ones((N, 3))
    div_length = 20
    hat_shapefilter = shapefilter_olaform(shapefilter, div_length)
    res = np.sum(hat_shapefilter[N:, :])
    assert res == 0.0


if __name__ == "__main__":
    #test_reshape_lbd()
    #test_reshape_lbd_simple()
    #test_optimal_mini_batch()
    #test_lbd_olaform_simple()
    #test_lbd_olaform()
    test_unbiased_lsd_simple()
    
    #test_vmap_unbiased_lsd_simple()
    
    #test_shapefilter_olaform()