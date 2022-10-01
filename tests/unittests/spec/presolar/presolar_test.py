import numpy as np
from exojax.signal.ola import _input_length
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec.presolar import optimal_mini_batch
from exojax.spec.presolar import lbd_olaform
from exojax.spec.presolar import _reshape_lbd
from exojax.spec.presolar import shapefilter_olaform

def _example_filter(N, filter_length):
    nu_grid, wav, resolution = gen_wavenumber_grid(3000.0,
                                                   5000.0,
                                                   N,
                                                   unit="cm-1",
                                                   xsmode="presolar")
    return filter_length, nu_grid


def _example_lbd():
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
    lbd, filter_length = _example_lbd()
    input_length = np.shape(lbd)[0]
    ndiv, div_length = optimal_mini_batch(input_length, filter_length)
    hat_lbd = lbd_olaform(lbd, ndiv, div_length, filter_length)
    assert np.shape(hat_lbd) == (3, 762048, 19, 10)


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
    lbd, filter_length = _example_lbd()
    input_length = np.shape(lbd)[0]
    ndiv, div_length = optimal_mini_batch(input_length, filter_length)
    rlbd = _reshape_lbd(lbd, ndiv, div_length)
    assert np.shape(rlbd) == (3, 712048, 19, 10)

def test_shapefilter_olaform():
    N=10
    shapefilter=np.ones((N,3,4))
    div_length=20
    hat_shapefilter = shapefilter_olaform(shapefilter, div_length)
    res = np.sum(hat_shapefilter[N:,:,:])
    assert res==0.0
    
if __name__ == "__main__":
    test_shapefilter_olaform()
    test_lbd_olaform_simple()
    test_lbd_olaform()
    test_optimal_mini_batch()
    test_reshape_lbd_simple()
    test_reshape_lbd()