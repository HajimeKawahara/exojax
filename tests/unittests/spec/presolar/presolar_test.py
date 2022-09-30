import numpy as np
from exojax.signal.ola import _input_length
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec.presolar import optimal_mini_batch


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


from exojax.signal.ola import generate_padding_matrix


def reshape_lbd(lbd, ndiv, div_length, padding_value=-np.inf):
    input_length, n_broadening_k, n_E_h = np.shape(lbd)
    if ndiv * div_length < input_length:
        raise ValueError(
            "ndiv*div_length should be larger than input length = shape(lbd)[0]"
        )
    residual = ndiv * div_length - input_length
    padding_shape = (residual, n_broadening_k, n_E_h)
    padding_matrix = np.full(padding_shape, padding_value)
    rlbd = np.vstack((lbd, padding_matrix))
    return rlbd.reshape((ndiv, div_length,n_broadening_k, n_E_h))

def test_reshape_lbd_simple():
    lbd = _simple_example_lbd()
    input_length = np.shape(lbd)[0]
    ndiv, div_length = 3, 5
    rlbd = reshape_lbd(lbd, ndiv, div_length)
    residual = ndiv * div_length - input_length
    res = np.sum(1.0/rlbd[-1,div_length-residual:,:,:]) #summation of 1/padding elements
    assert res == 0.0


def test_reshape_lbd():
    lbd, filter_length = _example_lbd()
    input_length = np.shape(lbd)[0]
    ndiv, div_length = optimal_mini_batch(input_length, filter_length)
    rlbd = reshape_lbd(lbd, ndiv, div_length)
    assert np.shape(rlbd) == (3, 712048, 19, 10)
    

def generate_hat_lbd(lbd, filter_length):
    input_length = np.shape(lbd)[0]
    ndiv, div_length = optimal_mini_batch(input_length, filter_length)
    rlbd = reshape_lbd(lbd, ndiv, div_length)
    hat_lbd = generate_padding_matrix(-np.inf, rlbd, filter_length)
    return hat_lbd


def test_generate_hat_lbd():
    lbd = _example_lbd()
    #hat_lbd = generate_hat_lbd(reshaped_lbd, filter_length)


if __name__ == "__main__":
    test_optimal_mini_batch()
    #test_generate_hat_lbd()
    test_reshape_lbd_simple()
    test_reshape_lbd()