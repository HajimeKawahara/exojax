"""LPF filter test module.

Notes:
    LPF filter is a filter of a line profile generated by LPF Direct method.
    This filter is used in MODIT/PreMODIT to convolve the line profile.
    Reading this tests, one may understand the relation between the open and closed lpffilter, and the analytical Voigt kernel in ditkernel.
    The dimension of the open filter is odd, and the dimension of the closed filter is even.

"""
import pytest
import jax.numpy as jnp
from jax import vmap
import numpy as np
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.spec.ditkernel import fold_voigt_kernel_logst
from exojax.spec.lpffilter import generate_closed_lpffilter
from exojax.spec.lpffilter import generate_open_lpffilter
from jax import config


def test_generate_closed_lpffilter():
    filter_length_oneside, nsigmaD, ngammaL = _test_values_lpffilter()
    shapefilter = generate_closed_lpffilter(filter_length_oneside, nsigmaD, ngammaL)
    ref = _refvalue_closed_lpffilter()
    assert jnp.allclose(shapefilter, ref)


def test_generate_open_lpffilter():
    filter_length_oneside, nsigmaD, ngammaL = _test_values_lpffilter()
    shapefilter = generate_open_lpffilter(filter_length_oneside, nsigmaD, ngammaL)
    ref = _refvalue_open_lpffilter()
    assert jnp.allclose(shapefilter, ref)


def _refvalue_open_lpffilter():
    openref = jnp.array(
        [
            0.00937768,
            0.01388492,
            0.02281364,
            0.04338582,
            0.09071519,
            0.16579562,
            0.20870925,
            0.16579562,
            0.09071519,
            0.04338582,
            0.02281364,
            0.01388492,
            0.00937768
        ]
    )
    return openref


def _refvalue_closed_lpffilter():
    openref = _refvalue_open_lpffilter()
    openref = openref[:-1]
    filter_length_oneside, nsigmaD, ngammaL = _test_values_lpffilter()
    closeref = jnp.concatenate(
        [openref[filter_length_oneside:], openref[:filter_length_oneside]]
    )
    return closeref


def _test_values_lpffilter():
    filter_length_oneside = 6
    nsigmaD = 1.0
    ngammaL = 1.0
    return filter_length_oneside, nsigmaD, ngammaL


def closed_lpffilter_agreement_with_fold_voigt_kernel_logst(figure=False):
    config.update("jax_enable_x64", True)
    nu_grid, wav, resolution = mock_wavenumber_grid()
    pmarray = np.ones(len(nu_grid) + 1)
    pmarray[1::2] = pmarray[1::2] * -1.0
    nsigmaD = 5.0
    ngammaL_grid = jnp.array([1.0, 10.0, 100.0])
    log_ngammaL_grid = jnp.log(ngammaL_grid)
    filter_length_oneside = len(nu_grid)
    vk = fold_voigt_kernel_logst(
        jnp.fft.rfftfreq(2 * filter_length_oneside, 1),
        nsigmaD,
        log_ngammaL_grid,
        filter_length_oneside,
        pmarray,
    )
    vkfilter = jnp.fft.irfft(vk, axis=0)
    vmap_generate_lpffilter = vmap(generate_closed_lpffilter, (None, None, 0), 0)
    lpffilter = vmap_generate_lpffilter(filter_length_oneside, nsigmaD, ngammaL_grid)
    diff_filter = np.abs(vkfilter.T - lpffilter)
    print(np.max(diff_filter))

    assert np.max(diff_filter) < 3.0e-9  # 2.7824980652901843e-09 Feb 5th 2025

    if figure:
        # ifft/fft error
        vkrecover = jnp.fft.rfft(vkfilter, axis=0)
        diff_fftrecover = np.abs(vk - vkrecover)
        _lpffilter_figure(
            ngammaL_grid, vkfilter, lpffilter, diff_filter, diff_fftrecover
        )

    return vk, vkfilter, lpffilter, diff_filter, ngammaL_grid


def _lpffilter_figure(ngammaL_grid, vkfilter, lpffilter, diff_filter, diff_fftrecover):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_title("lpffilter vs vk")
    for i in range(0, len(ngammaL_grid)):
        plt.plot(vkfilter[:, i], label="tilde vk", alpha=0.3, color="C" + str(i))
        plt.plot(
            lpffilter[i, :],
            label="lpffilter",
            ls="dashed",
            alpha=1.0,
            color="C" + str(i),
        )
        plt.plot(diff_filter[i, :], label="diff", alpha=0.3, color="C" + str(i))
    plt.yscale("log")
    plt.legend()
    ax = fig.add_subplot(212)
    ax.set_title("ifft/fft error")
    for i in range(0, len(ngammaL_grid)):
        plt.plot(diff_fftrecover[:, i], label="diff", alpha=0.3, color="C" + str(i))
        plt.legend()
    plt.savefig("tildevk.png")
    plt.show()

@pytest.mark.parametrize("i", [0, -1])
def test_lpffilter_aliasing_area(i, figure=False):
    from exojax.signal.ola import _fft_length
    config.update("jax_enable_x64", True)
    
    filter_length_oneside, nsigmaD, ngammaL = _test_values_lpffilter()

    # shape filter
    shapefilter = generate_open_lpffilter(filter_length_oneside, nsigmaD, ngammaL)
    filter_length = len(shapefilter)

    # signal setting
    signal_length = 14
    signal = np.zeros(signal_length)
    signal[i]= 1.0

    # OLA size setting
    fft_length = _fft_length(signal_length, filter_length)

    print("M (filter) =", filter_length)
    print("L (signal) =", signal_length)
    print("M+L-1 (fft) =", fft_length)

    # OLA zero-padding
    filter_zeropad = np.zeros(fft_length)
    filter_zeropad[:filter_length] = shapefilter
    signal_zeropad = np.zeros(fft_length)
    signal_zeropad[:signal_length] = signal
    # convolution
    fft_filter = jnp.fft.rfft(jnp.array(filter_zeropad))
    fft_signal = jnp.fft.rfft(jnp.array(signal_zeropad))
    conv = jnp.fft.irfft(fft_filter * fft_signal)
    ind = jnp.arange(0, fft_length)
    # no aliasing area
    edge = filter_length_oneside
    noaliase_area = conv[edge:-edge]
    ind_noaliase = ind[edge:-edge]
    
    if i==0:
        diff = noaliase_area[0:filter_length_oneside+1] - shapefilter[filter_length_oneside:]
    elif i==-1:
        diff = noaliase_area[-filter_length_oneside-1:] - shapefilter[0:filter_length_oneside+1]
    res = jnp.max(jnp.abs(diff))
    #print("max diff",res)
    assert res < 3.e-17 #2.7755575615628914e-17

    if figure:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(ind, conv, ".", label="convolution")
        plt.plot(ind_noaliase, noaliase_area)
        plt.savefig("aliasing_area.png")


if __name__ == "__main__":
    closed_lpffilter_agreement_with_fold_voigt_kernel_logst(figure=True)
    #test_generate_closed_lpffilter()
    #test_generate_open_lpffilter()
    #test_lpffilter_aliasing_area(0)
