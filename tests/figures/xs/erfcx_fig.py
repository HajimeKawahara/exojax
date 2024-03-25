"""test for erfcx.

- This test compares hjert with scipy.erfcx, see Appendix in Paper I.
"""

import pytest
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from scipy.special import erfcx as sc_erfcx
from exojax.special import erfcx
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
#from jax import config                                                 #
#config.update("jax_enable_x64", True)

N = 10000
xv = jnp.logspace(-5, 5, N)
xvc = np.logspace(-5, 5, N)
verfcx = vmap(erfcx)
ref = sc_erfcx(xvc)


def test_comparison_erfcx_scipy():
    d = (verfcx(xv) - ref) / ref
    print('erfcx: MEDIAN=', np.median(d), ' MAX=',
          np.max(d), ' MEAN=', np.mean(d))

    fig = plt.figure(figsize=(7, 2.3))
    ax = fig.add_subplot(111)
    ax.plot(xvc, d, '.', alpha=0.1, rasterized=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.e'))
    plt.ylabel('(our erfcx - scipy)/scipy')
    plt.xscale('log')
    plt.xlabel('$x$')
    plt.ylim(-3.e-6, 3.e-6)
    plt.savefig('erfcx.png', bbox_inches='tight', pad_inches=0.0)
    plt.savefig('erfcx.pdf', bbox_inches='tight', pad_inches=0.0)

    assert np.max(d) < 2.e-6




if __name__ == '__main__':
    import time
    test_comparison_erfcx_scipy()

    if False:
        # comparison
        for j in range(0, 3):
            ts = time.time()
            for i in range(0, 10000):
                verfcx(xv)
            te = time.time()
            print('direct', te-ts)

