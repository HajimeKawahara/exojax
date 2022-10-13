import pytest
from exojax.spec.spin_rotation import rotkernel
import jax.numpy as jnp

def test_rotkernel(fig=False):
    N = 101
    x = jnp.linspace(-1.0, 1.0, N)
    u1 = 0.1
    u2 = 0.1
    kernel = rotkernel(x, u1, u2)
    assert jnp.sum(kernel) == 143.85559
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(x, kernel)
        plt.show()


if __name__ == "__main__":
    test_rotkernel(fig=True)