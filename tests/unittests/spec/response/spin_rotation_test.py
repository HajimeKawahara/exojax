import pytest
from exojax.spec.spin_rotation import rotkernel
import jax.numpy as jnp

def test_rotkernel(fig=False):
    N = 201
    x_1 = jnp.linspace(-2.0, 2.0, N)
    u1 = 0.1
    u2 = 0.1
    kernel_1 = rotkernel(x_1, u1, u2)
    N = 101
    x_2 = jnp.linspace(-1.0, 1.0, N)
    kernel_2 = rotkernel(x_2, u1, u2)
    assert jnp.sum(kernel_1) == pytest.approx(143.85559)
    assert jnp.sum(kernel_2) == pytest.approx(143.85559)
    
    if fig:
        import matplotlib.pyplot as plt
        plt.plot(x_1, kernel_1)
        plt.plot(x_2, kernel_2)
        plt.show()


if __name__ == "__main__":
    test_rotkernel(fig=True)