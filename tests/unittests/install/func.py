import pytest
import jax.numpy as jnp
from jax.lax import scan


def test_cuDNN():
    # convolution requires cuDNN
    v = jnp.array([1, 9, 1])
    s = jnp.array([0, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1])
    c = jnp.convolve(s, v, mode='same')
    res = c-jnp.array([0.,  0.,  0.,  1., 11., 19.,  2.,  0.,
                      2., 18.,  2.,  0.,  0.,  0.,  1.,  9., ])
    assert jnp.sum(res**2) == 0.0


def test_vaex():
    import vaex


if __name__ == '__main__':
    test_cuDNN()
    test_vaex()
