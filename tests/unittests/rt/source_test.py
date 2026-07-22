"""Tests for source-function utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.rt.source import source_from_opacity_emissivity


def test_source_uses_finite_zero_opacity_convention():
    source = source_from_opacity_emissivity(
        jnp.array([2.0, 0.0, -2.0]),
        jnp.array([6.0, 5.0, 4.0]),
    )

    np.testing.assert_allclose(source, np.array([3.0, 0.0, -2.0]))


def test_source_inputs_must_have_same_shape():
    with pytest.raises(ValueError, match="same shape"):
        source_from_opacity_emissivity(jnp.ones((2, 1)), jnp.ones((2,)))
