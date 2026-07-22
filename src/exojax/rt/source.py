"""Source-function utilities for radiative transfer."""

import jax.numpy as jnp


def source_from_opacity_emissivity(absorption_coefficient, emissivity_pi):
    """Form a pi-scaled source with a finite zero-opacity convention.

    Both inputs must have the same shape. Elements with zero absorption are
    assigned a zero source.
    """
    absorption_coefficient = jnp.asarray(absorption_coefficient)
    emissivity_pi = jnp.asarray(emissivity_pi)
    if absorption_coefficient.shape != emissivity_pi.shape:
        raise ValueError(
            "absorption_coefficient and emissivity_pi must have the same shape."
        )
    nonzero = absorption_coefficient != 0.0
    denominator = jnp.where(nonzero, absorption_coefficient, 1.0)
    return jnp.where(nonzero, emissivity_pi / denominator, 0.0)
