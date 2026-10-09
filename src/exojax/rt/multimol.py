"""Combine named molecular line opacities on a common wavenumber grid."""

from collections.abc import Mapping

import jax.numpy as jnp

from exojax.opacity.multimol import validate_opacity_grids
from exojax.rt.layeropacity import layer_optical_depth


def _layer_profile(value, name, number_of_layers, *, allow_scalar=False):
    """Validate a dynamic input using only its static array shape."""
    value = jnp.asarray(value)
    if allow_scalar and value.ndim == 0:
        return value
    if value.shape != (number_of_layers,):
        expected = f"({number_of_layers},)"
        if allow_scalar:
            expected = f"a scalar or {expected}"
        raise ValueError(f"{name} must have shape {expected}; got {value.shape}.")
    return value


def layer_optical_depth_multi(
    opas, temperature, pressure, dpressure, *, mmr, gravity
):
    """Sum molecular line optical depths using named mass mixing ratios.

    Prepare the opacity calculators before calling this function. To use
    ``jax.jit`` or automatic differentiation, capture ``opas`` in a closure;
    the calculators, their names, and their grids are fixed configuration.
    Temperature, pressure, pressure intervals, mass mixing ratios, and gravity
    remain dynamic JAX inputs. The calculators retain their existing broadening
    assumptions: changing an MMR does not change the broadening gas mixture.

    Calculator grids and metadata are validated on the host, including during
    tracing. Dynamic inputs are checked by shape only. Species contributions
    are added without stacking, clipping, abundance normalization, or skipping
    zero abundances, so absorption derivatives remain available at zero MMR.

    Args:
        opas: Nonempty mapping from species names to prepared line-by-line
            calculators with ``xsmatrix(T, P)``, ``nu_grid``, and ``molmass``.
            The calculator must accept dynamic temperature and pressure, as
            OpaPremodit does. Fixed-pressure OpaDiffgrid is not supported.
            All calculators must use the same wavenumber grid. CKD calculators
            require a separate mixing method and are not supported here.
        temperature: Temperature in K, shape ``(Nlayer,)``.
        pressure: Pressure in bar, shape ``(Nlayer,)``.
        dpressure: Layer pressure intervals in bar, shape ``(Nlayer,)``.
        mmr: Mapping with exactly the same keys as ``opas``. Each mass mixing
            ratio is a scalar or has shape ``(Nlayer,)``. Molecular masses are
            read from the calculators; these inputs are not volume fractions.
        gravity: Gravity in cm/s2, a scalar or shape ``(Nlayer,)``.

    Returns:
        Dimensionless optical depth with shape ``(Nlayer, Nnu)``.
    """
    validate_opacity_grids(opas)
    if not isinstance(mmr, Mapping):
        raise TypeError("mmr must be a mapping from species names to mass mixing ratios.")
    missing = [name for name in opas if name not in mmr]
    extra = [name for name in mmr if name not in opas]
    if missing or extra:
        raise ValueError(
            "mmr keys must exactly match opas; "
            f"missing species: {missing}; extra species: {extra}."
        )

    temperature = jnp.asarray(temperature)
    if temperature.ndim != 1 or temperature.size == 0:
        raise ValueError("temperature must be a nonempty one-dimensional array.")
    number_of_layers = temperature.shape[0]
    pressure = _layer_profile(pressure, "pressure", number_of_layers)
    dpressure = _layer_profile(dpressure, "dpressure", number_of_layers)
    gravity = _layer_profile(
        gravity, "gravity", number_of_layers, allow_scalar=True
    )
    number_of_wavenumbers = len(next(iter(opas.values())).nu_grid)
    expected_shape = (number_of_layers, number_of_wavenumbers)

    total = 0.0
    for name, opa in opas.items():
        abundance = _layer_profile(
            mmr[name], f"mmr[{name!r}]", number_of_layers, allow_scalar=True
        )
        cross_section = jnp.asarray(opa.xsmatrix(temperature, pressure))
        if cross_section.shape != expected_shape:
            raise ValueError(
                f"xsmatrix for {name!r} must have shape {expected_shape}; "
                f"got {cross_section.shape}."
            )
        total = total + layer_optical_depth(
            dpressure, cross_section, abundance, opa.molmass, gravity
        )
    return total
