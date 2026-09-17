"""Atmospheric profile function."""

from functools import partial

from exojax.utils.constants import G, bar_cgs, kB, m_u
import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from jax import jit


def pressure_layer_logspace(
    log_pressure_top=-8.0,
    log_pressure_btm=2.0,
    nlayer=20,
    mode="ascending",
    reference_point=0.5,
    numpy=False,
):
    """Pressure layer evenly spaced in logspace, i.e. logP interval is constant

    Args:
        log_pressure_top: log10(P[bar]) at the top layer
        log_pressure_btm: log10(P[bar]) at the bottom layer
        nlayer: the number of the layers
        mode: ascending or descending
        reference_point (float): reference point in a layer (0-1). Center:0.5, lower boundary:1.0, upper boundary:0
        numpy: if True use numpy array instead of jnp array

    Returns:
        pressures: representative pressures (array) of the layers
        delta_pressures: delta pressure layer, the old name is dParr
        pressure_decrease_rate: pressure decrease rate of the layer (k-factor; k < 1) pressure[i-1] = pressure_decrease_rate*pressure[i]

    Note:
        d logP is constant using this function.
    """
    dlogP = (log_pressure_btm - log_pressure_top) / (nlayer - 1)
    if numpy:
        pressures = np.logspace(log_pressure_top, log_pressure_btm, nlayer)
    else:
        pressures = jnp.logspace(log_pressure_top, log_pressure_btm, nlayer)

    k = 10**-dlogP
    delta_pressures = (k ** (reference_point - 1.0) - k**reference_point) * pressures

    if mode == "descending":
        pressures = pressures[::-1]
        delta_pressures = delta_pressures[::-1]

    return pressures, delta_pressures, k


def pressure_layer_logspace_from_boundaries(
    log_pressure_top_boundary,
    log_pressure_btm_boundary,
    nlayer,
    reference_point=0.5,
    numpy=False,
):
    """Create a log-spaced pressure grid from exact layer boundaries.

    Args:
        log_pressure_top_boundary (float): Log10 pressure in bar at the top
            boundary.
        log_pressure_btm_boundary (float): Log10 pressure in bar at the bottom
            boundary.
        nlayer (int): Number of atmospheric layers.
        reference_point (float): Fractional position of the representative
            pressure within each layer in log pressure. The upper boundary is
            0, the geometric center is 0.5, and the lower boundary is 1.
        numpy (bool): If True, use NumPy arrays instead of JAX arrays.

    Returns:
        tuple: Representative pressures, layer pressure differences, pressure
            decrease rate, and the ``nlayer + 1`` pressure boundaries. The
            pressure boundaries are the source of truth for the other arrays.

    Notes:
        Backend representability is validated during eager execution and is
        skipped while JAX is tracing. Build the grid eagerly when validation
        is required. Valid traced inputs remain compatible with JIT
        compilation and automatic differentiation.
    """
    if not isinstance(nlayer, (int, np.integer)) or isinstance(nlayer, bool):
        raise ValueError("Number of layers must be a positive integer.")
    if nlayer < 1:
        raise ValueError("Number of layers must be a positive integer.")
    for name, value in (
        ("Top boundary log pressure", log_pressure_top_boundary),
        ("Bottom boundary log pressure", log_pressure_btm_boundary),
        ("Reference point", reference_point),
    ):
        ndim = value.ndim if hasattr(value, "ndim") else np.ndim(value)
        if ndim != 0:
            raise ValueError(f"{name} must be scalar.")
    top_boundary_is_traced = isinstance(
        log_pressure_top_boundary, jax.core.Tracer
    )
    bottom_boundary_is_traced = isinstance(
        log_pressure_btm_boundary, jax.core.Tracer
    )
    reference_point_is_traced = isinstance(reference_point, jax.core.Tracer)
    if numpy and (
        top_boundary_is_traced
        or bottom_boundary_is_traced
        or reference_point_is_traced
    ):
        raise ValueError("The NumPy backend does not support traced inputs.")
    if not reference_point_is_traced:
        if not 0.0 <= reference_point <= 1.0:
            raise ValueError("Reference point must be between 0 and 1.")
    if not top_boundary_is_traced and not np.isfinite(
        log_pressure_top_boundary
    ):
        raise ValueError("Pressure boundary logs must be finite.")
    if not bottom_boundary_is_traced and not np.isfinite(
        log_pressure_btm_boundary
    ):
        raise ValueError("Pressure boundary logs must be finite.")
    if not top_boundary_is_traced and not bottom_boundary_is_traced:
        if log_pressure_btm_boundary <= log_pressure_top_boundary:
            raise ValueError(
                "Bottom boundary pressure must be greater than the top "
                "boundary pressure."
            )

    array_module = np if numpy else jnp
    if numpy:
        with np.errstate(over="ignore", under="ignore"):
            pressure_boundaries = np.logspace(
                log_pressure_top_boundary,
                log_pressure_btm_boundary,
                nlayer + 1,
            )
    else:
        pressure_boundaries = jnp.logspace(
            log_pressure_top_boundary,
            log_pressure_btm_boundary,
            nlayer + 1,
        )
    try:
        with np.errstate(over="ignore", under="ignore"):
            pressure_endpoints = array_module.asarray(
                [
                    10.0**log_pressure_top_boundary,
                    10.0**log_pressure_btm_boundary,
                ],
                dtype=pressure_boundaries.dtype,
            )
    except OverflowError as err:
        raise ValueError(
            "Pressure boundaries cannot be represented by the array backend."
        ) from err
    if numpy:
        pressure_boundaries[[0, -1]] = pressure_endpoints
    else:
        pressure_boundaries = pressure_boundaries.at[0].set(
            pressure_endpoints[0]
        )
        pressure_boundaries = pressure_boundaries.at[-1].set(
            pressure_endpoints[1]
        )
    if not isinstance(pressure_boundaries, jax.core.Tracer):
        pressure_boundaries_host = np.asarray(pressure_boundaries)
        smallest_normal = jnp.finfo(pressure_boundaries.dtype).tiny
        if not np.all(np.isfinite(pressure_boundaries_host)) or np.any(
            pressure_boundaries_host < smallest_normal
        ):
            raise ValueError(
                "Pressure boundaries cannot be represented by the active "
                "array dtype."
            )
        if np.any(np.diff(pressure_boundaries_host) <= 0.0):
            raise ValueError(
                "Pressure layers are not distinct in the active array dtype."
            )
    pressure_upper = pressure_boundaries[:-1]
    pressure_lower = pressure_boundaries[1:]
    pressures = (
        pressure_upper ** (1.0 - reference_point)
        * pressure_lower**reference_point
    )
    delta_pressures = array_module.diff(pressure_boundaries)
    dlogP = (log_pressure_btm_boundary - log_pressure_top_boundary) / nlayer
    pressure_decrease_rate = array_module.asarray(
        10.0**-dlogP, dtype=pressure_boundaries.dtype
    )
    smallest_normal = jnp.finfo(pressure_boundaries.dtype).tiny
    if not isinstance(pressures, jax.core.Tracer):
        pressures_host = np.asarray(pressures)
        if not np.all(np.isfinite(pressures_host)) or np.any(
            pressures_host < smallest_normal
        ):
            raise ValueError(
                "Pressure grid cannot be represented by the active array dtype."
            )
    if not isinstance(delta_pressures, jax.core.Tracer):
        delta_pressures_host = np.asarray(delta_pressures)
        if not np.all(np.isfinite(delta_pressures_host)) or np.any(
            delta_pressures_host < smallest_normal
        ):
            raise ValueError(
                "Pressure grid cannot be represented by the active array dtype."
            )
    if not isinstance(pressure_decrease_rate, jax.core.Tracer):
        pressure_decrease_rate_host = np.asarray(pressure_decrease_rate)
        if (
            not np.isfinite(pressure_decrease_rate_host)
            or pressure_decrease_rate_host < smallest_normal
            or pressure_decrease_rate_host >= 1.0
        ):
            raise ValueError(
                "Pressure grid cannot be represented by the active array dtype."
            )

    return (
        pressures,
        delta_pressures,
        pressure_decrease_rate,
        pressure_boundaries,
    )


def pressure_upper_logspace(pressures, pressure_decrease_rate, reference_point=0.5):
    """computes pressure at the upper point of the layers

    Args:
        pressures (_type_): representative pressure (output of pressure_layer_logspace)
        pressure_decrease_rate: pressure decrease rate of the layer (k-factor; k < 1) pressure[i-1] = pressure_decrease_rate*pressure[i]
        reference_point (float): reference point in a layer (0-1). Center:0.5, lower boundary:1.0, upper boundary:0

    Returns:
        _type_: pressure at the upper point (\overline{P}_i)
    """
    return (pressure_decrease_rate**reference_point) * pressures


def pressure_lower_logspace(pressures, pressure_decrease_rate, reference_point=0.5):
    """computes pressure at the lower point of the layers

    Args:
        pressures (_type_): representative pressure (output of pressure_layer_logspace)
        pressure_decrease_rate: pressure decrease rate of the layer (k-factor; k < 1) pressure[i-1] = pressure_decrease_rate*pressure[i]
        reference_point (float): reference point in a layer (0-1). Center:0.5, lower boundary:1.0, upper boundary:0

    Returns:
        _type_: pressure at the lower point (underline{P}_i)
    """
    return (pressure_decrease_rate ** (reference_point - 1.0)) * pressures


def pressure_boundary_logspace(
    pressures, pressure_decrease_rate, reference_point=0.5, numpy=False
):
    """computes pressure at the boundary of the layers (Nlayer + 1)

    Args:
        pressures (_type_): representative pressure (output of pressure_layer_logspace)
        pressure_decrease_rate: pressure decrease rate of the layer (k-factor; k < 1) pressure[i-1] = pressure_decrease_rate*pressure[i]
        reference_point (float): reference point in a layer (0-1). Center:0.5, lower boundary:1.0, upper boundary:0
        numpy: if True use numpy array instead of jnp array

    Returns:
        _type_: pressure at the boundary (Nlayer + 1)
    """
    pressure_bottom_boundary = (
        pressure_decrease_rate ** (reference_point - 1.0)
    ) * pressures[-1]
    pressure_upper = pressure_upper_logspace(
        pressures, pressure_decrease_rate, reference_point
    )
    if numpy:
        return np.append(pressure_upper, pressure_bottom_boundary)
    else:
        return jnp.append(pressure_upper, pressure_bottom_boundary)


@jit
def hydrostatic_radius_profile(
    pressure_boundaries,
    mass_density_layers,
    planet_mass,
    radius_bottom,
):
    """Compute radius and gravity at pressure boundaries.

    This function integrates hydrostatic equilibrium from the bottom boundary
    upward, neglecting atmospheric mass and treating density as constant in
    each layer.

    Args:
        pressure_boundaries (1D array): pressure boundaries in bar, ordered
            from atmospheric top to bottom, with shape (Nlayer + 1,)
        mass_density_layers (1D array): layer mass densities in g/cm3, ordered
            from atmospheric top to bottom, with shape (Nlayer,)
        planet_mass (float): planet mass in g
        radius_bottom (float): radius in cm at pressure_boundaries[-1]

    Returns:
        tuple: radius boundaries in cm and gravity boundaries in cm/s2, both
            with shape (Nlayer + 1,)
    """
    delta_pressure_layers = jnp.diff(pressure_boundaries) * bar_cgs
    gravity_bottom = G * planet_mass / radius_bottom / radius_bottom

    def integrate_layer(normalized_inverse_radius_lower, layer):
        delta_pressure_layer, mass_density_layer = layer
        normalized_inverse_radius_upper = (
            normalized_inverse_radius_lower
            - delta_pressure_layer
            / mass_density_layer
            / gravity_bottom
            / radius_bottom
        )
        return normalized_inverse_radius_upper, normalized_inverse_radius_upper

    normalized_inverse_radius_bottom = jnp.ones_like(radius_bottom)
    _, normalized_inverse_radius_upper = scan(
        integrate_layer,
        normalized_inverse_radius_bottom,
        (delta_pressure_layers, mass_density_layers),
        reverse=True,
    )
    normalized_inverse_radius_boundaries = jnp.append(
        normalized_inverse_radius_upper, normalized_inverse_radius_bottom
    )
    radius_boundaries = radius_bottom / normalized_inverse_radius_boundaries
    gravity_boundaries = gravity_bottom * normalized_inverse_radius_boundaries**2
    return radius_boundaries, gravity_boundaries


@partial(jit, static_argnames=("hydrostatic_scheme",))
def hydrostatic_radius_profile_ideal_gas(
    pressure_boundaries,
    temperature,
    mean_molecular_weight,
    radius_bottom,
    gravity_bottom,
    hydrostatic_scheme="variable_gravity",
):
    """Compute ideal-gas radius and gravity at pressure boundaries.

    The atmosphere is integrated upward from the bottom boundary using the
    hydrostatic equation and the ideal-gas pressure scale height. Atmospheric
    mass is neglected. The pressure grid may have nonuniform log-pressure
    spacing.

    Args:
        pressure_boundaries (1D array): Pressure boundaries in bar, ordered
            from atmospheric top to bottom, with shape ``(Nlayer + 1,)``.
        temperature (1D array): Layer temperatures in K, ordered from
            atmospheric top to bottom, with shape ``(Nlayer,)``.
        mean_molecular_weight (float or 1D array): Mean molecular weight in
            atomic mass units, either scalar or with shape ``(Nlayer,)``.
        radius_bottom (float): Radius in cm at
            ``pressure_boundaries[-1]``.
        gravity_bottom (float): Gravity in cm/s2 at
            ``pressure_boundaries[-1]``.
        hydrostatic_scheme (str): Hydrostatic discretization.
            ``"variable_gravity"`` analytically accounts for inverse-square
            gravity within each layer. ``"layer_constant_gravity"`` holds
            gravity fixed at the lower boundary of each layer.

    Returns:
        tuple: Radius boundaries in cm and gravity boundaries in cm/s2, both
            with shape ``(Nlayer + 1,)`` and ordered from atmospheric top to
            bottom.
    """
    if hydrostatic_scheme not in (
        "variable_gravity",
        "layer_constant_gravity",
    ):
        raise ValueError(
            "Unknown hydrostatic scheme. Choose 'variable_gravity' or "
            "'layer_constant_gravity'."
        )

    pressure_boundaries = jnp.asarray(pressure_boundaries)
    temperature = jnp.asarray(temperature)
    mean_molecular_weight = jnp.asarray(mean_molecular_weight)
    radius_bottom = jnp.asarray(radius_bottom)
    gravity_bottom = jnp.asarray(gravity_bottom)

    if pressure_boundaries.ndim != 1 or pressure_boundaries.shape[0] < 2:
        raise ValueError(
            "pressure_boundaries must be one-dimensional with at least two "
            "elements."
        )
    nlayer = pressure_boundaries.shape[0] - 1
    if temperature.ndim != 1 or temperature.shape[0] != nlayer:
        raise ValueError("temperature must have shape (Nlayer,).")
    if mean_molecular_weight.ndim not in (0, 1) or (
        mean_molecular_weight.ndim == 1
        and mean_molecular_weight.shape[0] != nlayer
    ):
        raise ValueError(
            "mean_molecular_weight must be scalar or have shape (Nlayer,)."
        )
    if radius_bottom.ndim != 0:
        raise ValueError("radius_bottom must be scalar.")
    if gravity_bottom.ndim != 0:
        raise ValueError("gravity_bottom must be scalar.")

    dtype = jnp.result_type(
        pressure_boundaries,
        temperature,
        mean_molecular_weight,
        radius_bottom,
        gravity_bottom,
        1.0,
    )
    pressure_boundaries = pressure_boundaries.astype(dtype)
    temperature = temperature.astype(dtype)
    mean_molecular_weight = jnp.broadcast_to(
        mean_molecular_weight.astype(dtype), temperature.shape
    )
    radius_bottom = radius_bottom.astype(dtype)
    gravity_bottom = gravity_bottom.astype(dtype)
    log_pressure_ratio = jnp.diff(jnp.log(pressure_boundaries))

    def integrate_layer(radius_lower, layer):
        temperature_layer, mean_molecular_weight_layer, log_pressure_layer = (
            layer
        )
        gravity_lower = gravity_bottom * (radius_bottom / radius_lower) ** 2
        scale_height_lower = pressure_scale_height(
            gravity_lower,
            temperature_layer,
            mean_molecular_weight_layer,
        )
        if hydrostatic_scheme == "variable_gravity":
            radius_upper = radius_lower / (
                1.0
                - scale_height_lower * log_pressure_layer / radius_lower
            )
        else:
            radius_upper = (
                radius_lower + scale_height_lower * log_pressure_layer
            )
        return radius_upper, radius_upper

    _, radius_upper = scan(
        integrate_layer,
        radius_bottom,
        (temperature, mean_molecular_weight, log_pressure_ratio),
        reverse=True,
    )
    radius_boundaries = jnp.append(radius_upper, radius_bottom)
    gravity_boundaries = gravity_bottom * (
        radius_bottom / radius_boundaries
    ) ** 2
    return radius_boundaries, gravity_boundaries


@jit
def normalized_layer_height(
    temperature, pressure_decrease_rate, mean_molecular_weight, radius_btm, gravity_btm
):
    """compute normalized height/radius at the upper boundary of the atmospheric layer, neglecting atmospheric mass, examining non-constant gravity.

    Note:
        This method computes the height of the atmospheric layers taking the effect of the decrease of gravity (i.e. $ \propto 1/r^2 $) into account.

    Args:
        temperature (1D array): temperature profile (K) of the layer, (Nlayer, from atmospheric top to bottom)
        pressure_decrease_rate:  pressure decrease rate of the layer (k-factor; k < 1) pressure[i-1] = pressure_decrease_rate*pressure[i]
        mean_molecular_weight (1D array): mean molecular weight profile, (Nlayer, from atmospheric top to bottom)
        radius_btm (float): radius (cm) at the lower boundary of the bottom layer, R0 or r_N
        gravity_btm (float): gravity (cm/s2) at the lower boundary of the bottom layer, g_N

    Returns:
        1D array (Nlayer) : layer height normalized by radius_btm starting from top atmosphere
        1D array (Nlayer) : radius at lower bondary normalized by radius_btm starting from top atmosphere
    """
    inverse_Tarr = temperature[::-1]
    inverse_mmr_arr = mean_molecular_weight[::-1]
    stacked_profiles = jnp.vstack([inverse_Tarr, inverse_mmr_arr]).T

    def compute_radius(normalized_radius_lower, arr):
        T_layer, mmw_layer = arr
        gravity_lower = gravity_btm / normalized_radius_lower**2
        Hn_lower = pressure_scale_height(gravity_lower, T_layer, mmw_layer) / radius_btm
        a = 1.0 + Hn_lower / normalized_radius_lower * jnp.log(pressure_decrease_rate)
        fac = 1.0 / a - 1.0
        normalized_height_layer = fac * normalized_radius_lower
        carry = normalized_radius_lower + normalized_height_layer
        return carry, [normalized_height_layer, normalized_radius_lower]

    _, results = scan(compute_radius, 1.0, stacked_profiles)
    normalized_height = results[0][::-1]
    normalized_radius_lower = results[1][::-1]
    return normalized_height, normalized_radius_lower


def gh_product(temperature, mean_molecular_weight):
    """product of gravity and pressure scale height

    Args:
        temperature: isothermal temperature (K)
        mean_molecular_weight: mean molecular weight

    Returns:
        gravity x pressure scale height cm2/s2
    """
    return (
        kB * temperature / m_u / mean_molecular_weight
    )  # Apply mmw (jnp array) last to minimize rounding errors in 32bit mode.


def pressure_scale_height(gravity, T, mean_molecular_weight):
    """pressure scale height assuming an isothermal atmosphere.

    Args:
        gravity: gravity acceleration (cm/s2)
        T: isothermal temperature (K)
        mean_molecular_weight: mean molecular weight

    Returns:
        pressure scale height (cm)
    """

    return gh_product(T, mean_molecular_weight) / gravity


def atmprof_powerlow(pressures, T0, alpha):
    """powerlaw temperature profile

    Args:
        pressures: pressure array (bar)
        T0 (float): T at P=1 bar in K
        alpha (float): powerlaw index

    Returns:
        array: temperature profile
    """
    return T0 * pressures**alpha


def atmprof_gray(pressures, gravity, kappa, Tint):
    """

    Args:
        pressures (1D array): pressure array (bar)
        gravity (float): gravity (cm/s2)
        kappa: infrared opacity
        Tint: temperature equivalence of the intrinsic energy flow

    Returns:
        array: temperature profile

    """

    tau = pressures * 1.0e6 * kappa / gravity
    Tarr = (0.75 * Tint**4 * (2.0 / 3.0 + tau)) ** 0.25
    return Tarr


def atmprof_Guillot(pressures, gravity, kappa, gamma, Tint, Tirr, f=0.25):
    """

    Notes:
        Guillot (2010) Equation (29)

    Args:
        pressures: pressure array (bar)
        gravity: gravity (cm/s2)
        kappa: thermal/IR opacity (kappa_th in Guillot 2010)
        gamma: ratio of optical and IR opacity (kappa_v/kappa_th), gamma > 1 means thermal inversion
        Tint: temperature equivalence of the intrinsic energy flow
        Tirr: temperature equivalence of the irradiation
        f = 1 at the substellar point, f = 1/2 for a day-side average
            and f = 1/4 for an averaging over the whole planetary surface

    Returns:
        array: temperature profile

    """
    tau = pressures * 1.0e6 * kappa / gravity  # Equation (51)
    invsq3 = 1.0 / jnp.sqrt(3.0)
    fac = 2.0 / 3.0 + invsq3 * (
        1.0 / gamma + (gamma - 1.0 / gamma) * jnp.exp(-gamma * tau / invsq3)
    )
    Tarr = (0.75 * Tint**4 * (2.0 / 3.0 + tau) + 0.75 * Tirr**4 * f * fac) ** 0.25

    return Tarr


def Teq2Tirr(Teq):
    """Tirr from equilibrium temperature and intrinsic temperature.

    Args:
        Teq: equilibrium temperature

    Return:
        Tirr: iradiation temperature

    Note:
        Here we assume A=0 (albedo) and beta=1 (fully-energy distributed)
    """
    return (2.0**0.5) * Teq


def Teff2Tirr(Teff, Tint):
    """Tirr from effective temperature and intrinsic temperature.

    Args:
        Teff: effective temperature
        Tint: intrinsic temperature

    Return:
        Tirr: iradiation temperature

    Note:
        Here we assume A=0 (albedo) and beta=1 (fully-energy distributed)
    """
    return (4.0 * Teff**4 - Tint**4) ** 0.25
