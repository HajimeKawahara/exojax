from contextlib import contextmanager

import jax
import jax.numpy as jnp
import pytest
import numpy as np
from exojax.atm.atmprof import hydrostatic_radius_profile
from exojax.atm.atmprof import hydrostatic_radius_profile_ideal_gas
from exojax.atm.atmprof import normalized_layer_height
from exojax.atm.atmprof import pressure_boundary_logspace
from exojax.atm.atmprof import pressure_layer_logspace
from exojax.atm.atmprof import pressure_layer_logspace_from_boundaries
from exojax.atm.atmprof import pressure_upper_logspace
from exojax.atm.atmprof import pressure_lower_logspace
from exojax.atm.atmprof import pressure_scale_height
from exojax.utils.constants import G, bar_cgs, kB, m_u


@contextmanager
def temporary_x64(enabled):
    previous = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", enabled)
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_log_pressure_is_constant():
    pressure, dParr, k = pressure_layer_logspace(
        log_pressure_top=-8.0,
        log_pressure_btm=2.0,
        nlayer=20,
        mode="ascending",
        numpy=False,
    )

    # check P[n-1] = k P[n]
    assert np.all(np.abs(1.0 - pressure[1:] * k / pressure[:-1]) < 1.0e-5)


@pytest.mark.parametrize("use_numpy", [False, True])
def test_log_pressure_from_boundaries(use_numpy):
    pressure, dpressure, k, pressure_boundary = (
        pressure_layer_logspace_from_boundaries(
            log_pressure_top_boundary=-4.0,
            log_pressure_btm_boundary=2.0,
            nlayer=3,
            numpy=use_numpy,
        )
    )

    expected_boundary = np.array([1.0e-4, 1.0e-2, 1.0, 1.0e2])
    expected_pressure = np.sqrt(expected_boundary[:-1] * expected_boundary[1:])
    np.testing.assert_allclose(pressure_boundary, expected_boundary, rtol=1.0e-6)
    np.testing.assert_allclose(pressure, expected_pressure, rtol=1.0e-6)
    np.testing.assert_allclose(
        dpressure, np.diff(expected_boundary), rtol=1.0e-6
    )
    assert k == pytest.approx(1.0e-2)
    assert pressure.dtype == pressure_boundary.dtype
    assert dpressure.dtype == pressure_boundary.dtype
    assert k.dtype == pressure_boundary.dtype
    assert isinstance(pressure_boundary, np.ndarray) is use_numpy


@pytest.mark.parametrize("use_numpy", [False, True])
def test_log_pressure_from_boundaries_preserves_endpoints(use_numpy):
    top_boundary = 3.7e-7
    bottom_boundary = 42.3
    _, _, _, pressure_boundary = pressure_layer_logspace_from_boundaries(
        np.log10(top_boundary),
        np.log10(bottom_boundary),
        nlayer=7,
        numpy=use_numpy,
    )
    array_module = np if use_numpy else jnp

    expected_top = array_module.asarray(
        top_boundary, dtype=pressure_boundary.dtype
    )
    expected_bottom = array_module.asarray(
        bottom_boundary, dtype=pressure_boundary.dtype
    )
    assert pressure_boundary[0] == expected_top
    assert pressure_boundary[-1] == expected_bottom


@pytest.mark.parametrize("enable_x64", [False, True])
def test_log_pressure_from_boundaries_uses_active_jax_dtype(enable_x64):
    with temporary_x64(enable_x64):
        pressure, dpressure, k, pressure_boundary = (
            pressure_layer_logspace_from_boundaries(-4.0, 2.0, nlayer=4)
        )
        expected_dtype = np.dtype(np.float64 if enable_x64 else np.float32)

        assert pressure.dtype == expected_dtype
        assert dpressure.dtype == expected_dtype
        assert k.dtype == expected_dtype
        assert pressure_boundary.dtype == expected_dtype


def test_log_pressure_from_boundaries_supports_jit_and_grad():
    def grid_sum(log_pressure_boundaries):
        pressure, dpressure, k, pressure_boundary = (
            pressure_layer_logspace_from_boundaries(
                log_pressure_boundaries[0],
                log_pressure_boundaries[1],
                nlayer=4,
            )
        )
        return (
            jnp.sum(pressure)
            + jnp.sum(dpressure)
            + k
            + jnp.sum(pressure_boundary)
        )

    value, gradient = jax.jit(jax.value_and_grad(grid_sum))(
        jnp.array([-4.0, 2.0])
    )

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))

    static_grid = jax.jit(
        lambda top, bottom: pressure_layer_logspace_from_boundaries(
            top, bottom, nlayer=4
        ),
        static_argnums=(0, 1),
    )(-4.0, 2.0)
    assert static_grid[0].shape == (4,)
    assert static_grid[3].shape == (5,)


@pytest.mark.parametrize(
    "bottom_boundary, reference_point, message",
    [(2.0, 2.0, "between 0 and 1"), (np.inf, 0.5, "finite")],
)
def test_log_pressure_from_boundaries_validates_static_values_during_jit(
    bottom_boundary, reference_point, message
):
    compiled_grid = jax.jit(
        lambda top_boundary: pressure_layer_logspace_from_boundaries(
            top_boundary,
            bottom_boundary,
            nlayer=4,
            reference_point=reference_point,
        )
    )

    with pytest.raises(ValueError, match=message):
        compiled_grid(-4.0)


@pytest.mark.parametrize(
    "top_boundary, reference_point",
    [(np.array([-4.0]), 0.5), (-4.0, np.array([0.5]))],
)
def test_log_pressure_from_boundaries_rejects_nonscalar_inputs(
    top_boundary, reference_point
):
    with pytest.raises(ValueError, match="must be scalar"):
        pressure_layer_logspace_from_boundaries(
            top_boundary, 2.0, nlayer=4, reference_point=reference_point
        )


def test_log_pressure_from_boundaries_rejects_numpy_backend_during_jit():
    compiled_grid = jax.jit(
        lambda top_boundary: pressure_layer_logspace_from_boundaries(
            top_boundary, 2.0, nlayer=4, numpy=True
        )
    )

    with pytest.raises(ValueError, match="NumPy backend"):
        compiled_grid(-4.0)


def test_log_pressure_from_boundaries_validates_static_outputs_during_grad():
    def pressure_sum(reference_point):
        pressure, _, _, _ = pressure_layer_logspace_from_boundaries(
            -37.0, 37.0, nlayer=1, reference_point=reference_point
        )
        return jnp.sum(pressure)

    with temporary_x64(False):
        with pytest.raises(ValueError, match="active array dtype"):
            jax.grad(pressure_sum)(0.5)


def test_log_pressure_from_boundaries_supports_jax_bfloat16():
    pressure, dpressure, k, pressure_boundary = (
        pressure_layer_logspace_from_boundaries(
            jnp.asarray(-4.0, dtype=jnp.bfloat16),
            jnp.asarray(2.0, dtype=jnp.bfloat16),
            nlayer=4,
        )
    )

    assert pressure.dtype == jnp.bfloat16
    assert dpressure.dtype == jnp.bfloat16
    assert k.dtype == jnp.bfloat16
    assert pressure_boundary.dtype == jnp.bfloat16


def test_log_pressure_from_boundaries_matches_boundary_helpers():
    reference_point = 0.25
    pressure, _, k, pressure_boundary = (
        pressure_layer_logspace_from_boundaries(
            -4.0,
            2.0,
            nlayer=4,
            reference_point=reference_point,
            numpy=True,
        )
    )

    np.testing.assert_allclose(
        pressure_upper_logspace(pressure, k, reference_point),
        pressure_boundary[:-1],
    )
    np.testing.assert_allclose(
        pressure_lower_logspace(pressure, k, reference_point),
        pressure_boundary[1:],
    )


@pytest.mark.parametrize("reference_point", [0.0, 0.5, 1.0])
def test_log_pressure_from_boundaries_matches_legacy_grid(reference_point):
    log_pressure_top = -3.0
    log_pressure_btm = 2.0
    nlayer = 6
    pressure, dpressure, k = pressure_layer_logspace(
        log_pressure_top,
        log_pressure_btm,
        nlayer,
        reference_point=reference_point,
        numpy=True,
    )
    dlogP = (log_pressure_btm - log_pressure_top) / (nlayer - 1)
    exact_grid = pressure_layer_logspace_from_boundaries(
        log_pressure_top - reference_point * dlogP,
        log_pressure_btm + (1.0 - reference_point) * dlogP,
        nlayer,
        reference_point=reference_point,
        numpy=True,
    )

    np.testing.assert_allclose(exact_grid[0], pressure)
    np.testing.assert_allclose(exact_grid[1], dpressure)
    assert exact_grid[2] == pytest.approx(k)
    np.testing.assert_allclose(
        exact_grid[3],
        pressure_boundary_logspace(
            pressure, k, reference_point=reference_point, numpy=True
        ),
    )


@pytest.mark.parametrize(
    "reference_point, expected",
    [
        (0.0, 1.0e-3),
        (0.25, 10.0**-2.25),
        (0.5, 10.0**-1.5),
        (1.0, 1.0),
    ],
)
def test_log_pressure_from_boundaries_reference_point(reference_point, expected):
    pressure, _, _, _ = pressure_layer_logspace_from_boundaries(
        log_pressure_top_boundary=-3.0,
        log_pressure_btm_boundary=0.0,
        nlayer=1,
        reference_point=reference_point,
        numpy=True,
    )

    assert pressure.shape == (1,)
    assert pressure[0] == pytest.approx(expected)


def test_log_pressure_from_boundaries_single_layer():
    pressure, dpressure, k, pressure_boundary = (
        pressure_layer_logspace_from_boundaries(
            log_pressure_top_boundary=-2.0,
            log_pressure_btm_boundary=1.0,
            nlayer=1,
        )
    )

    np.testing.assert_allclose(pressure_boundary, [1.0e-2, 1.0e1])
    np.testing.assert_allclose(pressure, [np.sqrt(1.0e-2 * 1.0e1)])
    np.testing.assert_allclose(dpressure, [1.0e1 - 1.0e-2])
    assert k == pytest.approx(1.0e-3)


@pytest.mark.parametrize("nlayer", [0, -1, 1.5, True])
def test_log_pressure_from_boundaries_rejects_invalid_nlayer(nlayer):
    with pytest.raises(ValueError, match="positive integer"):
        pressure_layer_logspace_from_boundaries(-3.0, 1.0, nlayer)


@pytest.mark.parametrize("reference_point", [-0.1, 1.1])
def test_log_pressure_from_boundaries_rejects_invalid_reference_point(
    reference_point,
):
    with pytest.raises(ValueError, match="between 0 and 1"):
        pressure_layer_logspace_from_boundaries(
            -3.0, 1.0, 4, reference_point=reference_point
        )


@pytest.mark.parametrize(
    "top_boundary, bottom_boundary, message",
    [
        (-3.0, -3.0, "greater than"),
        (0.0, -1.0, "greater than"),
        (np.nan, 1.0, "finite"),
        (-3.0, np.inf, "finite"),
    ],
)
def test_log_pressure_from_boundaries_rejects_invalid_boundaries(
    top_boundary, bottom_boundary, message
):
    with pytest.raises(ValueError, match=message):
        pressure_layer_logspace_from_boundaries(
            top_boundary, bottom_boundary, nlayer=4
        )


@pytest.mark.parametrize(
    "top_boundary, bottom_boundary", [(-400.0, 0.0), (0.0, 400.0)]
)
def test_log_pressure_from_boundaries_rejects_unrepresentable_numpy_bounds(
    top_boundary, bottom_boundary
):
    with pytest.raises(ValueError, match="cannot be represented"):
        pressure_layer_logspace_from_boundaries(
            top_boundary, bottom_boundary, nlayer=1, numpy=True
        )


@pytest.mark.parametrize(
    "top_boundary, bottom_boundary", [(-50.0, 0.0), (0.0, 40.0)]
)
def test_log_pressure_from_boundaries_rejects_unrepresentable_jax_bounds(
    top_boundary, bottom_boundary
):
    with temporary_x64(False):
        with pytest.raises(ValueError, match="cannot be represented"):
            pressure_layer_logspace_from_boundaries(
                top_boundary, bottom_boundary, nlayer=1
            )


def test_log_pressure_from_boundaries_rejects_indistinct_jax_layers():
    with temporary_x64(False):
        with pytest.raises(ValueError, match="not distinct"):
            pressure_layer_logspace_from_boundaries(0.0, 1.0e-8, nlayer=2)


@pytest.mark.parametrize(
    "top_boundary, bottom_boundary, nlayer",
    [
        (-38.0, 0.0, 1),
        (-37.0, -36.99999, 1),
        (-37.0, 37.0, 1),
    ],
)
def test_log_pressure_from_boundaries_rejects_unstable_float32_grid(
    top_boundary, bottom_boundary, nlayer
):
    with temporary_x64(False):
        with pytest.raises(ValueError, match="active array dtype"):
            pressure_layer_logspace_from_boundaries(
                top_boundary, bottom_boundary, nlayer
            )


def test_log_pressure_from_boundaries_rejects_underflowed_numpy_k():
    with pytest.raises(ValueError, match="active array dtype"):
        pressure_layer_logspace_from_boundaries(
            -300.0, 300.0, nlayer=1, numpy=True
        )


def test_pressure_upper_logspace():
    pressure, dParr, k = pressure_layer_logspace(
        log_pressure_top=-3.0,
        log_pressure_btm=2.0,
        nlayer=6,
        mode="ascending",
        numpy=False,
    )
    p_upper = pressure_upper_logspace(pressure, k)
    ref = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5])
    np.testing.assert_allclose(np.log10(p_upper), ref, rtol=0.0, atol=1.0e-5)


def test_pressure_lower_logspace():
    pressure, dParr, k = pressure_layer_logspace(
        log_pressure_top=-3.0,
        log_pressure_btm=2.0,
        nlayer=6,
        mode="ascending",
        numpy=False,
    )
    p_lower = pressure_lower_logspace(pressure, k)
    ref = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    np.testing.assert_allclose(np.log10(p_lower), ref, rtol=0.0, atol=1.0e-5)


@pytest.mark.parametrize(
    "gravity_earth, T, mu_earth, ref",
    [
        (980.665, 288.15, 28.9644, 843465.7516276574),
        (980.0, 300.0, 28.8, 883764.8664527453),
    ],
)
def test_pressure_scale_height_earth(gravity_earth, T, mu_earth, ref):
    H = pressure_scale_height(gravity_earth, T, mu_earth)

    assert H == pytest.approx(ref)


def test_hydrostatic_radius_profile_matches_layer_solution():
    pressure_boundaries = jnp.array([1.0, 4.0, 10.0])
    mass_density_layers = jnp.array([5.0e-4, 2.0e-3])
    planet_mass = 5.0e27
    radius_bottom = 6.0e8

    radius, gravity = hydrostatic_radius_profile(
        pressure_boundaries,
        mass_density_layers,
        planet_mass,
        radius_bottom,
    )

    gravitational_parameter = G * planet_mass
    inverse_radius_bottom = 1.0 / radius_bottom
    inverse_radius_middle = inverse_radius_bottom - (
        (10.0 - 4.0) * bar_cgs
        / (2.0e-3 * gravitational_parameter)
    )
    inverse_radius_top = inverse_radius_middle - (
        (4.0 - 1.0) * bar_cgs
        / (5.0e-4 * gravitational_parameter)
    )
    expected_radius = 1.0 / np.array(
        [inverse_radius_top, inverse_radius_middle, inverse_radius_bottom]
    )
    expected_gravity = gravitational_parameter / expected_radius**2

    assert radius.shape == pressure_boundaries.shape
    assert gravity.shape == pressure_boundaries.shape
    assert radius[-1] == jnp.asarray(radius_bottom, dtype=radius.dtype)
    np.testing.assert_allclose(radius, expected_radius, rtol=1.0e-6)
    np.testing.assert_allclose(gravity, expected_gravity, rtol=1.0e-6)


def test_hydrostatic_radius_profile_supports_jit_and_grad():
    pressure_boundaries = jnp.array([1.0, 4.0, 10.0])
    mass_density_layers = jnp.array([5.0e-4, 2.0e-3])

    def radius_top(log_scales):
        density_scale, mass_scale = jnp.exp(log_scales)
        radius, _ = hydrostatic_radius_profile(
            pressure_boundaries,
            mass_density_layers * density_scale,
            1.8986e30 * mass_scale,
            7.1492e9,
        )
        return radius[0]

    radius, gradient = jax.jit(jax.value_and_grad(radius_top))(jnp.zeros(2))
    assert jnp.isfinite(radius)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient < 0.0)


@pytest.mark.parametrize(
    "scheme", ["variable_gravity", "layer_constant_gravity"]
)
def test_hydrostatic_radius_profile_ideal_gas_nonuniform_manual(scheme):
    pressure_boundaries = np.array([1.0e-5, 3.0e-4, 2.0e-2, 1.0])
    temperature = np.array([500.0, 800.0, 1200.0])
    mean_molecular_weight = np.array([2.3, 5.0, 18.0])
    radius_bottom = 7.0e9
    gravity_bottom = 2500.0

    expected_radius = np.empty_like(pressure_boundaries)
    expected_radius[-1] = radius_bottom
    for index in range(len(temperature) - 1, -1, -1):
        radius_lower = expected_radius[index + 1]
        gravity_lower = gravity_bottom * (radius_bottom / radius_lower) ** 2
        scale_height = (
            kB
            * temperature[index]
            / (m_u * mean_molecular_weight[index] * gravity_lower)
        )
        log_pressure_ratio = np.log(
            pressure_boundaries[index + 1] / pressure_boundaries[index]
        )
        if scheme == "variable_gravity":
            expected_radius[index] = radius_lower / (
                1.0 - scale_height * log_pressure_ratio / radius_lower
            )
        else:
            expected_radius[index] = (
                radius_lower + scale_height * log_pressure_ratio
            )
    expected_gravity = gravity_bottom * (
        radius_bottom / expected_radius
    ) ** 2

    radius, gravity = hydrostatic_radius_profile_ideal_gas(
        pressure_boundaries,
        temperature,
        mean_molecular_weight,
        radius_bottom,
        gravity_bottom,
        hydrostatic_scheme=scheme,
    )

    np.testing.assert_allclose(radius, expected_radius, rtol=2.0e-6)
    np.testing.assert_allclose(gravity, expected_gravity, rtol=2.0e-6)
    assert radius.shape == pressure_boundaries.shape
    assert gravity.shape == pressure_boundaries.shape
    assert radius[-1] == jnp.asarray(radius_bottom, dtype=radius.dtype)
    assert gravity[-1] == jnp.asarray(gravity_bottom, dtype=gravity.dtype)
    assert jnp.all(radius[:-1] > radius[1:])
    assert jnp.all(gravity[:-1] < gravity[1:])


def test_hydrostatic_radius_profile_ideal_gas_matches_normalized_height():
    pressure_boundaries = jnp.logspace(-6.0, 1.0, 9)
    pressure_decrease_rate = pressure_boundaries[0] / pressure_boundaries[1]
    temperature = jnp.linspace(450.0, 950.0, 8)
    mean_molecular_weight = jnp.linspace(2.3, 12.0, 8)
    radius_bottom = 7.1492e9
    gravity_bottom = 2478.6

    radius, _ = hydrostatic_radius_profile_ideal_gas(
        pressure_boundaries,
        temperature,
        mean_molecular_weight,
        radius_bottom,
        gravity_bottom,
    )
    normalized_height, normalized_radius_lower = normalized_layer_height(
        temperature,
        pressure_decrease_rate,
        mean_molecular_weight,
        radius_bottom,
        gravity_bottom,
    )

    np.testing.assert_allclose(
        radius[1:] / radius_bottom,
        normalized_radius_lower,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        radius[:-1] / radius_bottom,
        normalized_radius_lower + normalized_height,
        rtol=2.0e-6,
    )


def test_hydrostatic_radius_profile_ideal_gas_scalar_mmw_one_layer():
    pressure_boundaries = jnp.array([0.1, 1.0])
    temperature = jnp.array([700.0])
    radius_bottom = 6.0e9
    gravity_bottom = 1000.0

    radius_scalar, gravity_scalar = hydrostatic_radius_profile_ideal_gas(
        pressure_boundaries,
        temperature,
        2.3,
        radius_bottom,
        gravity_bottom,
    )
    radius_profile, gravity_profile = hydrostatic_radius_profile_ideal_gas(
        pressure_boundaries,
        temperature,
        jnp.array([2.3]),
        radius_bottom,
        gravity_bottom,
    )

    np.testing.assert_allclose(radius_scalar, radius_profile)
    np.testing.assert_allclose(gravity_scalar, gravity_profile)
    assert radius_scalar[-1] == jnp.asarray(
        radius_bottom, dtype=radius_scalar.dtype
    )
    assert gravity_scalar[-1] == jnp.asarray(
        gravity_bottom, dtype=gravity_scalar.dtype
    )


@pytest.mark.parametrize(
    "argument,value,error",
    [
        ("pressure_boundaries", jnp.ones((2, 2)), "pressure_boundaries"),
        ("pressure_boundaries", jnp.ones(1), "pressure_boundaries"),
        ("temperature", jnp.ones(3), "temperature"),
        ("temperature", jnp.asarray(500.0), "temperature"),
        ("mean_molecular_weight", jnp.ones(3), "mean_molecular_weight"),
        ("mean_molecular_weight", jnp.ones((2, 1)), "mean_molecular_weight"),
        ("radius_bottom", jnp.ones(1), "radius_bottom"),
        ("gravity_bottom", jnp.ones(1), "gravity_bottom"),
    ],
)
def test_hydrostatic_radius_profile_ideal_gas_rejects_bad_shapes(
    argument, value, error
):
    inputs = {
        "pressure_boundaries": jnp.array([0.01, 0.1, 1.0]),
        "temperature": jnp.array([500.0, 700.0]),
        "mean_molecular_weight": jnp.array([2.3, 2.5]),
        "radius_bottom": 7.0e9,
        "gravity_bottom": 2500.0,
    }
    inputs[argument] = value

    with pytest.raises(ValueError, match=error):
        hydrostatic_radius_profile_ideal_gas(**inputs)


def test_hydrostatic_radius_profile_ideal_gas_rejects_bad_scheme():
    with pytest.raises(ValueError, match="Unknown hydrostatic scheme"):
        hydrostatic_radius_profile_ideal_gas(
            jnp.array([0.01, 0.1, 1.0]),
            jnp.array([500.0, 700.0]),
            2.3,
            7.0e9,
            2500.0,
            hydrostatic_scheme="unknown",
        )


@pytest.mark.parametrize(
    "scheme", ["variable_gravity", "layer_constant_gravity"]
)
def test_hydrostatic_radius_profile_ideal_gas_supports_jit_and_grad(scheme):
    pressure_boundaries = jnp.array([1.0e-5, 1.0e-2, 1.0])
    temperature = jnp.array([500.0, 900.0])

    def radius_top(log_temperature_scale):
        radius, _ = hydrostatic_radius_profile_ideal_gas(
            pressure_boundaries,
            temperature * jnp.exp(log_temperature_scale),
            2.3,
            7.0e9,
            2500.0,
            hydrostatic_scheme=scheme,
        )
        return radius[0]

    radius, gradient = jax.jit(jax.value_and_grad(radius_top))(0.0)
    assert jnp.isfinite(radius)
    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_atmospheric_scale_height_for_isothermal_with_analytic():
    from exojax.rt import ArtTransPure

    mu_fid = 28.00863
    T_fid = 500.0
    art = ArtTransPure(pressure_top=1.0e-10, pressure_btm=1.0e1, nlayer=100)
    Tarr = T_fid * np.ones_like(art.pressure)
    gravity_btm = 2478.57730044555
    radius_btm = 7149200000.0
    mmw = mu_fid * np.ones_like(art.pressure)

    normalized_height, normalized_radius_lower = art.atmosphere_height(
        Tarr, mmw, radius_btm, gravity_btm
    )

    # theoretical value
    H_btm = pressure_scale_height(gravity_btm, T_fid, mu_fid)
    dq = np.arange(0, len(art.pressure))[::-1] * np.log(
        art.pressure_decrease_rate
    )  # n log(k)
    normalized_radius_theory = 1 / (1 + H_btm / radius_btm * dq)
    res = 1.0 - (normalized_radius_lower - 1.0) / (normalized_radius_theory - 1.0)

    assert np.all(np.abs(res[:-1]) < 1.0e-11)


def test_first_layer_height_from_compute_normalized_radius_profile():
    from exojax.atm.atmprof import pressure_layer_logspace
    pressure, dParr, pressure_decrease_rate = pressure_layer_logspace(
        log_pressure_top=-8., log_pressure_btm=2., nlayer=20)
    T0 = 300.0
    mmw0 = 28.8
    temperature = T0 * np.ones_like(pressure)
    mmw = mmw0 * np.ones_like(pressure)
    radius_btm = 6500.0 * 1.e5
    gravity_btm = 980.

    normalized_height, normalized_radius_lower = normalized_layer_height(
        temperature, pressure_decrease_rate, mmw, radius_btm, gravity_btm)

    normalized_radius_top = normalized_radius_lower[0] + normalized_height[0]
    assert normalized_radius_top == pytest.approx(1.0340775666464417)
    assert jnp.sum(normalized_height[1:]) + 1.0 == pytest.approx(
        normalized_radius_lower[0])
    assert normalized_radius_lower[-1] == 1.0
