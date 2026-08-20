import jax
import jax.numpy as jnp
import pytest
import numpy as np
from exojax.atm.atmprof import hydrostatic_radius_profile
from exojax.atm.atmprof import pressure_boundary_logspace
from exojax.atm.atmprof import pressure_layer_logspace
from exojax.atm.atmprof import pressure_layer_logspace_from_boundaries
from exojax.atm.atmprof import pressure_upper_logspace
from exojax.atm.atmprof import pressure_lower_logspace
from exojax.atm.atmprof import pressure_scale_height
from exojax.utils.constants import G, bar_cgs


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
    context = (
        jax.experimental.enable_x64()
        if enable_x64
        else jax.experimental.disable_x64()
    )
    with context:
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

    with jax.experimental.disable_x64():
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
    with jax.experimental.disable_x64():
        with pytest.raises(ValueError, match="cannot be represented"):
            pressure_layer_logspace_from_boundaries(
                top_boundary, bottom_boundary, nlayer=1
            )


def test_log_pressure_from_boundaries_rejects_indistinct_jax_layers():
    with jax.experimental.disable_x64():
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
    with jax.experimental.disable_x64():
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
    assert np.all(np.log10(p_upper) - ref < 1.0e-5)


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
    assert np.all(np.log10(p_lower) - ref < 1.0e-5)


def test_pressure_scale_height_earth():
    gravity_earth = 980.665  # cm/s2
    T = 288.15  # K
    mu_earth = 28.9644
    H = pressure_scale_height(gravity_earth, T, mu_earth)
    ref = 843465.7516276574  # cm (8.4km)

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


def test_atmospheric_scale_height_for_isothermal_with_analytic():
    from exojax.utils.grids import wavenumber_grid
    from exojax.rt import ArtTransPure
    from jax import config

    config.update("jax_enable_x64", True)
    mu_fid = 28.00863
    T_fid = 500.0
    Nx = 100000
    nu_grid, wav, res = wavenumber_grid(
        22000.0, 26500.0, Nx, unit="AA", xsmode="premodit"
    )
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


if __name__ == "__main__":
    test_log_pressure_is_constant()
    test_atmospheric_scale_height_for_isothermal_with_analytic()
    test_pressure_upper_logspace()
    test_pressure_lower_logspace()
    test_pressure_scale_height_earth()
