import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity import OpaDiffgrid
from exojax.opacity.diffgrid.diagnostics import (
    compare_diffgrid_with_teacher,
    diffgrid_interval_midpoint_temperatures,
)


class _LogLinearTeacher:
    method = "analytic"
    ready = True
    nu_grid = np.asarray([1000.0, 1001.0, 1002.0], dtype=np.float32)

    def xsmatrix(self, temperature, pressure):
        temperature = jnp.asarray(temperature)
        pressure = jnp.asarray(pressure)
        spectral_offset = jnp.asarray([-0.2, 0.0, 0.3])
        log_cross_section = (
            -700.0 / temperature[:, None]
            + 0.1 * jnp.log(pressure[:, None])
            + spectral_offset[None, :]
        )
        return jnp.exp(log_cross_section)


class _FixedDiffgrid:
    ready = True
    nu_grid = np.asarray([1000.0, 1001.0])
    pressure_grid = jnp.asarray([0.1, 1.0])
    inverse_temperature_grid = jnp.asarray(
        [1.0 / 1600.0, 1.0 / 800.0, 1.0 / 400.0]
    )
    log_cross_section_floor = jnp.log(jnp.asarray(1.0e-4))

    def __init__(self, cross_section):
        self.cross_section = jnp.asarray(cross_section)

    def xsmatrix(self, temperature):
        return self.cross_section


class _FixedTeacher:
    ready = True

    def __init__(self, cross_section, nu_grid=(1000.0, 1001.0)):
        self.cross_section = jnp.asarray(cross_section)
        self.nu_grid = np.asarray(nu_grid)
        self.received_pressure = None

    def xsmatrix(self, temperature, pressure):
        self.received_pressure = np.asarray(pressure)
        return self.cross_section


def test_diffgrid_midpoint_temperatures_are_harmonic_means():
    diffgrid = _FixedDiffgrid(np.ones((2, 2)))

    actual = diffgrid_interval_midpoint_temperatures(diffgrid)

    np.testing.assert_allclose(actual, [3200.0 / 3.0, 1600.0 / 3.0])


def test_comparison_uses_archive_floor_and_reports_scalar_metrics():
    diffgrid = _FixedDiffgrid(
        [[1.0e-4, 2.0e-4], [4.0e-4, 8.0e-4]]
    )
    teacher = _FixedTeacher(
        [[-1.0e-5, 1.0e-4], [2.0e-4, 2.0e-4]]
    )

    summary = compare_diffgrid_with_teacher(
        diffgrid,
        teacher,
        np.asarray([500.0, 600.0]),
        quantiles=(0.5, 1.0),
    )

    assert summary.quantiles == (0.5, 1.0)
    np.testing.assert_allclose(
        summary.absolute_log_cross_section_error_quantiles,
        [np.log(2.0), np.log(4.0)],
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        summary.maximum_absolute_log_cross_section_error,
        np.log(4.0),
        rtol=2.0e-6,
    )
    assert summary.maximum_error_layer_index == 1
    assert summary.maximum_error_wavenumber_index == 1
    np.testing.assert_array_equal(
        teacher.received_pressure,
        np.asarray(diffgrid.pressure_grid),
    )


def test_real_diffgrid_matches_log_linear_teacher_at_interval_midpoints():
    teacher = _LogLinearTeacher()
    pressure_grid = np.asarray([0.1, 1.0], dtype=np.float32)
    diffgrid = OpaDiffgrid(
        teacher,
        temperature_grid=np.asarray(
            [1200.0, 800.0, 400.0], dtype=np.float32
        ),
        pressure_grid=pressure_grid,
    )

    midpoint_temperatures = diffgrid_interval_midpoint_temperatures(diffgrid)
    np.testing.assert_allclose(
        midpoint_temperatures,
        [960.0, 1600.0 / 3.0],
        rtol=2.0e-7,
    )
    for temperature in midpoint_temperatures:
        summary = compare_diffgrid_with_teacher(
            diffgrid,
            teacher,
            np.full(pressure_grid.shape, temperature),
        )
        assert summary.maximum_absolute_log_cross_section_error < 2.0e-6


@pytest.mark.parametrize("quantiles", [(), 0.5, (-0.01,), (1.01,), (np.nan,)])
def test_comparison_rejects_invalid_quantiles(quantiles):
    diffgrid = _FixedDiffgrid(np.ones((2, 2)))
    teacher = _FixedTeacher(np.ones((2, 2)))

    with pytest.raises(ValueError, match="quantiles"):
        compare_diffgrid_with_teacher(
            diffgrid,
            teacher,
            np.asarray([500.0, 600.0]),
            quantiles=quantiles,
        )


def test_comparison_rejects_pointwise_wavenumber_mismatch():
    diffgrid = _FixedDiffgrid(np.ones((2, 2)))
    teacher = _FixedTeacher(np.ones((2, 2)), nu_grid=(1001.0, 1000.0))

    with pytest.raises(ValueError, match="point by point"):
        compare_diffgrid_with_teacher(
            diffgrid,
            teacher,
            np.asarray([500.0, 600.0]),
        )


def test_comparison_rejects_wrong_temperature_and_output_shapes():
    diffgrid = _FixedDiffgrid(np.ones((2, 2)))
    teacher = _FixedTeacher(np.ones((2, 2)))
    with pytest.raises(ValueError, match="temperature_profile shape"):
        compare_diffgrid_with_teacher(diffgrid, teacher, np.asarray([500.0]))

    diffgrid = _FixedDiffgrid(np.ones((2, 3)))
    with pytest.raises(ValueError, match="Diffgrid cross-section matrix"):
        compare_diffgrid_with_teacher(
            diffgrid,
            teacher,
            np.asarray([500.0, 600.0]),
        )

    diffgrid = _FixedDiffgrid(np.ones((2, 2)))
    teacher = _FixedTeacher(np.ones((2, 3)))
    with pytest.raises(ValueError, match="Teacher cross-section matrix"):
        compare_diffgrid_with_teacher(
            diffgrid,
            teacher,
            np.asarray([500.0, 600.0]),
        )


@pytest.mark.parametrize(
    ("source", "diffgrid_values", "teacher_values"),
    [
        ("Diffgrid", [[np.nan, 1.0], [1.0, 1.0]], np.ones((2, 2))),
        ("Teacher", np.ones((2, 2)), [[np.inf, 1.0], [1.0, 1.0]]),
    ],
)
def test_comparison_rejects_nonfinite_cross_sections(
    source,
    diffgrid_values,
    teacher_values,
):
    diffgrid = _FixedDiffgrid(diffgrid_values)
    teacher = _FixedTeacher(teacher_values)

    with pytest.raises(FloatingPointError, match=source):
        compare_diffgrid_with_teacher(
            diffgrid,
            teacher,
            np.asarray([500.0, 600.0]),
        )
