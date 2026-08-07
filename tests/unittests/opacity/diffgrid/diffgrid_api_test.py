import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database.contracts import Lines
from exojax.database.contracts import MDBMeta
from exojax.database.contracts import MDBSnapshot
from exojax.opacity import OpaDiffgrid
from exojax.opacity.premodit.api import OpaPremodit
from exojax.rt import ArtEmisPure
from exojax.utils.grids import wavenumber_grid


class _AnalyticTeacher:
    method = "analytic"
    ready = True
    nu_grid = np.asarray([1000.0, 1001.0])

    def xsmatrix(self, temperature, pressure):
        value = jnp.exp(-1000.0 / temperature)
        return jnp.broadcast_to(
            value[:, None], (len(pressure), len(self.nu_grid))
        )


def _small_exomol_snapshot():
    meta = MDBMeta(
        dbtype="exomol",
        molmass=18.0,
        T_gQT=np.asarray([300.0, 600.0, 1000.0, 1500.0, 2200.0]),
        gQT=np.asarray([1.0, 1.35, 1.9, 2.7, 3.8]),
    )
    lines = Lines(
        nu_lines=np.asarray([997.0, 999.0, 1000.0, 1002.0, 1004.0]),
        elower=np.asarray([20.0, 350.0, 900.0, 1800.0, 3200.0]),
        line_strength_ref_original=np.asarray(
            [2.0e-23, 4.0e-23, 3.0e-23, 5.0e-23, 2.5e-23]
        ),
    )
    return MDBSnapshot(
        meta=meta,
        lines=lines,
        n_Texp=np.asarray([0.45, 0.55, 0.5, 0.65, 0.4]),
        alpha_ref=np.asarray([0.06, 0.07, 0.05, 0.08, 0.06]),
    )


@pytest.fixture(scope="module")
def diffgrid_setup():
    nu_grid, _, _ = wavenumber_grid(
        990.0, 1010.0, 64, unit="cm-1", xsmode="premodit"
    )
    teacher = OpaPremodit.from_snapshot(
        _small_exomol_snapshot(),
        nu_grid,
        allow_32bit=True,
        broadening_resolution={"mode": "single", "value": None},
    )
    teacher.manual_setting(
        dE=100.0,
        Tref=1000.0,
        Twt=1200.0,
        Tmin=500.0,
        Tmax=2000.0,
    )
    pressure = np.logspace(0.0, 1.0, 3)
    temperature_grid = np.asarray(
        [600.0, 850.0, 1150.0, 1500.0, 1900.0], dtype=np.float32
    )
    opa = OpaDiffgrid(teacher, temperature_grid, pressure)
    return opa, teacher, pressure, temperature_grid


def _linear_log_cross_section(opa, temperature):
    inverse_temperature_grid = np.asarray(opa.inverse_temperature_grid)
    log_cross_section_grid = np.asarray(opa.log_cross_section_grid)
    inverse_temperature = 1.0 / np.asarray(temperature)
    interpolated = []
    for layer, coordinate in enumerate(inverse_temperature):
        upper = np.searchsorted(
            inverse_temperature_grid, coordinate, side="right"
        )
        lower = np.clip(upper - 1, 0, len(inverse_temperature_grid) - 2)
        upper = lower + 1
        fraction = (
            (coordinate - inverse_temperature_grid[lower])
            / (inverse_temperature_grid[upper] - inverse_temperature_grid[lower])
        )
        interpolated.append(
            (1.0 - fraction) * log_cross_section_grid[layer, lower]
            + fraction * log_cross_section_grid[layer, upper]
        )
    return np.asarray(interpolated)


def test_diffgrid_matches_premodit_at_temperature_nodes(diffgrid_setup):
    opa, teacher, pressure, _ = diffgrid_setup
    temperature = np.asarray([600.0, 1150.0, 1900.0])

    expected = np.maximum(
        np.asarray(teacher.xsmatrix(temperature, pressure)),
        np.exp(np.asarray(opa.diffgrid_info.log_cross_section_floor)),
    )
    actual = opa.xsmatrix(Tarr=temperature, Parr=pressure)
    inverse_temperature = 1.0 / temperature
    node_index = np.argmin(
        np.abs(
            np.asarray(opa.inverse_temperature_grid)[None, :]
            - inverse_temperature[:, None]
        ),
        axis=1,
    )
    stored = np.exp(
        np.stack(
            [
                np.asarray(opa.log_cross_section_grid)[layer, index]
                for layer, index in enumerate(node_index)
            ]
        )
    )

    assert opa.method == "diffgrid"
    assert opa.teacher_method == "premodit"
    np.testing.assert_allclose(actual, stored, rtol=3.0e-5, atol=0.0)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=0.0)


def test_diffgrid_improves_on_linear_log_interpolation(diffgrid_setup):
    opa, teacher, pressure, _ = diffgrid_setup
    temperature = np.asarray([720.0, 1000.0, 1700.0])
    direct = np.asarray(teacher.xsmatrix(temperature, pressure))
    direct_log = np.log(np.maximum(direct, 1.0e-35))
    hermite_log = np.log(np.asarray(opa.xsmatrix(temperature)))
    linear_log = _linear_log_cross_section(opa, temperature)

    hermite_error = np.mean(np.abs(hermite_log - direct_log))
    linear_error = np.mean(np.abs(linear_log - direct_log))

    assert hermite_error < 0.5 * linear_error


def test_diffgrid_rejects_pressure_mismatch(diffgrid_setup):
    opa, _, pressure, _ = diffgrid_setup
    temperature = np.asarray([700.0, 1000.0, 1600.0])

    with pytest.raises(ValueError, match="rebuild the table"):
        opa.xsmatrix(temperature, pressure * 1.01)
    with pytest.raises(ValueError, match="shape does not match"):
        opa.xsmatrix(temperature, pressure[:-1])

    with pytest.raises(ValueError, match="cannot be a traced argument"):
        jax.jit(lambda value, p: opa.xsmatrix(value, p))(temperature, pressure)


def test_diffgrid_rejects_temperature_nodes_collapsed_by_jax_dtype():
    previous_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="active JAX dtype"):
            OpaDiffgrid(
                _AnalyticTeacher(),
                np.asarray([1000.0, 1000.00001]),
                np.asarray([1.0]),
            )
        with pytest.raises(ValueError, match="smallest normal value"):
            OpaDiffgrid(
                _AnalyticTeacher(),
                np.asarray([800.0, 1200.0]),
                np.asarray([1.0]),
                min_cross_section=1.0e-45,
            )
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_diffgrid_supports_jax_transformations(diffgrid_setup):
    opa, _, pressure, _ = diffgrid_setup
    temperature = jnp.asarray([720.0, 1000.0, 1700.0])
    tangent = jnp.asarray([1.0, -0.5, 0.25])

    compiled = jax.jit(opa.xsmatrix)(temperature)
    compiled_with_pressure = jax.jit(
        lambda value: opa.xsmatrix(value, pressure)
    )(temperature)
    batched = jax.vmap(opa.xsmatrix)(
        jnp.stack([temperature, temperature + jnp.asarray([10.0, 10.0, -10.0])])
    )

    def objective(value):
        return jnp.sum(jnp.log(opa.xsmatrix(value)))

    gradient = jax.grad(objective)(temperature)
    value, directional_derivative = jax.jvp(objective, (temperature,), (tangent,))

    assert compiled.shape == (3, len(opa.nu_grid))
    np.testing.assert_allclose(compiled_with_pressure, compiled)
    assert batched.shape == (2, 3, len(opa.nu_grid))
    assert np.all(np.isfinite(np.asarray(compiled)))
    assert np.all(np.isfinite(np.asarray(batched)))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(directional_derivative))


def test_diffgrid_in_jitted_emission_model(diffgrid_setup):
    opa, teacher, pressure, _ = diffgrid_setup
    art = ArtEmisPure(
        pressure_top=pressure[0],
        pressure_btm=pressure[-1],
        nlayer=len(pressure),
        nu_grid=opa.nu_grid,
        nstream=2,
    )
    mixing_ratio = jnp.full(len(pressure), 1.0e-3)

    def spectrum(temperature):
        cross_section = opa.xsmatrix(temperature)
        optical_depth = art.opacity_profile_xs(
            cross_section, mixing_ratio, teacher.molmass, 2500.0
        )
        return art.run(optical_depth, temperature)

    temperature = jnp.asarray([720.0, 1000.0, 1700.0])
    flux = jax.jit(spectrum)(temperature)
    gradient = jax.grad(lambda value: jnp.sum(spectrum(value)))(temperature)

    assert flux.shape == opa.nu_grid.shape
    assert np.all(np.isfinite(np.asarray(flux)))
    assert np.all(np.isfinite(np.asarray(gradient)))
