"""Optional gas-chemistry integration tests, independent of retrieval data."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("exogibbs", reason="Optional RCE/TCE chemistry dependency")
from exogibbs.api.chemistry import ChemicalSetup

_SPEC = importlib.util.spec_from_file_location(
    "_rce_chemistry",
    Path(__file__).resolve().parents[3] / "examples" / "_rce_chemistry.py",
)
chemistry = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = chemistry
_SPEC.loader.exec_module(chemistry)


@pytest.fixture(autouse=True)
def x64():
    with jax.experimental.enable_x64():
        yield


def small_setup(hvector=None):
    # Independent ideal-gas H/H2 and C/O/CO association plus H+/e-/H- ions.
    return ChemicalSetup(
        formula_matrix=jnp.array(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0],
            ]
        ),
        hvector_func=hvector
        or (
            lambda t: jnp.array(
                [
                    0.0,
                    -6000.0 / t,
                    0.0,
                    0.0,
                    0.0,
                    -16000.0 / t,
                    0.0,
                    10000.0 / t,
                    500.0 / t,
                ]
            )
        ),
        species=("H", "H2", "He", "C", "O", "CO", "e-", "H+", "H-"),
        elements=("H", "He", "C", "O", "e-"),
        element_vector_reference=jnp.array([1.0, 0.08, 3e-4, 5e-4, 0.0]),
    )


def prepare(setup=None, **kwargs):
    options = dict(
        element_masses_u={"H": 1.00794, "He": 4.002602, "C": 12.0107, "O": 15.9994},
        isotope_convention="Natural-abundance neutral atomic masses; bulk chemistry",
        required_species=("H2", "He", "CO", "H+", "H-"),
        electron_species="e-",
        atomic_hydrogen_species="H",
        temperature_range=(500.0, 5000.0),
        pressure_range=(1e-6, 100.0),
    )
    options.update(kwargs)
    return chemistry.prepare_chemistry(
        setup or small_setup(), jnp.array([0.03, 0.4, 2.0]), **options
    )


def test_units_conservation_abundance_convention_and_layer_masses():
    adapter = prepare()
    temperature = jnp.array([1500.0, 2200.0, 3500.0])
    state = jax.jit(adapter)(temperature, 0.3, 1.2)
    assert np.all(state.diagnostics.valid)
    assert np.all(state.diagnostics.converged)
    np.testing.assert_allclose(
        state.n @ np.asarray(small_setup().formula_matrix).T,
        np.broadcast_to(state.b, (3, 5)),
        atol=1e-11,
    )
    np.testing.assert_allclose(np.sum(state.x, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(state.x, state.n / state.n.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(state.mass_fractions.sum(axis=1), 1.0, atol=1e-12)
    assert np.ptp(state.mmw) > 0.1
    np.testing.assert_allclose(state.b[:2], [1.0, 0.08])
    assert state.b[2] / state.b[3] == pytest.approx(1.2)
    assert state.b[2] + state.b[3] == pytest.approx(8e-4 * 10**0.3)
    density = 1e6 * adapter.pressure_bar / (chemistry.kB * temperature)
    np.testing.assert_allclose(state.number_density_e, state.x[:, 6] * density)
    np.testing.assert_allclose(state.number_density_h, state.x[:, 0] * density)
    other = adapter(temperature, 0.3, 0.5)
    assert state.metal_mass_fraction != pytest.approx(
        other.metal_mass_fraction, rel=1e-3
    )
    assert state.diagnostics.element_relative_error.max() < 1e-8
    assert state.diagnostics.charge_relative_error.max() < 1e-8


def test_repeated_jit_no_leaks_and_derivatives_against_resolved_finite_differences():
    adapter = prepare()
    temperature = jnp.array([1500.0, 2200.0, 3500.0])
    run = jax.jit(lambda p: adapter(temperature * jnp.exp(p[0]), p[1], p[2]))
    point, direction = jnp.array([0.0, 0.2, 0.8]), jnp.array([0.2, 0.3, 0.4])
    with jax.checking_leaks():
        original, shifted = run(point), run(point + 0.05 * direction)
    assert np.all(original.diagnostics.valid) and np.all(shifted.diagnostics.valid)
    assert not np.allclose(original.x, shifted.x)

    def objective(p):
        return jnp.sum(run(p).mmw)

    _, tangent = jax.jvp(objective, (point,), (direction,))
    gradient = jax.jit(jax.grad(objective))(point)
    assert tangent == pytest.approx(jnp.dot(gradient, direction), rel=1e-10)
    for step in (1e-3, 3e-4):
        plus, minus = run(point + step * direction), run(point - step * direction)
        assert np.all(plus.diagnostics.valid) and np.all(minus.diagnostics.valid)
        finite_difference = (plus.mmw.sum() - minus.mmw.sum()) / (2 * step)
        assert tangent == pytest.approx(finite_difference, rel=1e-5, abs=1e-10)


def test_failure_domain_guard_and_mixed_columns():
    adapter = prepare(max_iter=1)
    failed = jax.jit(adapter)(jnp.array([1500.0, 2200.0, 3500.0]), 0.0, 0.6)
    np.testing.assert_array_equal(failed.diagnostics.status, chemistry.NOT_CONVERGED)
    assert not np.any(failed.diagnostics.valid)

    # Runtime callbacks prove no thermochemistry runs on the invalid branch.
    called = []
    base_hvector = small_setup().hvector_func

    def checked_hvector(t):
        jax.debug.callback(lambda value: called.append(float(value)), t)
        return base_hvector(t)

    adapter = prepare(small_setup(checked_hvector))
    invalid = jax.jit(adapter)(jnp.array([100.0, 2200.0, 3500.0]), 0.0, 0.6)
    jax.block_until_ready(invalid)
    assert called == []
    np.testing.assert_array_equal(invalid.diagnostics.status, chemistry.OUTSIDE_DOMAIN)
    assert not invalid.diagnostics.in_domain
    assert np.all(np.isnan(invalid.x))
    for metal_scale, ratio in ((0.0, -1.0), (jnp.nan, 0.6), (1000.0, 0.6)):
        assert not adapter(
            jnp.array([1500.0, 2200.0, 3500.0]), metal_scale, ratio
        ).diagnostics.in_domain
    mixed = jax.jit(lambda ts: jax.lax.map(lambda t: adapter(t, 0.0, 0.6), ts))(
        jnp.array([[1500.0, 2200.0, 3500.0], [100.0, 2200.0, 3500.0]])
    )
    assert np.all(mixed.diagnostics.valid[0])
    np.testing.assert_array_equal(mixed.diagnostics.status[1], chemistry.OUTSIDE_DOMAIN)


def test_solver_convergence_is_not_sufficient_for_valid_composition():
    temperature = jnp.array([1500.0, 2200.0, 3500.0])
    loose = prepare(epsilon_crit=1e3)(temperature, 0.0, 0.6)
    assert np.all(loose.diagnostics.converged)
    np.testing.assert_array_equal(
        loose.diagnostics.status, chemistry.CONSERVATION_FAILED
    )
    assert not np.any(loose.diagnostics.valid)
    broken = prepare(small_setup(lambda t: jnp.full(9, jnp.nan)))(temperature, 0.0, 0.6)
    np.testing.assert_array_equal(broken.diagnostics.status, chemistry.NONFINITE)
    assert not np.any(broken.diagnostics.valid)


def test_missing_species_and_invalid_metadata_fail_at_preparation():
    with pytest.raises(ValueError, match="Missing required species"):
        prepare(required_species=("H2O",))
    with pytest.raises(ValueError, match="Electron and neutral atomic H"):
        prepare(atomic_hydrogen_species="H2")
    with pytest.raises(ValueError, match="Missing element masses"):
        prepare(element_masses_u={"H": 1.0})
    with pytest.raises(ValueError, match="isotope convention"):
        prepare(isotope_convention="")
    with pytest.raises(ValueError, match="pressure grid"):
        prepare(pressure_range=(1.0, 100.0))


def test_hminus_uses_number_densities_and_broadcasts_varying_layer_mmw():
    from exojax.database.hminus import log_hminus_continuum_single

    adapter = prepare()
    temperature = jnp.array([1500.0, 2200.0, 3500.0])
    state = adapter(temperature, 0.2, 0.8)
    nus = jnp.linspace(4000.0, 16000.0, 7)
    delta_pressure = 0.2 * adapter.pressure_bar
    gravity = 2e4
    actual = chemistry.hminus_optical_depth(
        nus, temperature, adapter.pressure_bar, delta_pressure, gravity, state
    )
    expected = []
    for layer in range(3):
        loga = log_hminus_continuum_single(
            nus,
            temperature[layer],
            state.number_density_e[layer],
            state.number_density_h[layer],
        )
        path_length = (
            chemistry.kB
            * temperature[layer]
            / (chemistry.m_u * state.mmw[layer] * gravity)
            * 0.2
        )
        expected.append(10**loga * path_length)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert actual.shape == (3, 7)
    assert np.all(np.isfinite(actual)) and np.all(actual > 0)


def test_packaged_fastchem4_ion_table_smoke():
    """Exercise real ions; this small profile does not certify retrieval priors."""
    from exogibbs.presets.fastchem4 import chemsetup
    from exojax.database.molinfo import element_mass

    setup = chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    adapter = chemistry.prepare_chemistry(
        setup,
        jnp.array([0.01, 0.1, 1.0]),
        element_masses_u=element_mass,
        isotope_convention="ExoJAX neutral atom mass table; bulk elemental chemistry",
        required_species=("H2", "He1", "H1+", "H1-", "H2O1", "C1O1"),
        electron_species="e1-",
        atomic_hydrogen_species="H1",
        temperature_range=(1500.0, 4500.0),
        pressure_range=(1e-5, 100.0),
        epsilon_crit=1e-14,
        conservation_rtol=1e-7,
    )
    run = jax.jit(adapter)
    with jax.checking_leaks():
        state = run(jnp.array([2200.0, 2800.0, 3500.0]), 0.0, 0.6)
        changed = run(jnp.array([2200.0, 2800.0, 3500.0]), 0.2, 0.8)
    assert np.all(state.diagnostics.valid), state.diagnostics
    assert np.all(changed.diagnostics.valid), changed.diagnostics
    assert np.all(state.number_density_e > 0)
    assert np.all(state.number_density_h > 0)
    assert not np.allclose(state.x, changed.x)
