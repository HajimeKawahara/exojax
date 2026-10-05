"""Physical checks for conserved-mass entropy and the optional RCE adapter."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_rce_entropy",
    Path(__file__).resolve().parents[3] / "examples" / "_rce_entropy.py",
)
entropy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = entropy
_SPEC.loader.exec_module(entropy)


@pytest.fixture(autouse=True)
def x64():
    with jax.experimental.enable_x64():
        yield


def test_constant_cp_adiabat_and_pressure_temperature_derivatives():
    n = jnp.array([0.5, 0.1])
    masses = jnp.array([2.0, 4.0])
    cp_over_r = jnp.array([3.5, 2.5])
    reference_s = jnp.array([10.0, 8.0])
    pressure = jnp.array([0.01, 0.1, 1.0])
    adiabatic_gradient = n.sum() / (n @ cp_over_r)
    temperature = 600.0 * (pressure / pressure[0]) ** adiabatic_gradient
    s = entropy.specific_entropy(
        jnp.broadcast_to(n, (3, 2)),
        pressure,
        masses,
        reference_s + jnp.log(temperature[:, None] / 600.0) * cp_over_r,
    )
    np.testing.assert_allclose(s, jnp.full(3, s[0]), rtol=1e-13)
    for excess in (-0.02, 0.02):
        shifted_temperature = temperature * (pressure / pressure[0]) ** excess
        shifted_entropy = entropy.specific_entropy(
            jnp.broadcast_to(n, (3, 2)),
            pressure,
            masses,
            reference_s
            + jnp.log(shifted_temperature[:, None] / 600.0) * cp_over_r,
        )
        assert np.all(np.sign(jnp.diff(shifted_entropy)) == np.sign(excess))

    def scalar_entropy(log_tp):
        t, p = jnp.exp(log_tp)
        return entropy.specific_entropy(
            n, p, masses, reference_s + jnp.log(t / 600.0) * cp_over_r
        )

    derivative = jax.jit(jax.grad(scalar_entropy))(jnp.log(jnp.array([800.0, 0.1])))
    gas_constant = entropy.GAS_CONSTANT / (n @ masses * 1e-3)
    np.testing.assert_allclose(
        derivative, gas_constant * jnp.array([n @ cp_over_r, -n.sum()]), rtol=1e-13
    )


def test_conserved_mass_and_amount_normalization_during_dissociation():
    # One mole of H nuclei: molecule count changes but total mass stays fixed.
    molecular_amount = jnp.array([0.49, 0.25, 0.01])
    n = jnp.stack((1.0 - 2.0 * molecular_amount, molecular_amount), axis=-1)
    masses = jnp.array([1.00794, 2.01588])
    standard_s = jnp.array([[14.0, 18.0]] * 3)
    pressure = jnp.ones(3)
    s = entropy.specific_entropy(n, pressure, masses, standard_s)
    x = n / n.sum(axis=-1, keepdims=True)
    independently_normalized = (
        entropy.GAS_CONSTANT
        * jnp.sum(x * (standard_s - jnp.log(x)), axis=-1)
        / (x @ masses * 1e-3)
    )
    np.testing.assert_allclose(s, independently_normalized, rtol=1e-13)
    for factor in (0.001, 1e4):
        np.testing.assert_allclose(
            entropy.specific_entropy(
                factor * n,
                pressure,
                masses,
                standard_s,
                conserved_mass_kg=factor * 1.00794e-3,
            ),
            s,
            rtol=1e-13,
        )
    assert not np.allclose(s / s[0], n.sum(axis=-1) / n[0].sum())
    with pytest.raises(ValueError, match="Conserved mass"):
        entropy.specific_entropy(
            n, pressure, masses, standard_s, conserved_mass_kg=jnp.ones((3, 1))
        )


def test_exactly_absent_species_has_finite_entropy_and_physical_tangent():
    n = jnp.array([0.5, 0.0, 0.1])
    masses = jnp.array([2.0, 1.0, 4.0])
    cp_over_r = jnp.array([3.5, 2.5, 2.5])

    def evaluate(log_temperature):
        return entropy.specific_entropy(
            n, jnp.asarray(1.0), masses, 10.0 + cp_over_r * log_temperature
        )

    value, tangent = jax.jvp(
        jax.jit(evaluate), (jnp.asarray(0.2),), (jnp.asarray(1.0),)
    )
    assert np.isfinite(value) and np.isfinite(tangent)
    expected = entropy.GAS_CONSTANT * (n @ cp_over_r) / (n @ masses * 1e-3)
    assert tangent == pytest.approx(expected, rel=1e-13)
    assert jax.grad(evaluate)(0.2) == pytest.approx(expected, rel=1e-13)


def test_reference_pressure_conversion_leaves_entropy_unchanged():
    n, masses = jnp.array([0.5, 0.1]), jnp.array([2.0, 4.0])
    s0 = jnp.array([10.0, 8.0])
    base = entropy.specific_entropy(n, jnp.asarray(0.2), masses, s0)
    shifted = entropy.specific_entropy(
        n,
        jnp.asarray(0.2),
        masses,
        s0 - jnp.log(1.01325),
        reference_pressure_bar=1.01325,
    )
    assert shifted == pytest.approx(base, rel=1e-13)


@pytest.fixture
def toy_model():
    pytest.importorskip("exogibbs", reason="Optional RCE/TCE chemistry dependency")
    from dataclasses import replace
    from types import SimpleNamespace
    from exogibbs.api.chemistry import ChemicalSetup

    spec = importlib.util.spec_from_file_location(
        "_rce_chemistry",
        Path(__file__).resolve().parents[3] / "examples" / "_rce_chemistry.py"
    )
    chemistry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = chemistry
    spec.loader.exec_module(chemistry)
    matrix = jnp.array(
        [[1, 2, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, -1, 1]], dtype=float
    )
    cp = jnp.array([2.5, 3.5, 2.5, 2.5, 2.5, 3.5, 2.5, 2.5, 2.5])
    offsets = jnp.array([15, 20, 10, 15, 15, 20, 5, 10, 10], dtype=float)
    formation_h_over_r = jnp.array(
        [22000, 0, 0, 10000, 10000, 0, 0, 50000, 24000], dtype=float
    )

    def standard_s(t):
        return offsets + cp * jnp.log(t / 300.0)

    def standard_g(t):
        return formation_h_over_r / t + cp - standard_s(t)

    setup = ChemicalSetup(
        formula_matrix=matrix,
        hvector_func=standard_g,
        species=("H", "H2", "He", "C", "O", "CO", "e-", "H+", "H-"),
        elements=("H", "He", "C", "O", "e-"),
        element_vector_reference=jnp.array([1.0, 0.08, 3e-4, 5e-4, 0.0]),
    )
    provider = SimpleNamespace(
        chemical_setup=setup,
        species=setup.species,
        elements=setup.elements,
        hvector_func=standard_g,
        standard_gibbs_rt=standard_g,
        standard_entropy_r=standard_s,
        standard_cp_r=lambda t: cp,
        standard_pressure_bar=1.0,
        temperature_range=(500.0, 6000.0),
        isotope_convention="Test neutral-atom masses with electron corrections",
        element_masses_u={"H": 1.0, "He": 4.0, "C": 12.0, "O": 16.0},
    )
    adapter = chemistry.prepare_chemistry(
        setup,
        jnp.array([0.03, 0.4, 2.0]),
        element_masses_u=provider.element_masses_u,
        isotope_convention=provider.isotope_convention,
        required_species=setup.species,
        electron_species="e-",
        atomic_hydrogen_species="H",
        temperature_range=provider.temperature_range,
        pressure_range=(1e-5, 100.0),
        epsilon_crit=1e-13,
    )
    return adapter, provider, replace


def test_hydrogen_dissociation_entropy_matches_analytic_equilibrium():
    pytest.importorskip("exogibbs", reason="Optional RCE/TCE chemistry dependency")
    from exogibbs.api.chemistry import ChemicalSetup
    from exogibbs.api.equilibrium import equilibrium_profile, EquilibriumOptions

    cp, offsets = jnp.array([2.5, 3.5]), jnp.array([15.0, 20.0])
    formation_h_over_r = jnp.array([22000.0, 0.0])
    masses = jnp.array([1.0, 2.0])

    def standard_s(t):
        return offsets + cp * jnp.log(t / 300.0)

    def standard_g(t):
        return formation_h_over_r / t + cp - standard_s(t)

    setup = ChemicalSetup(jnp.array([[1.0, 2.0]]), standard_g)
    pressure = jnp.array([0.1])

    def evaluate(log_temperature):
        t = jnp.exp(log_temperature)[None]
        result = equilibrium_profile(
            setup, t, pressure, jnp.array([1.0]),
            options=EquilibriumOptions(method="vmap_cold", epsilon_crit=1e-13),
        )
        return entropy.specific_entropy(
            result.n, pressure, masses, jax.vmap(standard_s)(t), conserved_mass_kg=1e-3
        )[0]

    def reference(log_temperature):
        t = jnp.exp(log_temperature)
        g = standard_g(t)
        association = jnp.exp(2 * g[0] - g[1]) * pressure[0]
        atomic_amount = 1 / jnp.sqrt(1 + 4 * association)
        n = jnp.array([atomic_amount, (1 - atomic_amount) / 2])
        x = n / n.sum()
        return entropy.GAS_CONSTANT / 1e-3 * jnp.sum(
            n * (standard_s(t) - jnp.log(x * pressure[0]))
        )

    for t in (1800.0, 2800.0, 4000.0):
        point = jnp.log(t)
        actual, cp_eq = jax.jvp(evaluate, (point,), (jnp.asarray(1.0),))
        expected, expected_cp = jax.jvp(reference, (point,), (jnp.asarray(1.0),))
        assert actual == pytest.approx(expected, rel=1e-10)
        assert cp_eq == pytest.approx(expected_cp, rel=1e-8)
        assert cp_eq > 0
        assert jax.grad(evaluate)(point) == pytest.approx(cp_eq, rel=1e-10)
        for step in (1e-3, 3e-4):
            difference = (evaluate(point + step) - evaluate(point - step)) / (2 * step)
            assert cp_eq == pytest.approx(difference, rel=2e-5)


def test_entropy_adapter_preserves_dynamic_elements_and_first_derivatives(toy_model):
    chemistry, provider, _ = toy_model
    adapter = entropy.prepare_equilibrium_entropy(
        chemistry, provider, entropy_scale=1e4
    )
    temperatures = jnp.array([1800.0, 2800.0, 4000.0])
    point = jnp.array([0.0, 0.2, 0.8])
    direction = jnp.array([0.1, 0.3, 0.2])

    @jax.jit
    def run(p):
        return adapter(temperatures * jnp.exp(p[0]), p[1], p[2])

    with jax.checking_leaks():
        state = run(point)
        changed = run(point + 0.02 * direction)
    assert np.all(state.valid) and np.all(changed.valid)
    assert not np.allclose(state.specific_entropy, changed.specific_entropy)
    mass = state.chemistry.b @ chemistry.element_masses_u * 1e-3
    np.testing.assert_allclose(
        state.chemistry.n @ chemistry.masses_u * 1e-3, mass, rtol=1e-10
    )
    np.testing.assert_allclose(
        state.chemistry.n @ np.asarray([
            [1, 2, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, -1, 1],
        ]).T,
        jnp.broadcast_to(state.chemistry.b, (3, 5)), atol=1e-12,
    )
    objective = lambda p: run(p).specific_entropy.sum()
    _, tangent = jax.jvp(objective, (point,), (direction,))
    gradient = jax.jit(jax.grad(objective))(point)
    assert tangent == pytest.approx(gradient @ direction, rel=1e-10)
    for step in (1e-3, 3e-4):
        difference = (
            objective(point + step * direction) - objective(point - step * direction)
        ) / (2 * step)
        assert tangent == pytest.approx(difference, rel=1e-5)


def test_column_evaluator_shares_bottom_chemistry_and_guards_radiation(toy_model):
    from copy import copy

    chemistry, provider, replace = toy_model
    chemistry_calls, radiation_calls = [], []

    def counted_chemistry(t, metal, ratio):
        jax.debug.callback(lambda _: chemistry_calls.append(1), t[0])
        return chemistry(t, metal, ratio)

    counted = replace(chemistry, evaluate=counted_chemistry)
    adapter = entropy.prepare_equilibrium_entropy(counted, provider, entropy_scale=1e4)

    def radiation(t, bottom_t, parameters, state):
        jax.debug.callback(lambda _: radiation_calls.append(1), t[0])
        return state.mmw.astype(jnp.float32)

    evaluator = entropy.make_entropy_evaluator(
        adapter, radiation, lambda p: (p[0], p[1])
    )
    temperatures = jnp.array([1800.0, 2800.0], dtype=jnp.float32)
    params = jnp.array([0.2, 0.8])
    column = jax.jit(evaluator)(temperatures, 4000.0, params)
    jax.block_until_ready(column)
    assert chemistry_calls == [1] and radiation_calls == [1]
    assert column.net_flux.dtype == chemistry.pressure_bar.dtype
    direct = adapter(jnp.append(temperatures, 4000.0), *params)
    np.testing.assert_allclose(column.net_flux, direct.chemistry.mmw)
    np.testing.assert_allclose(
        column.stability_excess,
        jnp.diff(direct.specific_entropy)
        / (1e4 * jnp.diff(jnp.log(chemistry.pressure_bar))),
    )
    assert column.physics_valid and column.physics_status == entropy.SUCCESS
    def objective(p):
        return evaluator(
            temperatures * jnp.exp(p[0]), 4000.0, p[1:]
        ).stability_excess.sum()

    point = jnp.array([0.0, 0.2, 0.8])
    assert np.all(np.isfinite(jax.jit(jax.grad(objective))(point)))

    bad_provider = copy(provider)
    bad_provider.standard_cp_r = lambda t: provider.standard_cp_r(t).at[-1].set(-1.0)
    invalid_adapter = entropy.prepare_equilibrium_entropy(
        counted, bad_provider, entropy_scale=1e4
    )
    rejected = entropy.make_entropy_evaluator(
        invalid_adapter, radiation, lambda p: (p[0], p[1])
    )
    radiation_calls.clear()
    failure = jax.jit(rejected)(temperatures, 4000.0, params)
    jax.block_until_ready(failure)
    assert not failure.physics_valid
    assert failure.physics_status == entropy.NONPOSITIVE_HEAT_CAPACITY
    assert radiation_calls == []
    assert np.all(np.isnan(failure.net_flux))

    radiation_calls.clear()
    mapped = jax.jit(
        lambda ts: jax.lax.map(lambda t: evaluator(t[:2], t[2], params), ts)
    )
    mixed = mapped(jnp.array([[1800.0, 2800.0, 4000.0], [100.0, 2800.0, 4000.0]]))
    jax.block_until_ready(mixed)
    np.testing.assert_array_equal(mixed.physics_valid, [True, False])
    assert mixed.physics_status[1] == entropy.INVALID_CHEMISTRY
    assert radiation_calls == [1]


def test_entropy_contract_rejects_mismatched_or_incomplete_standard_states(toy_model):
    from copy import copy

    chemistry, provider, replace = toy_model
    for key, value, match in (
        ("species", provider.species[:-1], "same setup"),
        ("hvector_func", lambda t: provider.hvector_func(t), "same setup"),
        ("standard_pressure_bar", 1.01325, "1 bar"),
        ("isotope_convention", "different", "isotope convention"),
        ("element_masses_u", {"H": 2.0}, "atomic mass"),
        ("temperature_range", (1000.0, 5000.0), "domain"),
    ):
        invalid = copy(provider)
        setattr(invalid, key, value)
        with pytest.raises(ValueError, match=match):
            entropy.prepare_equilibrium_entropy(chemistry, invalid, entropy_scale=1e4)
    with pytest.raises(ValueError, match="entropy_scale"):
        entropy.prepare_equilibrium_entropy(chemistry, provider, entropy_scale=0)
    with pytest.raises(ValueError, match="device dtype"):
        entropy.prepare_equilibrium_entropy(chemistry, provider, entropy_scale=1e-320)
    with pytest.raises(ValueError, match="increasing pressure"):
        entropy.prepare_equilibrium_entropy(
            replace(chemistry, pressure_bar=jnp.array([0.1, 0.01, 2.0])),
            provider,
            entropy_scale=1e4,
        )

    missing = copy(provider)
    missing.standard_entropy_r = (
        lambda t: provider.standard_entropy_r(t).at[-1].set(jnp.nan)
    )
    invalid_adapter = entropy.prepare_equilibrium_entropy(
        chemistry, missing, entropy_scale=1e4
    )
    invalid = invalid_adapter(jnp.array([1800.0, 2800.0, 4000.0]), 0.0, 0.6)
    np.testing.assert_array_equal(invalid.status, entropy.INVALID_THERMODYNAMICS)
    assert not np.any(invalid.valid)

    def negative_amount(t, metal, ratio):
        state = chemistry(t, metal, ratio)
        return state._replace(n=state.n.at[0, 0].set(-1.0))

    invalid = entropy.prepare_equilibrium_entropy(
        replace(chemistry, evaluate=negative_amount), provider, entropy_scale=1e4
    )(jnp.array([1800.0, 2800.0, 4000.0]), 0.0, 0.6)
    assert invalid.status[0] == entropy.INVALID_CHEMISTRY


def test_stoichiometry_and_mass_metadata_follow_the_actual_prepared_setup(toy_model):
    from copy import copy

    chemistry, provider, replace = toy_model
    prepare = sys.modules["_rce_chemistry"].prepare_chemistry

    def prepare_with(setup, masses):
        return prepare(
            setup,
            chemistry.pressure_bar,
            element_masses_u=masses,
            isotope_convention=provider.isotope_convention,
            required_species=provider.species,
            electron_species="e-",
            atomic_hydrogen_species="H",
            temperature_range=provider.temperature_range,
            pressure_range=chemistry.pressure_range,
        )

    # Same names and chemical-potential function do not certify stoichiometry:
    # this altered CO column remains full rank and passes P2's gas setup checks.
    wrong_setup = replace(
        provider.chemical_setup,
        formula_matrix=provider.chemical_setup.formula_matrix.at[2, 5].set(2.0),
    )
    wrong = prepare_with(wrong_setup, provider.element_masses_u)
    with pytest.raises(ValueError, match="same setup"):
        entropy.prepare_equilibrium_entropy(wrong, provider, entropy_scale=1e4)

    # Coherent masses must compare in their device representation: converting
    # a noninteger mass to float32 is not an isotope-convention mismatch.
    rounded_provider = copy(provider)
    rounded_provider.element_masses_u = dict(provider.element_masses_u, H=1.00794)
    with jax.experimental.enable_x64(False):
        rounded = prepare_with(
            provider.chemical_setup, rounded_provider.element_masses_u
        )
        assert rounded.element_masses_u.dtype == jnp.float32
        entropy.prepare_equilibrium_entropy(
            rounded, rounded_provider, entropy_scale=1e4
        )


def _prepare_full_model(pressure):
    standard = pytest.importorskip(
        "exogibbs.thermo.standard", reason="Requires ExoGibbs standard-state provider"
    )
    spec = importlib.util.spec_from_file_location(
        "_rce_chemistry",
        Path(__file__).resolve().parents[3] / "examples" / "_rce_chemistry.py"
    )
    chemistry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = chemistry
    spec.loader.exec_module(chemistry)
    provider = standard.prepare_fastchem_thermodynamics(
        temperature_range=(1500.0, 4500.0)
    )
    adapter = chemistry.prepare_chemistry(
        provider.chemical_setup,
        pressure,
        element_masses_u=provider.element_masses_u,
        isotope_convention=provider.isotope_convention,
        required_species=("H2", "He1", "H1+", "H1-", "H2O1", "C1O1"),
        electron_species="e1-",
        atomic_hydrogen_species="H1",
        temperature_range=provider.temperature_range,
        pressure_range=(1e-5, 100.0),
        epsilon_crit=1e-14,
        conservation_rtol=1e-7,
    )
    return entropy.prepare_equilibrium_entropy(adapter, provider, entropy_scale=1e4)


def test_full_standard_state_model_coverage_and_equilibrium_entropy_gradients():
    prepared = _prepare_full_model(jnp.array([0.03, 0.4, 2.0]))
    provider = prepared.thermodynamics
    assert provider.metadata["source_species_count"] == len(provider.species) + 1
    assert tuple(provider.metadata["excluded_species"]) == ("C4H6O4",)
    assert len(provider.metadata["nasa_replacements"]) == 34
    assert provider.metadata["minimum_standard_cp_r"] > 0
    temperatures = jnp.array([2200.0, 2800.0, 3500.0])
    standard_cp = provider.standard_cp_r(temperatures)
    assert standard_cp.shape == (3, len(provider.species))
    assert np.all(np.isfinite(standard_cp)) and np.all(standard_cp > 0)

    @jax.jit
    def run(p):
        return prepared(temperatures * jnp.exp(p[0]), p[1], p[2])

    point, direction = jnp.array([0.0, 0.0, 0.6]), jnp.array([0.1, 0.2, 0.3])
    with jax.checking_leaks():
        state = run(point)
        changed = run(point + 0.01 * direction)
    assert np.all(state.valid), state.chemistry.diagnostics
    assert np.all(changed.valid), changed.chemistry.diagnostics
    assert state.chemistry.n.shape[-1] == len(provider.species)
    assert np.all(state.chemistry.number_density_e > 0)
    assert not np.allclose(state.specific_entropy, changed.specific_entropy)
    objective = lambda p: run(p).specific_entropy.sum()
    _, tangent = jax.jvp(objective, (point,), (direction,))
    gradient = jax.jit(jax.grad(objective))(point)
    assert np.all(np.isfinite(gradient))
    assert tangent == pytest.approx(gradient @ direction, rel=1e-9)
    for step in (1e-3, 3e-4):
        difference = (
            objective(point + step * direction) - objective(point - step * direction)
        ) / (2 * step)
        assert tangent == pytest.approx(difference, rel=1e-5)


def test_full_model_convective_rce_implicit_derivatives_and_guarded_likelihood():
    # P4 and P5 are independently reviewable branches. This acceptance check
    # runs once the device implicit solver and standard-state provider coexist.
    implicit = pytest.importorskip(
        "exojax.atm.rce_device_implicit", reason="Requires P4 device implicit solver"
    )
    prepared = _prepare_full_model(jnp.array([1.0, 2.0]))
    evaluator = entropy.make_entropy_evaluator(
        prepared,
        lambda t, tb, p, state: jnp.array(
            [(t[0] / 2500.0) ** 4, 0.2 * (tb / 3000.0) ** 4]
        ),
        lambda p: (p[1], p[2]),
    )

    def domain(t, bottom_t, parameters):
        nodes = jnp.append(t, bottom_t)
        return jnp.all((nodes > 1500.0) & (nodes < 4500.0))

    solve = implicit.make_device_implicit_rce_solver(
        jnp.array([1.0]),
        jnp.array([0.5, 2.0]),
        jnp.array([2450.0]),
        2900.0,
        lambda p: jnp.exp(4 * p[0]),
        evaluator,
        valid_temperature=domain,
        local_smoothness=domain,
        convective_mask_initial=jnp.array([True]),
        flux_atol=1e-10,
        flux_rtol=0.0,
        stability_atol=1e-10,
        invalid_derivative="zero",
    )
    point, direction = jnp.array([0.0, 0.0, 0.6]), jnp.array([0.1, 0.2, 0.3])
    result = solve(point)
    assert result.state.converged and result.derivative_valid, result
    np.testing.assert_array_equal(result.state.convective_mask, [True])
    assert result.state.convective_flux[1] > 0
    objective = lambda p: solve(p).state.bottom_temperature
    _, tangent = jax.jvp(objective, (point,), (direction,))
    gradient = jax.jit(jax.grad(objective))(point)
    assert np.all(np.isfinite(gradient))
    assert tangent == pytest.approx(gradient @ direction, rel=1e-9)
    for step in (1e-3, 3e-4):
        plus, minus = solve(point + step * direction), solve(point - step * direction)
        assert plus.derivative_valid and minus.derivative_valid
        difference = (
            plus.state.bottom_temperature - minus.state.bottom_temperature
        ) / (2 * step)
        assert tangent == pytest.approx(difference, rel=1e-5)
    log_prob = implicit.make_rce_log_prob(
        solve,
        lambda state, p: -0.5 * ((state.bottom_temperature - 2900.0) / 100.0) ** 2,
        prior_valid=lambda p: (p[2] > 0) & (jnp.abs(p[0]) < 0.1),
    )
    value_and_grad = jax.jit(jax.value_and_grad(log_prob, has_aux=True))
    (value, diagnostics), derivative = value_and_grad(point)
    assert np.isfinite(value) and np.all(np.isfinite(derivative))
    assert diagnostics.status == implicit.LogProbStatus.ACCEPTED
    (rejected, diagnostics), derivative = value_and_grad(point.at[2].set(-1.0))
    assert np.isneginf(rejected) and np.all(derivative == 0)
    assert diagnostics.status == implicit.LogProbStatus.PRIOR_EXCLUDED
