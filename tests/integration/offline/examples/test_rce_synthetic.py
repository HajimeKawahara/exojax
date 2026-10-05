"""Coupled chemistry/radiation/convection checks on the small synthetic column."""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def column():
    pytest.importorskip("exogibbs.thermo.standard")
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
            yield importlib.import_module("_rce_synthetic").SyntheticColumn()
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_rce_energy_stability_and_convective_transport(column):
    parameters = jnp.array([0.0, 0.6, 0.0])
    result = column.solve(parameters)
    state = result.state
    assert state.converged and state.physics_valid and result.derivative_valid
    np.testing.assert_array_equal(state.convective_mask, [False, False, True])
    # Check the physical balance itself, including the imposed internal flux.
    np.testing.assert_allclose(
        state.radiative_flux + state.convective_flux,
        column.internal_flux,
        rtol=0,
        atol=0.01,
    )
    assert np.all(np.asarray(state.stability_residual)[~state.convective_mask] < 0)
    np.testing.assert_allclose(
        np.asarray(state.stability_residual)[state.convective_mask],
        0.0,
        atol=1e-9,
    )
    assert np.all(np.asarray(state.convective_flux) >= -0.01)
    assert state.convective_flux[-1] > 0.01
    assert np.all(np.isfinite(column.rce_spectrum_from_state(state, parameters)))


def test_composition_changes_opacity_and_spectrum_at_fixed_temperature(column):
    nodes = column.tce_nodes(jnp.array([0.0, 0.6, np.log(0.8), np.log(1.2)]))

    def opacity_and_spectrum(composition):
        chemistry = column.chemistry(nodes, composition[0], composition[1])
        return (
            column.optical_depth(nodes[:-1], chemistry),
            column.spectrum(nodes[:-1], nodes[-1], chemistry),
        )

    derivative = jax.jit(jax.jacfwd(opacity_and_spectrum))(jnp.array([0.0, 0.6]))
    for quantity in derivative:
        assert np.all(np.isfinite(quantity))
        # Both metallicity and C/O must reach the observation through opacity.
        assert np.all(
            np.linalg.norm(np.asarray(quantity).reshape(-1, 2), axis=0) > 1e-8
        )


def test_chemistry_precision_from_independent_cold_initialization(column):
    from exogibbs.api.equilibrium import (
        EquilibriumInit,
        EquilibriumOptions,
        equilibrium_profile,
    )
    from _rce_entropy import specific_entropy

    chemistry = column.chemistry
    provider = column.entropy.thermodynamics
    charge_index = chemistry.elements.index("e-")
    element_mask = np.arange(len(chemistry.elements)) != charge_index
    matrix = chemistry.formula_matrix

    def initialize(request):
        # A separate, deterministic mass-action start with a positive electron
        # seed; each layer starts independently, without a previous solution.
        abundances = request.b.at[charge_index].set(1e-8)
        total = jnp.sum(abundances)
        log_amounts = (
            jnp.log(total / request.P)
            + matrix.T @ jnp.log(abundances * request.P / total)
            - request.setup.hvector_func(request.T)
        )
        limits = jnp.min(
            jnp.where(
                matrix[element_mask] > 0,
                abundances[element_mask, None] / jnp.maximum(matrix[element_mask], 1),
                jnp.inf,
            ),
            axis=0,
        )
        limits = jnp.where(jnp.isfinite(limits), limits, 1.0)
        log_amounts = jnp.maximum(jnp.minimum(log_amounts, jnp.log(limits)), -600.0)
        return EquilibriumInit(log_amounts, jax.scipy.special.logsumexp(log_amounts))

    alternate = jax.jit(
        lambda nodes, b: equilibrium_profile(
            provider.chemical_setup,
            nodes,
            column.nodes,
            b,
            initializer=initialize,
            options=EquilibriumOptions(
                epsilon_crit=column.chemistry_epsilon, method="vmap_cold"
            ),
            return_diagnostics=True,
        )
    )
    # These states exposed the trace-element conservation floor during P6.
    points = [(jnp.array([2100.0, 2500.0, 3200.0, 3700.0]), 0.01, 0.59)]
    for parameters in [
        [
            -0.07278051546066024,
            0.6374352601943145,
            -0.2686788022160997,
            0.2038620922312634,
        ],
        [
            0.10911991080777626,
            0.6058940315037551,
            -0.19945184781983882,
            0.17839028807078308,
        ],
        [
            0.0006302410160927904,
            0.6138782942651462,
            -0.2355142730112147,
            0.1891066154312731,
        ],
    ]:
        points.append((column.tce_nodes(jnp.array(parameters)), *parameters[:2]))
    for nodes, metallicity, c_over_o in points:
        reference = column.entropy(nodes, metallicity, c_over_o)
        state = reference.chemistry
        result, diagnostics = alternate(nodes, state.b)
        assert np.all(reference.valid) and np.all(diagnostics["converged"])
        element_error = jnp.abs(
            result.n @ matrix[element_mask].T - state.b[element_mask]
        )
        assert np.max(element_error / state.b[element_mask]) <= column.conservation_rtol
        charge_error = jnp.abs(result.n @ chemistry.charges) / (
            result.n @ jnp.abs(chemistry.charges)
        )
        assert np.max(charge_error) <= column.conservation_rtol
        other = state._replace(
            n=result.n, x=result.x, mmw=result.x @ chemistry.masses_u
        )
        np.testing.assert_allclose(
            column.spectrum(nodes[:-1], nodes[-1], state),
            column.spectrum(nodes[:-1], nodes[-1], other),
            rtol=0,
            atol=0.1 * 2e-6,
        )
        entropy = specific_entropy(
            result.n,
            column.nodes,
            chemistry.masses_u,
            jax.vmap(provider.standard_entropy_r)(nodes),
            conserved_mass_kg=state.b @ chemistry.element_masses_u * 1e-3,
        )
        np.testing.assert_allclose(
            entropy, reference.specific_entropy, rtol=0, atol=1e-6
        )


@pytest.mark.parametrize("mode", ["tce", "rce"])
def test_likelihood_first_derivatives_rejection_and_determinism(column, mode):
    truth = jnp.array(
        [0.0, 0.6, 0.0] if mode == "rce" else [0.0, 0.6, np.log(0.8), np.log(1.2)]
    )
    spectrum = getattr(column, mode + "_spectrum")
    sigma = jnp.full(column.W.shape[0], 1e-5)
    observed = spectrum(truth) + sigma * jnp.linspace(-0.3, 0.4, sigma.size)
    lower = jnp.full(truth.shape, -1.0)
    upper = jnp.full(truth.shape, 1.0)
    log_prob = column.log_prob(mode, observed, sigma, lower, upper)
    traces = []

    def traced_log_prob(p):
        traces.append(None)
        return log_prob(p)

    value_and_grad = jax.jit(jax.value_and_grad(traced_log_prob, has_aux=True))
    parameters = truth + jnp.array(
        [0.01, -0.01, 0.005] if mode == "rce" else [0.01, -0.01, 0.005, -0.005]
    )
    (value, diagnostics), gradient = value_and_grad(parameters)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert int(diagnostics.status if mode == "rce" else diagnostics) == 0

    scalar = lambda p: log_prob(p)[0]
    directional = jax.jit(lambda p, d: jax.jvp(scalar, (p,), (d,))[1])
    for axis in range(truth.size):
        direction = jnp.eye(truth.size)[axis]
        # value_and_grad is the scalar VJP; compare with the forward JVP too.
        np.testing.assert_allclose(
            directional(parameters, direction),
            gradient[axis],
            rtol=2e-8,
            atol=2e-7,
        )
        for step in (1e-4, 3e-5):
            plus = parameters + step * direction
            minus = parameters - step * direction
            finite_difference = (scalar(plus) - scalar(minus)) / (2 * step)
            np.testing.assert_allclose(
                finite_difference,
                gradient[axis],
                rtol=1e-4,
                atol=2e-3,
            )

    prior_excluded = parameters.at[0].set(-2.0)
    chemistry_excluded = parameters.at[1].set(-0.2)
    points = jnp.stack(
        [parameters, prior_excluded, chemistry_excluded, truth, parameters]
    )
    (values, statuses), gradients = jax.jit(
        lambda p: jax.lax.map(jax.value_and_grad(log_prob, has_aux=True), p)
    )(points)
    np.testing.assert_array_equal(
        statuses.status if mode == "rce" else statuses, [0, 1, 2, 0, 0]
    )
    assert np.all(np.isneginf(values[1:3]))
    np.testing.assert_array_equal(gradients[1:3], 0.0)
    np.testing.assert_allclose(np.asarray(values)[[0, 4]], value, rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(gradients)[[0, 4]], jnp.stack([gradient, gradient]), rtol=0, atol=0
    )

    trace_count = len(traces)
    assert trace_count == 1
    for point in (truth, chemistry_excluded, parameters):
        repeated = value_and_grad(point)
    np.testing.assert_array_equal(repeated[0][0], value)
    np.testing.assert_array_equal(repeated[1], gradient)
    assert len(traces) == trace_count
