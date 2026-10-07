"""Named molecular optical depths and their differentiable runtime inputs."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.opacity import OpaPremodit
from exojax.opacity.multimol import build_premodit
from exojax.rt import layer_optical_depth_multi
from exojax.rt.layeropacity import layer_optical_depth
from exojax.utils.constants import opacity_factor
from exojax.utils.grids import wavenumber_grid


def _opa(mass, cross_section):
    spectrum = jnp.asarray(cross_section)
    return SimpleNamespace(
        ready=True,
        nu_grid=np.array([1000.0, 1001.0, 1002.0]),
        molmass=mass,
        xsmatrix=lambda temperature, pressure: jnp.broadcast_to(
            spectrum, (temperature.shape[0], spectrum.size)
        ),
    )


@pytest.fixture
def opas():
    return {
        "H2O": _opa(18.0, [1.0e-23, 2.0e-23, 3.0e-23]),
        "CO": _opa(28.0, [4.0e-23, 2.0e-23, 1.0e-23]),
    }


@pytest.fixture
def inputs():
    return {
        "temperature": jnp.array([850.0, 1100.0]),
        "pressure": jnp.array([0.3, 1.0]),
        "dpressure": jnp.array([0.1, 0.6]),
        "mmr": {"H2O": 1.0e-3, "CO": jnp.array([2.0e-3, 3.0e-3])},
        "gravity": 1000.0,
    }


def test_named_mixture_matches_mmr_formula_and_name_order(opas, inputs):
    inputs["gravity"] = jnp.array([1000.0, 900.0])
    actual = layer_optical_depth_multi(opas, **inputs)
    expected = (
        opacity_factor
        * (inputs["dpressure"] / inputs["gravity"])[:, None]
        * (
            inputs["mmr"]["H2O"] / 18.0 * jnp.array([1.0e-23, 2.0e-23, 3.0e-23])
            + inputs["mmr"]["CO"][:, None]
            / 28.0 * jnp.array([4.0e-23, 2.0e-23, 1.0e-23])
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14)
    reordered = {name: opas[name] for name in reversed(opas)}
    reversed_mmr = dict(reversed(list(inputs["mmr"].items())))
    np.testing.assert_array_equal(
        layer_optical_depth_multi(reordered, **{**inputs, "mmr": reversed_mmr}),
        actual,
    )


def test_single_species_matches_existing_layer_optical_depth(opas, inputs):
    opa = opas["H2O"]
    inputs["mmr"] = {"H2O": inputs["mmr"]["H2O"]}
    actual = layer_optical_depth_multi({"H2O": opa}, **inputs)
    expected = layer_optical_depth(
        inputs["dpressure"], opa.xsmatrix(inputs["temperature"], inputs["pressure"]),
        inputs["mmr"]["H2O"], opa.molmass, inputs["gravity"],
    )
    np.testing.assert_array_equal(actual, expected)


def test_zero_abundance_preserves_jitted_absorption_gradient(opas, inputs):
    def objective(abundance):
        return layer_optical_depth_multi(
            opas, **{**inputs, "mmr": {"H2O": 0.0, "CO": abundance}}
        ).sum()

    derivative = jax.jit(jax.grad(objective))
    assert objective(0.0) == 0.0
    assert derivative(0.0) > 0.0
    np.testing.assert_allclose(derivative(0.0), derivative(0.2), rtol=1.0e-14)
    np.testing.assert_allclose(derivative(0.0), objective(1.0), rtol=1.0e-14)


def test_all_empty_line_selections_preserve_shapes_and_zero_gradients(inputs):
    databases = {
        name: MDBSnapshot(
            meta=MDBMeta(
                dbtype="exomol", molmass=mass,
                T_gQT=np.array([300.0, 1000.0, 2000.0]),
                gQT=np.array([1.0, 2.0, 4.0]),
            ),
            lines=Lines(
                nu_lines=np.array([]), elower=np.array([]),
                line_strength_ref_original=np.array([]),
            ),
        )
        for name, mass in [("H2O", 18.0), ("CO", 28.0)]
    }
    opas = build_premodit(
        databases, np.array([1000.0, 1001.0, 1002.0]),
        on_empty="zero", auto_trange=(500.0, 1500.0),
    )

    def optical_depth(state):
        return layer_optical_depth_multi(opas, **state)

    np.testing.assert_array_equal(jax.jit(optical_depth)(inputs), np.zeros((2, 3)))
    gradient = jax.jit(jax.grad(lambda state: optical_depth(state).sum()))(inputs)
    for leaf in jax.tree.leaves(gradient):
        np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_cross_sections_are_not_clipped_or_made_positive(opas, inputs):
    opas = {"H2O": _opa(18.0, [-1.0e-23, 0.0, 1.0e-23])}
    inputs["mmr"] = {"H2O": 0.3}
    result = layer_optical_depth_multi(opas, **inputs)
    assert jnp.all(result[:, 0] < 0.0)
    np.testing.assert_array_equal(result[:, 1], jnp.zeros(2))
    np.testing.assert_array_equal(result[:, 0], -result[:, 2])


@pytest.mark.parametrize("mmr", [{"H2O": 0.1}, {"H2O": 0.1, "CO": 0.2, "CH4": 0.3}])
def test_missing_or_extra_abundance_names_raise(opas, inputs, mmr):
    with pytest.raises(ValueError, match="mmr keys must exactly match"):
        layer_optical_depth_multi(opas, **{**inputs, "mmr": mmr})


def test_non_mapping_abundances_raise(opas, inputs):
    with pytest.raises(TypeError, match="mmr must be a mapping"):
        layer_optical_depth_multi(opas, **{**inputs, "mmr": jnp.array([0.1, 0.2])})


@pytest.mark.parametrize(
    "name,value",
    [
        ("temperature", 850.0),
        ("temperature", jnp.array([])),
        ("temperature", jnp.ones((2, 1))),
        ("pressure", 1.0),
        ("pressure", jnp.ones(3)),
        ("dpressure", jnp.ones((2, 1))),
        ("gravity", jnp.ones(3)),
        ("gravity", jnp.ones((2, 1))),
        ("mmr", {"H2O": jnp.ones(1), "CO": 0.2}),
        ("mmr", {"H2O": jnp.ones((2, 1)), "CO": 0.2}),
    ],
)
def test_invalid_runtime_shapes_raise(opas, inputs, name, value):
    with pytest.raises(ValueError, match=name):
        layer_optical_depth_multi(opas, **{**inputs, name: value})


@pytest.mark.parametrize("shape", [(3,), (1, 3), (2, 4), (2, 2, 3)])
def test_invalid_cross_section_shape_raises(opas, inputs, shape):
    opas["CO"].xsmatrix = lambda temperature, pressure: jnp.zeros(shape)
    with pytest.raises(ValueError, match="xsmatrix for 'CO' must have shape"):
        layer_optical_depth_multi(opas, **inputs)


def test_same_length_mismatched_grids_are_rejected_during_tracing(opas, inputs):
    opas["CO"].nu_grid = opas["CO"].nu_grid + 0.1
    with pytest.raises(ValueError, match="grid"):
        jax.jit(lambda state: layer_optical_depth_multi(opas, **state))(inputs)


def test_ckd_calculator_is_rejected(opas, inputs):
    opas["CO"] = SimpleNamespace(
        ready=True, method="ckd", molmass=28.0, nu_grid=opas["H2O"].nu_grid,
        xstensor_ckd=lambda temperature, pressure: jnp.ones((2, 4, 3)),
    )
    with pytest.raises((TypeError, ValueError), match="CKD|xsmatrix|line-by-line"):
        layer_optical_depth_multi(opas, **inputs)


def _premodit_pair():
    nu_grid, _, _ = wavenumber_grid(
        995.0, 1005.0, 32, unit="cm-1", xsmode="premodit"
    )
    opas = {}
    for name, mass, shift, strength in [
        ("H2O", 18.0, 0.0, 1.0), ("CO", 28.0, 0.6, 2.0)
    ]:
        snapshot = MDBSnapshot(
            meta=MDBMeta(
                dbtype="exomol", molmass=mass,
                T_gQT=np.array([300.0, 1000.0, 2000.0]),
                gQT=np.array([1.0, 2.0, 4.0]),
            ),
            lines=Lines(
                nu_lines=np.array([999.0, 1000.0, 1002.0]) + shift,
                elower=np.array([20.0, 350.0, 900.0]),
                line_strength_ref_original=np.array([2e-23, 4e-23, 3e-23]) * strength,
            ),
            n_Texp=np.array([0.5, 0.5, 0.5]),
            alpha_ref=np.array([0.06, 0.06, 0.06]),
        )
        opa = OpaPremodit.from_snapshot(
            snapshot, nu_grid, diffmode=0,
            broadening_resolution={"mode": "single", "value": (0.06, 0.5)},
        )
        opa.manual_setting(dE=300.0, Tref=1000.0, Twt=1200.0, Tmin=500.0, Tmax=1800.0)
        opas[name] = opa
    return opas


def test_real_premodit_mixture_jitted_gradients_match_finite_differences(inputs):
    opas = _premodit_pair()

    def objective(state):
        dtau = layer_optical_depth_multi(opas, **state)
        return jnp.sum(dtau * jnp.linspace(0.5, 1.5, dtau.shape[1]))

    value, gradient = jax.jit(jax.value_and_grad(objective))(inputs)
    flat, unflatten = ravel_pytree(inputs)
    gradient_flat, _ = ravel_pytree(gradient)
    differences = []
    for index in range(flat.size):
        step = 1.0e-4 * abs(flat[index])
        offset = jnp.zeros_like(flat).at[index].set(step)
        differences.append(
            (objective(unflatten(flat + offset)) - objective(unflatten(flat - offset)))
            / (2.0 * step)
        )

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient_flat))
    assert jnp.all(jnp.abs(gradient_flat) > 0.0)
    np.testing.assert_allclose(gradient_flat, differences, rtol=5.0e-5, atol=1.0e-10)
