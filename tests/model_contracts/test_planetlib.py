"""Read shared expressions through MCEG and compare the actual Planck kernels."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("mceg.schema_v06")
sp = pytest.importorskip("sympy")

from mceg import load_contract, mathematical_model_digest, to_sympy
from mceg.equation_checks import equation_checks
from mceg.integrity import file_integrity

from exojax.rt.planck import fac_planck, piB, piBarr
from exojax.utils.constants import ccgs, hcperk


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = load_contract(ROOT / "mceg/planetlib/contract.yaml")
MODEL = next(m for m in CONTRACT.mathematical_models if m.id == "exojax.planetlib.planck.flux")
EQUATIONS = {e.id: to_sympy(e.sympy) for e in CONTRACT.equations}


def flux_expression():
    """Substitute the four authored relations; maintain no separate formula."""
    values = {}
    for relation in MODEL.relations:
        equation = EQUATIONS[relation.equation.id]
        values[equation.lhs.name] = equation.rhs.xreplace(
            {s: values[s.name] for s in equation.rhs.free_symbols if s.name in values}
        )
    return values["piB"]


EXPRESSION = flux_expression()
SYMBOLS = {s.name: s for s in EXPRESSION.free_symbols}
ARGUMENTS = [SYMBOLS[name] for name in ("T", "wavenumber", "h", "c", "k")]
REFERENCE = sp.lambdify(ARGUMENTS, EXPRESSION, "numpy")
DT_REFERENCE = sp.lambdify(ARGUMENTS, sp.diff(EXPRESSION, SYMBOLS["T"]), "numpy")
DWN_REFERENCE = sp.lambdify(ARGUMENTS, sp.diff(EXPRESSION, SYMBOLS["wavenumber"]), "numpy")


@pytest.fixture(autouse=True)
def precision(isolated_test_environment):
    yield


def kernel_parameters():
    # Map the implementation's actual prefactors, not an unused constants.hcgs.
    effective_h = fac_planck / (2 * np.pi * ccgs**2)
    effective_k = effective_h * ccgs / hcperk
    return effective_h, ccgs, effective_k


def test_native_composition_and_pins():
    lock = json.loads((ROOT / "mceg/planetlib/imports.lock.json").read_text())
    for entry in lock["imports"]:
        pin = entry["model"]
        assert mathematical_model_digest(CONTRACT, pin["id"], pin["revision"]) == pin["sha256"]
    assert {r.equation.id.rsplit(".", 1)[-1] for r in MODEL.relations} == {
        "frequency_coordinate", "planck_frequency", "intensity_conversion", "hemisphere_planck"
    }
    assert {q.role for key, q in MODEL.quantities.items() if key in ("h", "c", "k")} == {"parameter"}
    for collection in (CONTRACT.subjects, CONTRACT.contexts, CONTRACT.claims, CONTRACT.assessments):
        assert all(record.id.startswith("exojax.") for record in collection)
    for assessment in CONTRACT.assessments:
        assert assessment.status == "unknown"
        checks = equation_checks(CONTRACT, assessment, root=ROOT)
        assert checks and all(c["status"] == "PASS" for c in checks)
        integrity = file_integrity(CONTRACT, assessment, set(assessment.evidence), ROOT)
        files = integrity["selected_files"] + integrity["artifact_hashes"]
        assert files and all(f["status"] == "PASS" for f in files)


@pytest.mark.parametrize("temperature", [100., 800., 2500.])
def test_flux_coordinate_jacobian_and_derivatives(temperature):
    # Positive finite points cover near Rayleigh-Jeans, intermediate and Wien regimes.
    wavenumber = jnp.asarray(np.array([.05, 2., 20., 100.]) * temperature / hcperk)
    parameters = kernel_parameters()
    expected = REFERENCE(temperature, np.asarray(wavenumber), *parameters)
    actual = piB(temperature, wavenumber)
    np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=0.)
    actual_dt = jax.jacrev(lambda t: piB(t, wavenumber))(temperature)
    actual_dwn = jnp.diag(jax.jacrev(lambda wn: piB(temperature, wn))(wavenumber))
    np.testing.assert_allclose(actual_dt, DT_REFERENCE(temperature, np.asarray(wavenumber), *parameters),
                               rtol=5e-13, atol=0.)
    np.testing.assert_allclose(actual_dwn, DWN_REFERENCE(temperature, np.asarray(wavenumber), *parameters),
                               rtol=5e-13, atol=0.)


def test_layer_and_spectral_axes():
    temperatures = jnp.array([350., 1000., 2200.])
    wavenumbers = jnp.array([100., 800., 2500., 6000.])
    expected = REFERENCE(np.asarray(temperatures)[:, None], np.asarray(wavenumbers), *kernel_parameters())
    actual = piBarr(temperatures, wavenumbers)
    assert actual.shape == (3, 4)
    np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=0.)
    for index, temperature in enumerate(temperatures):
        np.testing.assert_allclose(actual[index], piB(temperature, wavenumbers), rtol=4e-13, atol=0.)
    # Each layer depends only on its own temperature.
    derivatives = jax.jacrev(lambda t: piBarr(t, wavenumbers))(temperatures)
    for layer in range(3):
        for source_layer in range(3):
            target = DT_REFERENCE(float(temperatures[layer]), np.asarray(wavenumbers), *kernel_parameters()) if layer == source_layer else 0.
            np.testing.assert_allclose(derivatives[layer, :, source_layer], target, rtol=5e-13, atol=0.)
