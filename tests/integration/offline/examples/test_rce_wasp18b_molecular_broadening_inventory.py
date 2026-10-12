"""Verify fallback classification and collisional-width mixture conventions."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("rce_wasp18b_molecular_broadening_inventory")


def test_single_J0_and_unsupported_m0_remain_current_fallback(audit):
    rows = pd.DataFrame(dict(jlower=[0, 1, 2], jupper=[1, 2, 1]))
    a, n, explicit = audit.current_mapping(rows, {("a0", 0): (.146, .5)}, (.07, .5))
    np.testing.assert_array_equal(a, [.146, .07, .07])
    np.testing.assert_array_equal(explicit, [True, False, False])
    published = audit.documented_a0_tail(rows, {("a0", 0): (.146, .5)}, (.07, .5))
    np.testing.assert_array_equal(published[0], np.full(3, .146))
    a, n, explicit = audit.current_mapping(rows, {("m0", 1): (.15, .58)}, (.1063, .695))
    np.testing.assert_array_equal(a, np.full(3, .1063))
    assert not explicit.any()
    assert audit.documented_a0_tail(rows, {("m0", 1): (.15, .58)}, (.1063, .695)) is None


def test_a1_overrides_a0_but_other_J_uses_definition(audit):
    rows = pd.DataFrame(dict(jlower=[1, 1, 51], jupper=[2, 0, 52]))
    recipe = {("a0", 1): (.08, .6), ("a1", 1, 2): (.09, .7)}
    a, n, explicit = audit.current_mapping(rows, recipe, (.07, .5))
    np.testing.assert_array_equal(a, [.09, .08, .07])
    np.testing.assert_array_equal(n, [.7, .6, .5])
    np.testing.assert_array_equal(explicit, [True, True, False])


def test_known_mixture_uses_partial_pressure_and_different_temperature_laws(audit):
    t, p = np.array([296., 1184.]), np.array([1., 2.])
    fractions = np.array([[.6, .1, .3], [.8, .15, .05]])
    result = audit.recipe_comparison({("a0", 0): (.08, .5)}, {("a0", 0): (.04, 1.)}, t, p, fractions)
    np.testing.assert_allclose(np.array(result['known_over_pure_H2'])[:, 0], [.35, .0875], rtol=1e-15)
    np.testing.assert_allclose(np.array(result['gamma_known_H2_He_cm1'])[:, 0], [.028, .007], rtol=1e-15)
    assert result['H_coefficient'] is None
    with pytest.raises(ValueError, match='without renormalization'):
        audit.recipe_comparison({("a0", 0): (.08, .5)}, {("a0", 0): (.04, 1.)}, t, p, fractions * 2)
