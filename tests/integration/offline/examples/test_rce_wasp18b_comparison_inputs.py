"""Numerical comparison identity is shared and excludes numerical table pins."""

from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest

from tests.integration.offline.examples.test_rce_wasp18b_streaming import modules


def inputs(modules):
    manifest = SimpleNamespace(config={
        "stellar": {"asset": "star", "format": "npz"},
        "cia": {"H2-H2": {"asset": "cia_h2"}, "H2-He": {"asset": "cia_he"}},
        "opacity": {"ckd": {"H2O1": {"asset": "water_ckd"}},
                    "mapping": {"H2O1": {"isotope_fraction": .997, "broadening": "H2"}},
                    "omitted_absorbers": ["Fe1"], "scattering": "omitted",
                    "broadening_limitations": ["Fixed composition"]}},
        sha256={"__manifest__": "original", "star": "star-hash", "cia_h2": "h2-hash",
                "cia_he": "he-hash", "water_ckd": "original-ckd"})
    column = SimpleNamespace(settings=modules.column.ColumnSettings(18000., 200.),
        entropy=SimpleNamespace(thermodynamics=SimpleNamespace(metadata={"model": "pinned"})))
    return manifest, column


def run(identity):
    return dict(success=True, observation_sha256={"observation": "same"}, row_indices=[0, 1],
                parameters=[0., .6, -.7], mode="rce", physical_inputs=identity,
                prediction=[.001, .002], uncertainty=[.0001, .0001],
                pressure_bar=[.1, 1.], temperature_K=[2000., 2500.],
                broadband_flux={"net": [1., 1., 1.]})


def test_forward_and_validation_share_canonical_identity_for_numerical_tables(modules):
    forward = importlib.import_module("rce_wasp18b_forward")
    validation = importlib.import_module("rce_wasp18b_validation")
    assert validation.physical_inputs is forward.physical_inputs
    manifest, column = inputs(modules)
    original = forward.physical_inputs(manifest, column, 700.)
    changed = deepcopy(manifest)
    changed.sha256.update(__manifest__="refined-manifest", water_ckd="refined-ckd")
    refined = validation.physical_inputs(changed, column, 700.)
    assert original == refined and "asset_sha256" not in refined
    assert forward.compare_runs(run(refined), run(original))["passed"]


@pytest.mark.parametrize("changed_asset", ["star", "cia_h2", "cia_he"])
def test_changed_star_or_cia_pin_rejects_numerical_comparison(modules, changed_asset):
    forward = importlib.import_module("rce_wasp18b_forward")
    manifest, column = inputs(modules)
    original = forward.physical_inputs(manifest, column, 700.)
    changed = deepcopy(manifest)
    changed.sha256[changed_asset] = "different"
    current = forward.physical_inputs(changed, column, 700.)
    with pytest.raises(ValueError, match="identical physical inputs"):
        forward.compare_runs(run(current), run(original))


def test_changed_absorber_mapping_requires_physics_comparison(modules):
    forward = importlib.import_module("rce_wasp18b_forward")
    manifest, column = inputs(modules)
    original = forward.physical_inputs(manifest, column, 700.)
    changed = deepcopy(manifest)
    changed.config["opacity"]["mapping"]["H2O1"]["broadening"] = "H2+He"
    current = forward.physical_inputs(changed, column, 700.)
    with pytest.raises(ValueError, match="identical physical inputs"):
        forward.compare_runs(run(current), run(original))
    assert forward.compare_runs(run(current), run(original), kind="physics")["kind"] == "physics"
