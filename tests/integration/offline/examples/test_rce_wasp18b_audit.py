"""Independent limiting cases for the conditional scattering diagnostic."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "examples"))
    return importlib.import_module("_rce_wasp18b_audit")


def test_single_slab_competing_events(audit):
    absorption = np.array([[0., .7, 1e-14]])
    scattering = np.array([[0., .3, 1e-14]])
    actual = audit.first_scattering_probability(absorption, scattering, .5)
    expected = np.array([0., .3 * (1 - np.exp(-2.)), -np.expm1(-4e-14) / 2.])
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.)


def test_homogeneous_subdivision_and_pure_scattering(audit):
    single = audit.first_scattering_probability([[.7]], [[.3]], .2)
    divided = audit.first_scattering_probability(np.full((50, 1), .7 / 50),
                                                 np.full((50, 1), .3 / 50), .2)
    np.testing.assert_allclose(divided, single, rtol=2e-15)
    scattering = np.array([[.1, .2], [.4, .3]])
    pure = audit.first_scattering_probability(np.zeros_like(scattering), scattering, .7)
    np.testing.assert_allclose(pure, -np.expm1(-scattering.sum(axis=0) / .7))


def test_overlying_absorption_shields_scattering(audit):
    actual = audit.first_scattering_probability([[2.], [0.]], [[0.], [.4]], .5)
    np.testing.assert_allclose(actual, np.exp(-4.) * -np.expm1(-.8))


@pytest.mark.parametrize("absorption,scattering,mu0", [([-1.], [0.], .5),
    ([0.], [np.nan], .5), ([0.], [1.], 0.), ([0.], [1.], 1.1)])
def test_invalid_transfer_inputs(audit, absorption, scattering, mu0):
    with pytest.raises(ValueError):
        audit.first_scattering_probability(absorption, scattering, mu0)


def test_cross_sections_units_and_domain(audit):
    cross_sections = audit.scattering_cross_sections([1e4, 2e4])
    np.testing.assert_allclose(cross_sections["H1"][0], 5.943984e-29, rtol=1e-14)
    np.testing.assert_allclose(cross_sections["H2"][0], 8.26961e-29, rtol=1e-14)
    np.testing.assert_allclose(cross_sections["He1"][1] / cross_sections["He1"][0], 16.)
    np.testing.assert_array_equal(cross_sections["e1-"], [6.6524587321e-25] * 2)
    with pytest.raises(ValueError):
        audit.scattering_cross_sections([1e5])


def test_audit_stellar_weighting_and_chemical_mapping(audit):
    species = ("H1", "H2", "He1", "e1-", "H2O1", "Fe1")
    x = np.array([[0., 0., 0., 1e-4, 1e-4, 2e-5], [0., 0., 0., 1e-4, 1e-4, 2e-5]])
    chemistry = SimpleNamespace(x=x, mmw=np.ones(2),
                                diagnostics=SimpleNamespace(valid=np.ones(2, dtype=bool)))
    column = SimpleNamespace(boundaries=np.array([1., 2.]),
        settings=SimpleNamespace(gravity_cgs=1e4, mu0=.5),
        chemistry=SimpleNamespace(species=species), ckd={"H2O1": object()},
        band_nodes=np.array([[1e4], [2e4]]), band_weights=np.ones(1),
        g_weights=np.array([.25, .75]), star_bands=np.array([1., 3.]),
        band_widths=np.array([2., 1.]),
        optical_depth=lambda t,c: np.array([[[0., 1.], [2., 3.]]]))
    report = audit.audit_scattering(column, np.array([2500.]), chemistry)
    tau = 1e6 / (1e4 * audit.m_u) * 1e-4 * 6.6524587321e-25
    absorption = np.array([[0., 1.], [2., 3.]])
    expected = tau / (absorption + tau) * -np.expm1(-(absorption + tau) / .5)
    expected = np.array([.25, .75]) @ expected @ np.array([2., 3.]) / 5.
    np.testing.assert_allclose(report["incident_energy_redistribution_bound_fraction"], expected)
    assert report["certifies_rce_or_observation_error"] is False
    inventory = audit.chemical_opacity_inventory(column, chemistry)
    rows = {row["species"]: row for row in inventory["species"]}
    assert rows["Fe1"]["absorption_roles"] == []
    assert rows["H2O1"]["absorption_roles"] == ["line absorption"]
    assert inventory["omitted_absorption_error_bounded"] is False
    chemistry.diagnostics.valid[0] = False
    with pytest.raises(ValueError, match="valid chemistry"):
        audit.audit_scattering(column, np.array([2500.]), chemistry)


def test_component_removal_preserves_column_and_chemistry(audit):
    from collections import namedtuple
    from _rce_broadband import Fluxes

    State = namedtuple("State", "number_density_e diagnostics")
    chemistry = State(np.array([4., 4.]), SimpleNamespace(valid=np.ones(2, dtype=bool)))

    class Model:
        ckd = {"A": 1., "B": 2.}
        cia = {("H2", "H2"): 3.}
        g_weights = np.ones(1)
        band_widths = np.ones(1)

        def fluxes(self, temperature, bottom, parameters, state):
            coefficient = sum(self.ckd.values()) + sum(self.cia.values()) + state.number_density_e[0]
            up = np.ones((2, 1, 1)) * coefficient
            incident = np.ones_like(up) * (20. - coefficient)
            return Fluxes(up, np.zeros_like(up), incident, up - incident)

    column = Model()
    report = audit.fixed_column_component_sensitivity(column, [2500.], 2500., [0., .6, -.7], chemistry)
    values = report["removed_sources"]
    for name, coefficient in (("A", 1.), ("B", 2.), ("CIA:H2:H2", 3.), ("H-minus", 4.)):
        assert values[name]["top_thermal_flux_change_cgs"] == -coefficient
        assert values[name]["max_net_flux_change_cgs"] == 2 * coefficient
        assert values[name]["bottom_incident_flux_change_cgs"] == coefficient
    assert column.ckd == {"A": 1., "B": 2.}
    assert column.cia == {("H2", "H2"): 3.}
    np.testing.assert_array_equal(chemistry.number_density_e, [4., 4.])
    column.ckd = {"A": 1.}
    with pytest.raises(ValueError, match="at least two"):
        audit.fixed_column_component_sensitivity(column, [2500.], 2500., [0., .6, -.7], chemistry)
