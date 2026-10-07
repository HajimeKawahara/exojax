"""Offline contracts for named molecular opacity construction and validation."""

from copy import deepcopy
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.opacity.multimol import (
    build_premodit,
    multiopa_premodit,
    validate_opacity_grids,
)
from exojax.opacity.policies import MemoryPolicy
from exojax.opacity.premodit.api import OpaPremodit


def _snapshot(molmass=18.0, empty=False):
    line_count = 0 if empty else 3
    return MDBSnapshot(
        meta=MDBMeta(
            dbtype="exomol",
            molmass=molmass,
            T_gQT=np.array([300.0, 1000.0, 2000.0]),
            gQT=np.array([1.0, 2.0, 4.0]),
        ),
        lines=Lines(
            nu_lines=np.array([1000.0, 1005.0, 1010.0])[:line_count],
            elower=np.array([10.0, 20.0, 30.0])[:line_count],
            line_strength_ref_original=np.array([1e-22, 2e-22, 3e-22])[
                :line_count
            ],
        ),
        n_Texp=np.full(line_count, 0.5),
        alpha_ref=np.full(line_count, 0.1),
    )


class _MDB:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.snapshot_calls = 0

    def to_snapshot(self):
        self.snapshot_calls += 1
        return self.snapshot


def _ready_opa(nu_grid, molmass=18.0):
    return SimpleNamespace(
        nu_grid=np.asarray(nu_grid),
        molmass=molmass,
        ready=True,
        xsmatrix=lambda temperatures, pressures: jnp.ones(
            (len(temperatures), len(nu_grid))
        ),
    )


@pytest.fixture
def nu_grid():
    return np.geomspace(990.0, 1020.0, 32)


@pytest.fixture
def manual_options():
    return {"manual_params": (100.0, 1000.0, 1100.0)}


def test_build_ready_opacities_preserves_names_and_snapshot_masses(
    nu_grid, manual_options
):
    water = _snapshot(molmass=19.0)
    carbon_monoxide = _MDB(_snapshot(molmass=29.0))
    databases = {"H2O-18": water, "13CO": carbon_monoxide}
    original_lines = deepcopy(water.lines)

    opas = build_premodit(databases, nu_grid, **manual_options)

    assert tuple(opas) == tuple(databases)
    assert opas["H2O-18"].molmass == 19.0
    assert opas["13CO"].molmass == 29.0
    assert carbon_monoxide.snapshot_calls == 1
    for opa in opas.values():
        assert opa.ready
        xs = opa.xsmatrix(jnp.array([900.0, 1100.0]), jnp.array([0.1, 1.0]))
        assert xs.shape == (2, len(nu_grid))
        assert np.all(np.isfinite(xs))
        assert np.any(xs > 0.0)
        np.testing.assert_array_equal(opa.nu_grid, nu_grid)
    assert databases["H2O-18"] is water
    assert databases["13CO"] is carbon_monoxide
    for field in ("nu_lines", "elower", "line_strength_ref_original"):
        np.testing.assert_array_equal(
            getattr(water.lines, field), getattr(original_lines, field)
        )


def test_build_forwards_current_premodit_options(
    monkeypatch, nu_grid, manual_options
):
    calls = []

    def from_snapshot(snapshot, grid, **kwargs):
        calls.append((snapshot, grid, kwargs))
        return _ready_opa(grid, snapshot.meta.molmass)

    monkeypatch.setattr(OpaPremodit, "from_snapshot", staticmethod(from_snapshot))
    snapshot = _snapshot()
    policy = MemoryPolicy(allow_32bit=True)
    broadening = {"mode": "manual", "value": 0.5}
    options = {
        **manual_options,
        "broadening_resolution": broadening,
        "memory_policy": policy,
        "profile_kernel": "real_space",
        "diffmode": 1,
        "nstitch": 2,
    }

    build_premodit({"water": snapshot}, nu_grid, **options)

    assert len(calls) == 1
    assert calls[0][0] is snapshot
    np.testing.assert_array_equal(calls[0][1], nu_grid)
    assert calls[0][2] == options
    assert calls[0][2]["memory_policy"] is policy
    assert calls[0][2]["broadening_resolution"] is broadening


def test_build_requires_complete_premodit_setup(nu_grid):
    with pytest.raises(ValueError, match="auto_trange|manual_params"):
        build_premodit({"H2O": _snapshot()}, nu_grid)


@pytest.mark.parametrize("named", [True, False])
def test_both_builders_export_mdb_snapshot_once(monkeypatch, nu_grid, named):
    snapshot = _snapshot()
    mdb = _MDB(snapshot)
    constructed = []

    def from_snapshot(payload, grid, **kwargs):
        constructed.append(payload)
        return _ready_opa(grid, payload.meta.molmass)

    monkeypatch.setattr(OpaPremodit, "from_snapshot", staticmethod(from_snapshot))
    if named:
        result = build_premodit({"water": mdb}, nu_grid, auto_trange=(500.0, 1500.0))
        opa = result["water"]
    else:
        result = multiopa_premodit([[mdb]], nu_grid, auto_trange=(500.0, 1500.0))
        opa = result[0][0]

    assert mdb.snapshot_calls == 1
    assert len(constructed) == 1
    assert constructed[0] is snapshot
    assert opa.molmass == snapshot.meta.molmass


def test_fixed_pressure_diffgrid_interface_is_rejected(nu_grid):
    opa = _ready_opa(nu_grid)
    opa.method = "diffgrid"
    with pytest.raises(ValueError, match="CO.*fixed pressure"):
        validate_opacity_grids({"CO": opa})


def test_empty_lines_raise_with_species_name(nu_grid, manual_options):
    with pytest.raises(ValueError, match="H2O"):
        build_premodit({"H2O": _snapshot(empty=True)}, nu_grid, **manual_options)


@pytest.mark.parametrize("include_nonempty", [False, True])
def test_explicit_zero_keeps_empty_species_and_their_masses(
    monkeypatch, nu_grid, manual_options, include_nonempty
):
    calls = []

    def from_snapshot(snapshot, grid, **kwargs):
        assert len(snapshot.lines.nu_lines) > 0
        calls.append(snapshot)
        return _ready_opa(grid, snapshot.meta.molmass)

    monkeypatch.setattr(OpaPremodit, "from_snapshot", staticmethod(from_snapshot))
    databases = {"H2O": _snapshot(empty=True), "13CO": _snapshot(29.0, empty=True)}
    if include_nonempty:
        databases["CH4"] = _snapshot(16.0)

    opas = build_premodit(databases, nu_grid, on_empty="zero", **manual_options)

    assert tuple(opas) == tuple(databases)
    assert len(calls) == int(include_nonempty)
    for name in ("H2O", "13CO"):
        assert opas[name].ready
        assert opas[name].molmass == databases[name].meta.molmass
        np.testing.assert_array_equal(opas[name].nu_grid, nu_grid)
        np.testing.assert_array_equal(
            opas[name].xsmatrix(jnp.array([800.0, 1000.0]), jnp.array([0.1, 1.0])),
            np.zeros((2, len(nu_grid))),
        )
    validate_opacity_grids(opas)


def test_build_does_not_swallow_snapshot_errors(nu_grid, manual_options):
    failure = RuntimeError("The database payload is corrupt.")

    class BrokenMDB:
        def to_snapshot(self):
            raise failure

    with pytest.raises(RuntimeError) as caught:
        build_premodit(
            {"H2O": BrokenMDB()}, nu_grid, on_empty="zero", **manual_options
        )
    assert caught.value is failure


def test_build_does_not_treat_constructor_errors_as_empty(
    monkeypatch, nu_grid, manual_options
):
    failure = ValueError("Invalid broadening configuration.")

    def from_snapshot(*args, **kwargs):
        raise failure

    monkeypatch.setattr(OpaPremodit, "from_snapshot", staticmethod(from_snapshot))
    with pytest.raises(ValueError) as caught:
        build_premodit(
            {"H2O": _snapshot()}, nu_grid, on_empty="zero", **manual_options
        )
    assert caught.value is failure


def test_unknown_empty_policy_is_rejected(nu_grid, manual_options):
    with pytest.raises(ValueError, match="on_empty"):
        build_premodit(
            {"H2O": _snapshot()}, nu_grid, on_empty="ignore", **manual_options
        )


def test_validate_accepts_ready_opacities_on_equal_grids(nu_grid):
    validate_opacity_grids(
        {"H2O": _ready_opa(nu_grid), "CO": _ready_opa(nu_grid.copy(), 28.0)}
    )


def test_validate_rejects_same_length_but_different_grids(nu_grid):
    with pytest.raises(ValueError, match="grid|wavenumber"):
        validate_opacity_grids(
            {"H2O": _ready_opa(nu_grid), "CO": _ready_opa(nu_grid + 0.1, 28.0)}
        )


@pytest.mark.parametrize(
    "grid",
    [np.array([]), np.ones((2, 2)), [1000.0, np.nan], [1000.0, np.inf],
     [1001.0, 1000.0], [1000.0, 1000.0]],
)
def test_validate_rejects_invalid_grid(grid):
    with pytest.raises(ValueError):
        validate_opacity_grids({"H2O": _ready_opa(grid)})


@pytest.mark.parametrize("molmass", [0.0, -18.0, np.nan, np.inf])
def test_validate_rejects_invalid_mass(nu_grid, molmass):
    with pytest.raises(ValueError, match="mass|molmass"):
        validate_opacity_grids({"H2O": _ready_opa(nu_grid, molmass)})


@pytest.mark.parametrize("opas", [{}, {1: object()}, [object()]])
def test_validate_requires_nonempty_named_mapping(opas):
    with pytest.raises((TypeError, ValueError)):
        validate_opacity_grids(opas)


@pytest.mark.parametrize("attribute,value", [("ready", False), ("xsmatrix", None)])
def test_validate_rejects_unready_or_invalid_opacity(nu_grid, attribute, value):
    opa = _ready_opa(nu_grid)
    setattr(opa, attribute, value)
    with pytest.raises((TypeError, ValueError), match="H2O"):
        validate_opacity_grids({"H2O": opa})
