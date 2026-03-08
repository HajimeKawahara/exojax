import numbers
import sys
import types
import warnings

import numpy as np
import pytest

from exojax.database._common import radis_adapter


def _install_module(monkeypatch, module_name, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_identity_translation_helpers_roundtrip_and_isotope_name(monkeypatch):
    radis_mod = _install_module(monkeypatch, "radis")
    db_mod = _install_module(monkeypatch, "radis.db")
    setattr(radis_mod, "db", db_mod)

    classes_mod = _install_module(
        monkeypatch,
        "radis.db.classes",
        get_molecule=lambda molecule_identifier: {5: "CO"}[molecule_identifier],
        get_molecule_identifier=lambda simple_molecule_name: {"CO": 5}[simple_molecule_name],
    )
    setattr(db_mod, "classes", classes_mod)

    molparam_mod = _install_module(
        monkeypatch,
        "radis.db.molparam",
        MolParams=type(
            "MolParams",
            (),
            {"get": lambda self, simple_molecule_name, isotope, key: "(12C)(16O)"},
        ),
    )
    setattr(db_mod, "molparam", molparam_mod)

    molecule_id = radis_adapter.get_molecule_identifier("CO")
    assert isinstance(molecule_id, numbers.Integral)
    assert radis_adapter.get_molecule(molecule_id) == "CO"
    assert radis_adapter.get_isotope_name("CO", 1) == "(12C)(16O)"


def test_metadata_lookup_helpers_return_callables(monkeypatch):
    radis_mod = _install_module(monkeypatch, "radis")
    api_mod = _install_module(monkeypatch, "radis.api")
    setattr(radis_mod, "api", api_mod)

    hitranapi_mod = _install_module(
        monkeypatch,
        "radis.api.hitranapi",
        hit2df=lambda *args, **kwargs: None,
    )
    exomolapi_mod = _install_module(
        monkeypatch,
        "radis.api.exomolapi",
        get_exomol_database_list=lambda *args, **kwargs: [],
    )
    setattr(api_mod, "hitranapi", hitranapi_mod)
    setattr(api_mod, "exomolapi", exomolapi_mod)

    assert callable(radis_adapter.get_hit2df_func())
    assert callable(radis_adapter.get_exomol_database_list_func())


def test_partition_function_query_returns_positive_scalar(monkeypatch):
    radis_mod = _install_module(monkeypatch, "radis")
    levels_mod = _install_module(monkeypatch, "radis.levels")
    setattr(radis_mod, "levels", levels_mod)

    class DummyPartFuncTIPS:
        def __init__(self, molecule_identifier, isotope):
            self.molecule_identifier = molecule_identifier
            self.isotope = isotope

        def at(self, T):
            return 123.4 if T == 296.0 else 0.0

    partfunc_mod = _install_module(
        monkeypatch,
        "radis.levels.partfunc",
        PartFuncTIPS=DummyPartFuncTIPS,
    )
    setattr(levels_mod, "partfunc", partfunc_mod)

    q = radis_adapter.get_partition_function_value(5, 1, 296.0)
    assert np.isscalar(q)
    assert q > 0.0


@pytest.mark.parametrize(
    "radis_version, expected_support, expected_mode",
    [
        ("0.14", False, "compute_broadening"),
        ("0.15.2", False, "set_broadening_coef_legacy"),
        ("0.16", True, "set_broadening_coef_species"),
    ],
)
def test_capability_helpers_are_version_gated(
    monkeypatch, radis_version, expected_support, expected_mode
):
    monkeypatch.setattr(radis_adapter, "get_radis_version", lambda: radis_version)
    assert radis_adapter.supports_exomol_broadf_download() is expected_support
    assert radis_adapter.exomol_init_needs_bkgdatm() is (not expected_support)
    assert radis_adapter.exomol_broadening_mode() == expected_mode


def test_warn_if_exomol_broadf_download_unsupported_emits_only_for_legacy(monkeypatch):
    monkeypatch.setattr(radis_adapter, "get_radis_version", lambda: "0.15.2")
    with pytest.warns(UserWarning, match="does not support broadf_download"):
        radis_adapter.warn_if_exomol_broadf_download_unsupported()

    monkeypatch.setattr(radis_adapter, "get_radis_version", lambda: "0.16")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        radis_adapter.warn_if_exomol_broadf_download_unsupported()
    assert caught == []


@pytest.mark.parametrize(
    "needs_bkgdatm, required_key, forbidden_key",
    [
        (True, "bkgdatm", "broadf_download"),
        (False, "broadf_download", "bkgdatm"),
    ],
)
def test_init_exomol_manager_selects_expected_constructor_branch(
    monkeypatch, needs_bkgdatm, required_key, forbidden_key
):
    calls = []

    class DummyExomolManager:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(radis_adapter, "get_exomol_mdb_class", lambda: DummyExomolManager)
    monkeypatch.setattr(radis_adapter, "exomol_init_needs_bkgdatm", lambda: needs_bkgdatm)

    radis_adapter.init_exomol_manager(
        object(),
        path="dummy",
        local_databases="./",
        molecule="CO",
        nurange=[1000.0, 1001.0],
        engine="vaex",
        crit=0.0,
        broadf=True,
        broadf_download=False,
        skip_optional_data=True,
        bkgdatm="H2",
    )

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert required_key in kwargs
    assert forbidden_key not in kwargs


@pytest.mark.parametrize(
    "nonair_broadening, expected_extra_params",
    [(True, "all"), (False, None)],
)
def test_init_hitran_manager_sets_extra_params_from_nonair_flag(
    monkeypatch, nonair_broadening, expected_extra_params
):
    calls = []

    class DummyHitranManager:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        radis_adapter, "get_hitran_database_manager_class", lambda: DummyHitranManager
    )

    radis_adapter.init_hitran_manager(
        object(),
        molecule="CO",
        local_databases="./",
        engine="vaex",
        nonair_broadening=nonair_broadening,
    )

    assert len(calls) == 1
    assert calls[0]["extra_params"] == expected_extra_params


def test_init_hitemp_manager_retries_with_local_tag_on_name_collision(monkeypatch):
    names = []

    class DummyHitempManager:
        def __init__(self, *args, **kwargs):
            names.append(kwargs["name"])
            if len(names) == 1:
                raise ValueError("already registered in radis.json")

    monkeypatch.setattr(
        radis_adapter, "get_hitemp_database_manager_class", lambda: DummyHitempManager
    )

    radis_adapter.init_hitemp_manager(
        object(),
        molecule="CO",
        local_databases="/tmp/radis_local_db",
        engine="vaex",
    )

    assert names == ["HITEMP-CO", "HITEMP-CO-radis_local_db"]


def test_init_hitemp_manager_reraises_unrelated_valueerror(monkeypatch):
    class DummyHitempManager:
        def __init__(self, *args, **kwargs):
            raise ValueError("some other error")

    monkeypatch.setattr(
        radis_adapter, "get_hitemp_database_manager_class", lambda: DummyHitempManager
    )

    with pytest.raises(ValueError, match="some other error"):
        radis_adapter.init_hitemp_manager(
            object(),
            molecule="CO",
            local_databases="./",
            engine="vaex",
        )
