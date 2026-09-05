import importlib
import sys

import pytest


def _import_hitran_api_with_missing_manager(monkeypatch):
    from exojax.database._common import radis_adapter

    def _raise_missing_radis():
        raise ImportError(
            "HITRAN database manager access requires RADIS. "
            "Install it with `pip install radis`."
        )

    monkeypatch.setattr(radis_adapter, "get_hitran_database_manager_class", _raise_missing_radis)
    monkeypatch.delitem(sys.modules, "exojax.database.hitran.api", raising=False)
    return importlib.import_module("exojax.database.hitran.api")


def _import_hitemp_api_with_missing_manager(monkeypatch):
    from exojax.database._common import radis_adapter

    def _raise_missing_radis():
        raise ImportError(
            "HITEMP database manager access requires RADIS. "
            "Install it with `pip install radis`."
        )

    monkeypatch.setattr(radis_adapter, "get_hitemp_database_manager_class", _raise_missing_radis)
    monkeypatch.delitem(sys.modules, "exojax.database.hitemp.api", raising=False)
    return importlib.import_module("exojax.database.hitemp.api")


def test_hitran_api_import_survives_missing_radis_manager(monkeypatch):
    module = _import_hitran_api_with_missing_manager(monkeypatch)
    assert hasattr(module, "MdbHitran")


def test_hitemp_api_import_survives_missing_radis_manager(monkeypatch):
    module = _import_hitemp_api_with_missing_manager(monkeypatch)
    assert hasattr(module, "MdbHitemp")


def test_hitran_fallback_manager_use_fails_with_clear_importerror(monkeypatch):
    module = _import_hitran_api_with_missing_manager(monkeypatch)
    with pytest.raises(ImportError, match="requires RADIS"):
        module.HITRANDatabaseManager()


def test_hitemp_fallback_manager_use_fails_with_clear_importerror(monkeypatch):
    module = _import_hitemp_api_with_missing_manager(monkeypatch)
    with pytest.raises(ImportError, match="requires RADIS"):
        module.HITEMPDatabaseManager()
