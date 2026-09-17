import importlib
import sys

import pytest


def _import_exomol_api_with_missing_base(monkeypatch):
    from exojax.database._common import radis_adapter

    def _raise_missing_radis():
        raise ImportError(
            "ExoMol database manager access requires RADIS. "
            "Install it with `pip install radis`."
        )

    monkeypatch.setattr(radis_adapter, "get_exomol_mdb_class", _raise_missing_radis)
    monkeypatch.delitem(sys.modules, "exojax.database.exomol.api", raising=False)
    return importlib.import_module("exojax.database.exomol.api")


def test_exomol_api_import_survives_missing_radis_base(monkeypatch):
    module = _import_exomol_api_with_missing_base(monkeypatch)
    assert hasattr(module, "MdbExomol")


def test_exomol_class_use_fails_with_clear_importerror_when_radis_missing(monkeypatch):
    module = _import_exomol_api_with_missing_base(monkeypatch)
    with pytest.raises(ImportError, match="requires RADIS"):
        module.CapiMdbExomol()
