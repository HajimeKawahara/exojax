import builtins
import importlib
import sys


_BLOCKED_MODULES = (
    "exojax.database.exomol.api",
    "exojax.database.hitran.api",
    "exojax.database.hitemp.api",
    "exojax.test.emulate_mdb",
)


def _block_eager_backend_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in _BLOCKED_MODULES:
            raise ModuleNotFoundError(
                f"blocked eager import in test: {name}",
                name=name,
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _clear_modules(monkeypatch, module_names):
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_database_api_import_avoids_eager_radis_backed_modules(monkeypatch):
    _clear_modules(
        monkeypatch,
        [
            "exojax.database.api",
            "exojax.database.exomol.api",
            "exojax.database.hitran.api",
            "exojax.database.hitemp.api",
        ],
    )
    _block_eager_backend_imports(monkeypatch)

    module = importlib.import_module("exojax.database.api")

    assert module.__name__ == "exojax.database.api"
    assert "exojax.database.exomol.api" not in sys.modules
    assert "exojax.database.hitran.api" not in sys.modules
    assert "exojax.database.hitemp.api" not in sys.modules


def test_multimol_import_avoids_eager_radis_backed_modules(monkeypatch):
    _clear_modules(
        monkeypatch,
        [
            "exojax.database.multimol",
            "exojax.database.exomol.api",
            "exojax.database.hitran.api",
            "exojax.database.hitemp.api",
            "exojax.test.emulate_mdb",
        ],
    )
    _block_eager_backend_imports(monkeypatch)

    module = importlib.import_module("exojax.database.multimol")

    assert module.__name__ == "exojax.database.multimol"
    assert "exojax.database.exomol.api" not in sys.modules
    assert "exojax.database.hitran.api" not in sys.modules
    assert "exojax.database.hitemp.api" not in sys.modules
    assert "exojax.test.emulate_mdb" not in sys.modules
