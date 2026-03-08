# RADIS Refactor Plan (Initial)

## 1) Direct RADIS imports grouped by file

### `src/` (runtime code)
- `src/exojax/database/_common/setradis.py`
  - `from radis.api.dbmanager import get_auto_MEMORY_MAPPING_ENGINE`
- `src/exojax/database/_common/commonapi.py`
  - `from radis.db.classes import get_molecule`
  - `from radis.levels.partfunc import PartFuncTIPS`
- `src/exojax/database/_common/hitranapi.py`
  - `from radis.db.classes import get_molecule, get_molecule_identifier`
- `src/exojax/database/exomol/api.py`
  - `from radis import __version__ as radis_version`
  - `from radis.api.exomolapi import MdbExomol as CapiMdbExomol`
- `src/exojax/database/hitran/api.py`
  - `from radis.api.hitranapi import HITRANDatabaseManager`
- `src/exojax/database/hitemp/api.py`
  - `from radis.api.hitempapi import HITEMPDatabaseManager`
  - (lazy) `from radis.api.hitranapi import hit2df`
- `src/exojax/database/multimol.py`
  - (lazy) `from radis.db.classes import get_molecule_identifier`
  - (lazy) `from radis.api.exomolapi import get_exomol_database_list`
- `src/exojax/utils/molname.py`
  - `from radis.db.classes import get_molecule`
  - (lazy) `from radis.db.molparam import MolParams`
  - (lazy) `from radis.db.molparam import isotope_name_dict`
- `src/exojax/utils/mollabel.py`
  - (lazy) `from radis.db.classes import get_molecule_identifier`

### `tests/` (direct RADIS import)
- `tests/unittests/database/api_radis/exomolapi_test.py`
  - `from radis.api.exomolapi import check_code_level`
- `tests/unittests/database/api_radis/molecular_name_test.py`
  - `from radis.api.exomolapi import exact_molname_exomol_to_simple_molname`

### `examples/`
- No direct RADIS import found.

## 2) Imported symbols (flattened)
- `radis.__version__`
- `radis.api.dbmanager.get_auto_MEMORY_MAPPING_ENGINE`
- `radis.api.exomolapi.MdbExomol`
- `radis.api.exomolapi.get_exomol_database_list` (lazy)
- `radis.api.exomolapi.check_code_level` (tests)
- `radis.api.exomolapi.exact_molname_exomol_to_simple_molname` (tests)
- `radis.api.hitranapi.HITRANDatabaseManager`
- `radis.api.hitranapi.hit2df` (lazy)
- `radis.api.hitempapi.HITEMPDatabaseManager`
- `radis.db.classes.get_molecule`
- `radis.db.classes.get_molecule_identifier`
- `radis.db.molparam.MolParams` (lazy)
- `radis.db.molparam.isotope_name_dict` (lazy)
- `radis.levels.partfunc.PartFuncTIPS`

## 3) Dependency class: core vs optional vs test/example-only

### Core runtime coupling (imported in core DB API paths)
- `database/exomol/api.py`, `database/hitran/api.py`, `database/hitemp/api.py`
- `database/_common/commonapi.py`, `database/_common/setradis.py`, `database/_common/hitranapi.py`
- These are required to instantiate production MDB classes.

### Optional/runtime-conditional coupling
- `database/multimol.py`
  - `get_exomol_database_list` only used when local ExoMol dataset discovery fails.
  - `get_molecule_identifier` only used to build default HITRAN path.
- `utils/mollabel.py`
  - color helper degrades to `"gray"` on failure.
- `utils/molname.py`
  - some isotope conversion helpers use lazy `MolParams`/`isotope_name_dict`; however, module currently has top-level RADIS import (`get_molecule`), so effective import-time coupling still exists.

### Test-only direct RADIS coupling
- `tests/unittests/database/api_radis/*` directly test RADIS API consistency.
- No example-only direct RADIS imports found.

## 4) Additional RADIS-related dependencies not captured by grep outputs

- Packaging metadata coupling:
  - `pyproject.toml` includes `radis>=0.15.2` as a base dependency.
  - `pyproject.toml` optional extra `numpy2` pins RADIS from GitHub (`radis @ git+https://github.com/radis/radis.git@master`).
- Transitive/adjacent runtime coupling in RADIS codepaths:
  - `hitran-api` + `hapi` usage in `src/exojax/database/_common/hitranapi.py` (partition grid access through `hapi.TIPS_*`).
  - Engine/backend assumptions (`"vaex"` / `"pytables"`) appear across DB API modules; `tables` is listed in dependencies, while `vaex` is referenced in code/docs/tests and may be environment-conditional.
- Non-import references (docs/comments/strings) to RADIS GitHub issues and API names exist but are not functional dependencies.

## 5) Minimal adapter layer proposal

Create a single boundary module, e.g.:
- `src/exojax/database/_common/radis_adapter.py`

### Adapter responsibilities
- Centralize all RADIS imports behind small wrapper functions.
- Provide uniform error messages when RADIS (or submodules) is unavailable.
- Expose only the symbols needed by ExoJAX, e.g.:
  - `get_radis_version()`
  - `get_auto_engine()`
  - `get_molecule_name(molecid)` / `get_molecule_identifier(name)`
  - `get_partfunc_tips(molecid, isotope)` (or factory returning `PartFuncTIPS`)
  - `get_exomol_manager_class()` (returns RADIS `MdbExomol`)
  - `get_hitran_manager_class()` / `get_hitemp_manager_class()`
  - `hit2df(...)`
  - `get_exomol_database_list(...)`
  - `get_molparams()` / `get_isotope_name_dict()`

### Minimal migration steps
1. Add `radis_adapter.py` (no behavior changes).
2. Replace direct RADIS imports in `src/exojax/database/**` and `src/exojax/utils/**` with adapter calls.
3. Keep lazy import semantics where currently intentional (`multimol.py`, `hitemp/api.py`, `mollabel.py`, molparam helpers).
4. Add focused unit tests for adapter fallback/error paths (missing RADIS or missing submodule).
5. Leave `tests/unittests/database/api_radis/*` as direct RADIS tests, but optionally add adapter-level tests to decouple most of the suite.

This isolates vendor-specific API churn (RADIS version checks, symbol moves, backend changes) to one file while minimizing diff size in core database logic.
