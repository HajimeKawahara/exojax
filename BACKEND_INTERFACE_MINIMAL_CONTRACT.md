# Backend Interface Minimal Contract (Current ExoJAX)

This note defines the minimal backend-facing contract that ExoJAX core currently relies on, based on existing code paths.

Scope references:
- `src/exojax/database/_common/radis_adapter.py`
- Core call sites in `src/exojax/database/**` and `src/exojax/utils/**`
- Positioning note: `RADIS_BACKEND_POSITIONING.md`

## Purpose

ExoJAX core should depend on backend capabilities and data/query helpers, not backend package internals.
The current implementation backend is RADIS (`radis_adapter.py`).

## Contract categories

## 1) Identity / label translation

Current helpers:
- `get_molecule(molecule_identifier)`
- `get_molecule_identifier(simple_molecule_name)`
- `get_isotope_name(simple_molecule_name, isotope)`
- `get_isotope_name_dict()`

Used by:
- `src/exojax/database/_common/commonapi.py`
- `src/exojax/database/_common/hitranapi.py`
- `src/exojax/database/multimol.py`
- `src/exojax/utils/molname.py`
- `src/exojax/utils/mollabel.py`

Contract status:
- Stable backend-facing contract: yes (`get_molecule`, `get_molecule_identifier`, `get_isotope_name`)
- Transitional: `get_isotope_name_dict()` (dict schema is backend-shaped)
- Non-contract details that should stay hidden: backend database class names and internal table layout.

## 2) Metadata/data lookup helpers

Current helpers:
- `get_exomol_database_list_func()`
- `get_hit2df_func()`

Used by:
- `src/exojax/database/multimol.py`
- `src/exojax/database/hitemp/api.py`

Contract status:
- Stable backend-facing contract: partially (callers currently receive raw backend callables)
- Transitional: yes (callable-returning style leaks backend function signatures)
- Non-contract details: concrete RADIS function paths (`radis.api.*`).

## 3) Partition-function queries

Current helper:
- `get_partition_function_value(molecule_identifier, isotope, temperature)`

Used by:
- `src/exojax/database/_common/commonapi.py`

Contract status:
- Stable backend-facing contract: yes (value-oriented, backend-neutral input/output)
- Transitional: no major issues in current usage
- Non-contract details: backend class construction (`PartFuncTIPS`) should remain adapter-internal.

## 4) Capability / feature queries

Current helpers:
- `supports_exomol_broadf_download()`
- `exomol_init_needs_bkgdatm()`
- `exomol_broadening_mode()`
- `warn_if_exomol_broadf_download_unsupported()`

Used by:
- `src/exojax/database/exomol/api.py`

Contract status:
- Stable backend-facing contract: yes for capability checks (`supports_*`, mode queries)
- Transitional: warning helper is policy-specific but acceptable for now
- Non-contract details: raw backend version branching should not reappear in call sites.

## 5) Manager/object initialization helpers

Current helpers:
- `init_exomol_manager(...)`
- `init_hitran_manager(...)`
- `init_hitemp_manager(...)`

Used by:
- `src/exojax/database/exomol/api.py`
- `src/exojax/database/hitran/api.py`
- `src/exojax/database/hitemp/api.py`

Contract status:
- Stable backend-facing contract: yes as an initialization boundary
- Transitional: still inheritance-based; helpers initialize backend managers on `self`
- Non-contract details: backend constructor kwargs/version quirks should remain inside these helpers.

## 6) Activation/setup behavior (current state)

Current state:
- Backend-specific activation/setup is partially localized via capability + init helpers.
- Core classes still perform ExoJAX-level workflow (`load`, masking, activate) in their own modules.

Contract status:
- Stable backend-facing contract: partial
- Transitional: yes (inheritance lock-in remains)
- Non-contract details: backend manager internals and MRO assumptions should not spread further.

## Legacy / transitional APIs (do not expand)

Class-returning helpers (legacy transitional):
- `get_exomol_mdb_class()`
- `get_hitran_database_manager_class()`
- `get_hitemp_database_manager_class()`

Reason:
- Required by current inheritance architecture.
- Should not be expanded into new call sites; prefer value/capability/init helpers.

Low-level version primitive (adapter-internal primitive):
- `get_radis_version()`

Reason:
- Used to implement capability helpers inside adapter.
- ExoJAX call sites should not branch on this directly.

## What must not leak from backend to core

- Raw backend version comparisons.
- Backend constructor keyword/version compatibility branches.
- Backend exception-string coupling outside adapter where avoidable.
- Direct dependency on backend class APIs when a value/query helper can be used.

## What can remain RADIS-specific for now

- Adapter internals that call RADIS constructors and APIs.
- Transitional class-returning helpers needed for current inheritance.
- Existing backend-specific warning text in adapter compatibility helpers.

## Minimal first implementation target for a future native backend

A future native backend should first implement the currently used backend-facing helper families:
1. Identity/label translation (`get_molecule`, `get_molecule_identifier`, `get_isotope_name`)
2. Partition-function value query (`get_partition_function_value`)
3. Capability queries for ExoMol-like branches (`supports_*`, mode helpers)
4. Manager initialization helpers (`init_*_manager`)

Once those are available, ExoJAX core call sites can remain mostly unchanged while backend internals differ.
