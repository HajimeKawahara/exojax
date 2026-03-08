# RADIS Backend Positioning

## Current role of `radis_adapter`

`src/exojax/database/_common/radis_adapter.py` is the current RADIS-specific backend implementation layer for ExoJAX database code.

It is not only an import shim. It also contains backend logic such as:
- backend capability checks (for ExoMol feature support)
- backend manager initialization helpers
- backend compatibility/legacy-path handling
- backend-specific collision handling around manager registration

ExoJAX database call sites should prefer adapter helpers over embedding RADIS constructor/version details directly.

## What logic should live in `radis_adapter`

Good candidates for this layer:
- backend capability/query functions (feature support checks)
- backend manager construction/init wrappers/factories
- backend-specific exception compatibility handling
- backend data lookup helpers where ExoJAX only needs values

Logic that should stay in ExoJAX call sites:
- ExoJAX domain workflows and data flow
- ExoJAX public API semantics
- numerical/scientific behavior definitions

## Legacy that still remains

Class-returning APIs still exist:
- `get_exomol_mdb_class()`
- `get_hitran_database_manager_class()`
- `get_hitemp_database_manager_class()`

These remain because current DB classes still rely on inheritance from backend manager classes. This is an acknowledged transitional state.

## How a future native backend can sit alongside RADIS

A future native backend can be added as a parallel backend layer implementing the same adapter responsibilities:
- capability/query helpers
- manager/object creation helpers
- backend-specific compatibility logic

In that model, ExoJAX core call sites can remain stable while backend implementations vary behind the adapter boundary.

## Naming note (not changed in this PR)

A future low-risk naming cleanup may introduce a backend-neutral module name (for example, `backend_adapter.py`) with `radis_adapter.py` either delegating to it or remaining as the RADIS implementation module.
