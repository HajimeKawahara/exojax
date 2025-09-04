Loose Coupling: OpaPremodit ↔ MDB
=================================

This page shows how to construct `OpaPremodit` using data-only snapshots or
directly from an MDB object. The goal is loose coupling: opacity code does not
depend on concrete database classes.

Minimal Example
---------------

.. code-block:: python

   from exojax.database.exomol.api import MdbExomol
   from exojax.opacity import OpaPremodit
   from exojax.opacity.policies import MemoryPolicy
   from exojax.utils.grids import wavenumber_grid

   # 1) Make a grid (ESLOG) and load an MDB
   nu_grid, _, _ = wavenumber_grid(4200.0, 4300.0, 20000, xsmode="premodit")
   mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid)

   # 2) Snapshot path (data-only DTO)
   snap = mdb.to_snapshot()
   opa_snap = OpaPremodit.from_snapshot(
       snap, nu_grid, manual_params=(5.0, 1000.0, 1200.0)
   )

   # 3) Back-compat path (routes through snapshot internally)
   opa_mdb = OpaPremodit.from_mdb(
       mdb, nu_grid, manual_params=(5.0, 1000.0, 1200.0)
   )

Optional: Memory Policy
-----------------------

Override memory/runtime knobs explicitly (defaults unchanged if omitted):

.. code-block:: python

   policy = MemoryPolicy(allow_32bit=True, nstitch=2, cutwing=0.5)
   opa = OpaPremodit.from_mdb(mdb, nu_grid, manual_params=(5.0, 1000.0, 1200.0),
                              memory_policy=policy)

Notes
-----
- `from_snapshot` avoids any dependency on concrete MDB classes.
- `from_mdb` mirrors legacy usage for easy adoption.
- Both constructors are behaviorally equivalent to the legacy `OpaPremodit(mdb, ...)` path.

