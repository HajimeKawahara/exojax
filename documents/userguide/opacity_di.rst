Dependency Injection for OpaPremodit
====================================

This how-to shows the new DI-friendly constructors for PreMODIT opacity.

Minimal Example (ExoMol)
------------------------

.. code-block:: python

   from exojax.database.exomol.api import MdbExomol
   from exojax.opacity import OpaPremodit
   from exojax.utils.grids import wavenumber_grid

   # Make a grid and load an mdb
   nu_grid, _, _ = wavenumber_grid(4200.0, 4300.0, 20000, xsmode="premodit")
   mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid)

   # 1) Build from a snapshot (data-only DTO)
   snap = mdb.to_snapshot()
   opa1 = OpaPremodit.from_snapshot(
       snap, nu_grid, manual_params=(5.0, 1000.0, 1200.0)
   )

   # 2) Back-compat: build directly from an mdb
   opa2 = OpaPremodit.from_mdb(
       mdb, nu_grid, manual_params=(5.0, 1000.0, 1200.0)
   )

Notes
-----
- ``from_snapshot`` avoids dependencies on concrete mdb classes.
- ``from_mdb`` mirrors legacy usage but routes through a snapshot internally.
- Both APIs accept the same keyword options as the original constructor.

Optional Memory Policy
----------------------

Override memory/runtime knobs explicitly (defaults unchanged if omitted):

.. code-block:: python

   from exojax.opacity.policies import MemoryPolicy

   opa3 = OpaPremodit.from_mdb(
       mdb,
       nu_grid,
       manual_params=(5.0, 1000.0, 1200.0),
       memory_policy=MemoryPolicy(allow_32bit=True, nstitch=2, cutwing=0.5),
   )
