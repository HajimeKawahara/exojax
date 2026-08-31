Saving and Loading DiffGrid
===========================

An ``OpaDiffgrid`` table can be saved after construction and restored
without its PreMODIT teacher or molecular database.  This avoids rebuilding
the table when the same spectral, pressure, and temperature grids are used
again.  See :doc:`diffgrid` for the opacity calculation itself.

Worked example
--------------

First, build a small CO DiffGrid table.  The temperature nodes cover the
intended 500--1500 K range and are evenly spaced in inverse temperature.

.. code-block:: python

    from jax import config

    config.update("jax_enable_x64", True)

    import numpy as np

    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaDiffgrid, OpaPremodit, saveopa
    from exojax.utils.grids import wavenumber_grid

    nu_grid, wav, resolution = wavenumber_grid(
        22920.0,
        23000.0,
        512,
        unit="AA",
        xsmode="diffgrid",
    )
    pressure_grid = np.logspace(-4.0, 1.0, 8)
    inverse_temperature_grid = np.linspace(1.0 / 1500.0, 1.0 / 500.0, 9)
    temperature_grid = 1.0 / inverse_temperature_grid

    mdb = MdbExomol(
        ".database/CO/12C-16O/Li2015",
        nurange=nu_grid,
        gpu_transfer=False,
    )
    teacher = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        auto_trange=(500.0, 1500.0),
        broadening_resolution={"mode": "single", "value": None},
    )
    opa = OpaDiffgrid(
        teacher,
        temperature_grid=temperature_grid,
        pressure_grid=pressure_grid,
    )
    opa.check_pressure_grid(pressure_grid)

Save the table with the public :func:`exojax.opacity.saveopa` function.
``aux`` stores small values used by the surrounding model, while
``extra_meta`` records provenance.

.. code-block:: python

    saveopa(
        opa,
        "co_diffgrid.zarr",
        format="zarr",
        aux={"reference_mmr": 1.0e-3},
        extra_meta={"molecule": "CO", "line_list": "Li2015"},
    )

The archive is self-contained.  The following code deletes the teacher and
database, restores the table, and checks an opacity calculation.

.. code-block:: python

    temperature = np.linspace(600.0, 1400.0, pressure_grid.size)
    expected = np.asarray(opa.xsmatrix(temperature))

    del opa, teacher, mdb

    opa = OpaDiffgrid.from_saved_opa("co_diffgrid.zarr")
    opa.check_pressure_grid(pressure_grid)
    actual = np.asarray(opa.xsmatrix(temperature))
    np.testing.assert_allclose(actual, expected)

    print(opa.aux["reference_mmr"])
    print(opa.user_meta["molecule"])

The restored calculator remains compatible with ``jax.jit``, ``jax.grad``,
and ``jax.vmap``.  Inside a compiled model, use
``opa.xsmatrix(temperature)`` without a pressure argument because the
pressure layers are fixed by the table.

Notes
-----

* Zarr is the default format.  For NPZ, use
  ``saveopa(opa, "co_diffgrid.npz", format="npz")``; keep the resulting NPZ
  file and its sibling ``co_diffgrid_metadata.json`` together.
* Loading is strict by default: the ExoJAX version must match and the active
  JAX precision must preserve the saved table-array dtypes.  Enable 64-bit
  mode before loading a 64-bit archive, or use ``strict=False`` only after
  confirming compatibility.
* ``allow_downgrade=True`` relaxes a mismatch in the common opacity archive
  schema only.  The DiffGrid-specific schema, array shapes, units, finite
  values, and integrity digests are still validated.
* Values supplied through ``aux`` and ``extra_meta`` must be JSON-compatible;
  finite NumPy or JAX scalars and arrays are also accepted.
