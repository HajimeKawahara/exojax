Saving and Loading ``OpaPremodit``
==================================================================

*Hajime Kawahara, 11/9 (2025)*

You can save and load pre-calculated :class:`exojax.spec.opacalc.OpaPremodit` objects using the functions in :mod:`exojax.opacity.io.ioopa`.  
This is useful when you want to avoid re-computing the LSD (Line Shape Database) for the same spectral grid and temperature range.


Worked example
--------------

First, we create an :class:`OpaPremodit` object as usual:

.. code-block:: python

    from jax import config
    config.update("jax_enable_x64", True)

    from exojax.utils.grids import wavenumber_grid

    nu_grid, wav, resolution = wavenumber_grid(
        22920.0, 23000.0, 3500, unit="AA", xsmode="premodit"
    )
    from exojax.database.exomol.api import MdbExomol
    mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid)
    from exojax.opacity import OpaPremodit

    molmass = mdb.molmass # we use molmass later
    snap = mdb.to_snapshot() # extract snapshot from mdb
    del mdb # save the memory

    opa = OpaPremodit.from_snapshot(
        snap,
        nu_grid,
        auto_trange=(500.0, 1500.0),
        dit_grid_resolution=1.0,
    )

Then, we save the `opa` object using :func:`saveopa_premodit`.

.. code-block:: python

    from exojax.opacity.io.ioopa import saveopa_premodit
    saveopa_premodit(opa, "opa.zarr", format="zarr")


To load the saved `opa` object, use the class method :meth:`OpaPremodit.from_saved_opa`.

.. code-block:: python

    from exojax.opacity import OpaPremodit
    opa = OpaPremodit.from_saved_opa("opa.zarr")

Note that when loading the saved `opa`, you need to provide `molmass` separately, since it is not stored in the saved file.