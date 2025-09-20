Saving Pre‑calculated ``OpaPremodit`` Objects with ``cloudpickle``
==================================================================

*Shotaro Tada, 4/28 (2025)*

When you work with **ExoJAX** over a very wide spectral range, the number of molecular lines can be enormous.  Initialising an :class:`exojax.spec.opacalc.OpaPremodit` instance may therefore take several minutes.  If you need to run the calculation repeatedly under different conditions, it is convenient to serialise the initialised object once and reload it later instead of rebuilding it every time.

Unfortunately, the standard :pymod:`pickle` module sometimes fails to serialise :class:`OpaPremodit` because of internal JIT‑compiled functions.  In that case you will see an error similar to

.. code-block:: text

    _pickle.PicklingError: Can't pickle <function xsvector_zeroth at 0x7be0f3f29b40>: it's not the same object as exojax.spec.premodit.xsvector_zeroth

The solution is to use :pymod:`cloudpickle`, which is able to handle dynamic objects created by **JAX**.

Worked example
--------------

.. code-block:: python

    from exojax.rt import ArtTransPure
    from exojax.utils.grids import wavenumber_grid
    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaPremodit
    import cloudpickle


    dir_save = "output_so2_sio/"

    N = 5000
    print(N)
    nu_grid, wav, res = wavenumber_grid(
        3800.0,
        4300.0,
        N=N,
        unit="nm",
        xsmode="premodit",
    )

    diffmode = 0
    art = ArtTransPure(pressure_top=1.0e-10, pressure_btm=1.0e0, nlayer=60)
    Tlow = 500.0
    Thigh = 1500.0
    art.change_temperature_range(Tlow, Thigh)

    ndiv = 1
    mdb = MdbExomol(
        ".db_ExoMol/SO2/32S-16O2/ExoAmes/", nu_grid, gpu_transfer=False 
    )
    opa = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        nstitch=ndiv,
        diffmode=diffmode,
        auto_trange=[Tlow, Thigh],
        dit_grid_resolution=1,
        allow_32bit=True,
       cutwing=1.0 / (2.0 * ndiv),
    )

    with open("opa.pkl", "wb") as f:
        cloudpickle.dump(opa, f)

    # with open("opa.pkl", "rb") as f:
    #     opa = cloudpickle.load(f)
