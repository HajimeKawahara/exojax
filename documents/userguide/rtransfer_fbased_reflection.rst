Flux-based Reflection Spectrum
------------------------------------------------------

Refelection Light with No Emission from Atmospheric Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code:: ipython

    import jax.numpy as jnp
    from exojax.spec.atmrt import ArtReflectPure
    
    art = ArtReflectPure(pressure_top=1.e-5,
                         pressure_btm=1.e0,
                         nlayer=200,
                         nu_grid=nu_grid)
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_mmr_profile(0.0001)
    gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      auto_trange=[art.Tlow, art.Thigh])

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass,gravity)

    #almost no scattering from the air
    single_scattering_albedo = jnp.ones_like(dtau) * 0.0001
    asymmetric_parameter = jnp.ones_like(dtau) * 0.0001

    albedo = 0.5
    incoming_flux = jnp.ones_like(nu_grid)
    reflectivity_surface = albedo*jnp.ones_like(nu_grid)
    F0 = art.run(dtau, single_scattering_albedo,
                 asymmetric_parameter, reflectivity_surface, incoming_flux)

    