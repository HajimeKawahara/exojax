Transmission Spectroscopy
------------------------------


Uses ArtTransPure class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To calculate the transmission spectrum in ExoJAX, the `ArtTransPure` class is convenient. 
For chord integration, one can choose either Simpson's method or the trapezoidal method from the integration options.
Here is the example of the computation of the transmission radius. We here use Simpson's rule for the chord integration. 

In transmission spectroscopy, assuming a constant gravity across layers is often not a good approximation. 
The `gravity_profile` instance in the `ArtTransPure` class allows for easy calculation of the gravity profile, addressing this concern in ExoJAX.

.. code:: ipython
    
    from exojax.spec.atmrt import ArtTransPure
    from exojax.utils.constants import RJ

    art = ArtTransPure(pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100, integration="simpson") # integration="trapezoid" if you want
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_mmr_profile(0.1) # constant mass mixing ratio profile 
    mmw = 2.33 * np.ones_like(art.pressure) # mean molecular weight profile
    gravity_btm = 2478.57
    radius_btm = RJ
    gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm) # computes gravity profile

    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      auto_trange=[art.Tlow, art.Thigh])

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    
    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass,gravity)
    Rp2 = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)
    
