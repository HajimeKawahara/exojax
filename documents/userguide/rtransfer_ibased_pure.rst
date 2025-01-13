Intensity-based Emission with pure absorption
------------------------------------------------------

Intensity-based (ibased) emission calculation with pure absorption is a method to compute the outgoing flux from the top of the atmosphere,  
with a given temperature-pressure and opacity profile, by transfering **intensity** (not flux) through the layers assuming **no scattering**. 

:math:`I_0 (\mu) = I (\tau_B^\prime, \mu) e^{-\tau_B^\prime} +\int^{\tau_B^\prime}_{0} B (\tau) e^{-\tau^\prime} d \tau^\prime`

:math:`\approx B(T_B) e^{- \tau_{B}/\mu} + \sum_{n=0}^{N-1} B(T_n) (e^{-\tau_{n}/\mu} - e^{-\tau_{n+1}/\mu}).`

For the 
:math:`\mathsf{N}` 
-stream, the outgoing flux can be computed as

:math:`F_\mathrm{out} = 2 \pi \int_0^1 \mu I_0(\mu)  d \mu`

:math:`\approx 2 \pi \sum_{i=1}^{\mathsf{N}^\prime} w_i \mu_i I_0 (\mu_i),`

where 
:math:`\mathsf{N}^\prime = \mathsf{N}/2`
.

Uses ``ArtPureEmis`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform tue ibased calculation using ``ArtEmisPure`` in ExoJAX, 
one simply needs to change the ``rtsolver`` option to either ``ibased`` or ``ibased_linsap``. 

The difference between **ibased** and **ibased_linsap** (ibased w/ linear source approximation) in ExoJAX lies in the assumptions 
regarding the distribution of the source function within the layer. 
The former assumes a uniform source function, while the latter assumes a linearly approximated source function.
The latter can also be described as a linear version of Olson and Kunasz's method.

In the case of ibased, the number of streams (
:math:`\mathsf{N}` 
) can be specified. This is done using the ``nstream`` option (which needs to be an even number).

.. code:: ipython
    
    from exojax.spec.atmrt import ArtEmisPure

    art = ArtEmisPure(pressure_top=1.e-8,
                      pressure_btm=1.e2,
                      nlayer=100,
                      nu_grid=nu_grid, 
                      rtsolver="ibased", # "ibased" or "ibased_linsap"
                      nstream=8)         # specify the number of the streams
    
    art.change_temperature_range(400.0, 1500.0) #sets temperature range
    Tarr = art.powerlaw_temperature(1300.0, 0.1) # sets a powerlaw T-P profile
    mmr_arr = art.constant_mmr_profile(0.1) # sets a constant mass mixing ratio
    gravity = art.constant_gravity_profile(2478.57) #sets a constant gravity 

    # we here call OpaPremodit as opa just to compute xsmatrix 
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[art.Tlow, art.Thigh]) 
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)

    dtau = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, gravity) # computes optical depth profile  
    F0 = art.run(dtau, Tarr) # computes spectrum


Layer-wise computation ``OpartEmisPure``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^