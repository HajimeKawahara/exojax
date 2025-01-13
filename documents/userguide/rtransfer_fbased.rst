Flux-based Reflection, Emission with Scattering
------------------------------------------------------

As of January 2025, the radiative transfer solution in ExoJAX, including scattering and reflection, is based on a flux-based two-stream approximation. 
By default, the method employs the flux-adding treatment (Robinson and Crisp 2018) as the scheme for solving.


The flux-adding treatment solves the following two recurrence relations iteratively from the bottom upwards:

:math:`\hat{R}_n^+ = \mathcal{S}_n + \frac{\mathcal{T}_n^2 \hat{R}_{n+1}^+}{1-\mathcal{S}_n \hat{R}_{n+1}^+}`

:math:`\hat{S}_n^+ &= \hat{\mathcal{B}}_n + \frac{\mathcal{T}_n (\hat{S}_{n+1} + \hat{\mathcal{B}}_n \hat{R}_{n+1}^+)}{1 - \mathcal{S}_n \hat{R}_{n+1}^+}`

where 

:math:`\mathcal{T}_n \equiv \frac{{{\zeta^+_n}}^2 -{{\zeta^-_n}}^2 }{{\zeta^+_n}^2  - (\zeta^-_n\mathsf{T}_n)^2 } \mathsf{T}_n`

:math:`\mathcal{S}_n  \equiv \frac{\zeta^+_n \zeta^-_n }{{\zeta^+_n}^2  - (\zeta^-_n\mathsf{T}_n)^2 } (1-\mathsf{T}_n^2)`

are he transmission between the layer bottom and top and the scattering from the opposite direction of the flux, and

:math:`\mathsf{T}_n \equiv e^{-\lambda_n \Delta \tau_n}`

:math:`\hat{\mathcal{B}}_n \equiv (1 - \mathcal{T}_n - \mathcal{S}_n) \mathcal{B}_n`

:math:`\mathcal{B}^\pm (\tau) = \frac{ 2 (1-\omega_0)}{\gamma_1 - \gamma_2} B_\nu(\tau)`.

The outgoing flux is then computed as

:math:`F_0^+ = \hat{R}^+_0 F_\star + \hat{S}^+_0`,



See Section 4.3 in `Paper II <https://arxiv.org/abs/2410.06900>`_ for further details on the two-stream approximation method.

Robinson and Crisp (2019): Robinson, T. D., & Crisp, D. 2018, JQSRT, 211, 78, doi: 10.1016/j.jqsrt.2018.03.002, 

``ArtReflectPure``: Refelection Light with No Emission from Atmospheric Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


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

    