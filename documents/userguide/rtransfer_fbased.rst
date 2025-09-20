Flux-based Reflection, Emission with Scattering
------------------------------------------------------

As of January 2025, the radiative transfer solution in ExoJAX, including scattering and reflection, is based on a flux-based two-stream approximation. 
By default, the method employs the flux-adding treatment (Robinson and Crisp 2018) as the scheme for solving.


The flux-adding treatment solves the following two recurrence relations iteratively from the bottom upwards:

:math:`\hat{R}_n^+ = \mathcal{S}_n + \frac{\mathcal{T}_n^2 \hat{R}_{n+1}^+}{1-\mathcal{S}_n \hat{R}_{n+1}^+}`

:math:`\hat{S}_n^+ = \hat{\mathcal{B}}_n + \frac{\mathcal{T}_n (\hat{S}_{n+1} + \hat{\mathcal{B}}_n \hat{R}_{n+1}^+)}{1 - \mathcal{S}_n \hat{R}_{n+1}^+}`

where 

:math:`\mathcal{T}_n \equiv \frac{{{\zeta^+_n}}^2 -{{\zeta^-_n}}^2 }{{\zeta^+_n}^2  - (\zeta^-_n\mathsf{T}_n)^2 } \mathsf{T}_n`

:math:`\mathcal{S}_n  \equiv \frac{\zeta^+_n \zeta^-_n }{{\zeta^+_n}^2  - (\zeta^-_n\mathsf{T}_n)^2 } (1-\mathsf{T}_n^2)`

are the transmission between the layer bottom and top and the scattering from the opposite direction of the flux, and

:math:`\mathsf{T}_n \equiv e^{-\lambda_n \Delta \tau_n}`

:math:`\hat{\mathcal{B}}_n \equiv (1 - \mathcal{T}_n - \mathcal{S}_n) \mathcal{B}_n`.

:math:`\mathcal{B}_n` can be calculated from the scattering properties and the black-body radiation :math:`B_\nu` of each layer as follows:

:math:`\mathcal{B}_n (\tau) = \frac{ 2 (1-\omega)}{\gamma_1 - \gamma_2} B_\nu(\tau)`.

The coeffcients :math:`\gamma_1, \gamma_2` can be computed using assymetric parameter :math:`g` and single scattering albedo :math:`\omega` as

:math:`\gamma_1 = 2 - \omega (1 + g)` and :math:`\gamma_2 = \omega (1 - g)`.

The outgoing flux is then computed as

:math:`F_0^+ = \hat{R}^+_0 F_\star + \hat{S}^+_0`.


See Section 4.3 in `Paper II <https://arxiv.org/abs/2410.06900>`_ and Robinson and Crisp (2019) JQSRT, 211, 78 for further details on the two-stream approximation method.

Radiative transfer with scattering and reflection can be classified into three types:

- 1. ``ReflectPure``: The scattering and reflection spectrum of incident light, excluding radiation from the atmospheric layers.  
- 2. ``EmisScat``: The self-emission spectrum, including scattering, without incident light.  
- 3. ``ReflectEmis``: The scattering and reflection spectrum of incident light combined with self-emission from the atmosphere.  


Refelection Light with No Emission from Atmospheric Layers: ``ArtReflectPure`` or ``OpartReflectPure``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of ``ArtReflectPure``:

.. code:: python

    import jax.numpy as jnp
    from exojax.rt import ArtReflectPure
    
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

The real example of the reflection spectrum of Jupiter observed by Subaru/petitIRD can be found in the following repository:

- `exojaxample_jupiter <https://github.com/HajimeKawahara/exojaxample_jupiter>`_ as used in `Paper II <https://arxiv.org/abs/2410.06900>`_.


Also, ``OpartReflectPure``, the ``opart`` version of ``ReflectPure``,  is available. We need to define ``OpaLayer`` class.

.. code:: python

    from exojax.opacity import OpaPremodit
    from exojax.rt import OpartReflectPure
    from exojax.rt.layeropacity import single_layer_optical_depth
    from exojax.utils.grids import wavenumber_grid
    from exojax.database.exomol.api import MdbExomol
    from exojax.utils.astrofunc import gravity_jupiter
    import jax.numpy as jnp
    from jax import config
    config.update("jax_enable_x64", True)

    class OpaLayer:
        # user defined class, needs to define self.nugrid
        def __init__(self, Nnus=100000):
            self.nu_grid, self.wav, self.resolution = wavenumber_grid(
                #1900.0, 2300.0, Nnus, unit="cm-1", xsmode="premodit"
                2050.0, 2150.0, Nnus, unit="cm-1", xsmode="premodit"

            )
            self.mdb_co = MdbExomol(".database/CO/12C-16O/Li2015", nurange=self.nu_grid)
            self.opa_co = OpaPremodit(
                self.mdb_co,
                self.nu_grid,
                auto_trange=[500.0, 1500.0],
                dit_grid_resolution=1.0,
            )
            self.gravity = gravity_jupiter(1.0, 10.0)

        def __call__(self, params):
            temperature, pressure, dP, mixing_ratio = params
            xsv_co = self.opa_co.xsvector(temperature, pressure)
            dtau_co = single_layer_optical_depth(
                dP, xsv_co, mixing_ratio, self.mdb_co.molmass, self.gravity
            )
            single_scattering_albedo = jnp.ones_like(dtau_co) * 0.3
            asymmetric_parameter = jnp.ones_like(dtau_co) * 0.01
            return dtau_co, single_scattering_albedo, asymmetric_parameter

In addition, we need to define the layer update function, same as :doc:`rtransfer_ibased_pure`. 

.. code:: python

    opalayer = OpaLayer(Nnus=100000)
    opart = OpartReflectPure(opalayer, pressure_top=1.0e-5, pressure_btm=1.0e1, nlayer=20000)
    opart.change_temperature_range(400.0, 1500.0)
    def layer_update_function(carry_tauflux, params):
        carry_tauflux = opart.update_layer(carry_tauflux, params)
        return carry_tauflux, None

    temperature = opart.clip_temperature(opart.powerlaw_temperature(1300.0, 0.1))
    mixing_ratio = opart.constant_mmr_profile(0.0003)
    layer_params = [temperature, opart.pressure, opart.dParr, mixing_ratio]
    albedo = 1.0
    incoming_flux = jnp.ones_like(opalayer.nu_grid)
    reflectivity_surface = albedo * jnp.ones_like(opalayer.nu_grid)

    flux = opart(
        layer_params, layer_update_function, reflectivity_surface, incoming_flux
    )    
