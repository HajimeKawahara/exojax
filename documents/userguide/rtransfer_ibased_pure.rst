Intensity-based Emission with pure absorption
------------------------------------------------------

Intensity-based (ibased) emission calculation with pure absorption is a method to compute the outgoing flux from the top of the atmosphere,  
with a given temperature-pressure and opacity profile, by transfering **intensity** (not flux) through the layers assuming **no scattering**. 

:math:`I_0 (\mu) = I (\tau_B^\prime, \mu) e^{-\tau_B^\prime} +\int^{\tau_B^\prime}_{0} B (\tau) e^{-\tau^\prime} d \tau^\prime \approx B(T_B) e^{- \tau_{B}/\mu} + \sum_{n=0}^{N-1} B(T_n) (e^{-\tau_{n}/\mu} - e^{-\tau_{n+1}/\mu}).`

For the 
:math:`\mathsf{N}` 
-stream, the outgoing flux can be comuted by discretizing the flux using the Gauss-Legendre quadrature as

:math:`F_\mathrm{out} = 2 \pi \int_0^1 \mu I_0(\mu)  d \mu \approx 2 \pi \sum_{i=1}^{\mathsf{N}^\prime} w_i \mu_i I_0 (\mu_i),`

where 
:math:`\mathsf{N}^\prime = \mathsf{N}/2`
.

See Section 4.2 in `Paper II <https://arxiv.org/abs/2410.06900>`_ for further details on the intensity-based emission calculation method.


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
    
    from exojax.rt import ArtEmisPure

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

You can further reduce device memory usage with layer-wise computation. 
Please use ``OpartEmisPure`` for this purpose. 

Here is an example. The user needs to define a class that returns the optical depth profile for a given layer.

.. code:: ipython
    
    from exojax.opacity import OpaPremodit
    from exojax.rt import OpartEmisPure
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
            return dtau_co

Then, the user can define the layer-wise computation as follows.
However, users must define the layer's update function themselves. 
This requirement is designed to prevent XLA compilation overhead.

.. code:: ipython
    
    opalayer = OpaLayer(Nnus=100000)
    opart = OpartEmisPure(opalayer, pressure_top=1.0e-5, pressure_btm=1.0e1, nlayer=200, nstream=8)

    def layer_update_function(carry_tauflux, params):
        carry_tauflux = opart.update_layer(carry_tauflux, params)
        return carry_tauflux, None

    temperature = opart.clip_temperature(opart.powerlaw_temperature(1300.0, 0.1))
    mixing_ratio = opart.constant_mmr_profile(0.01)
    layer_params = [temperature, opart.pressure, opart.dParr, mixing_ratio]
    flux = opart(layer_params, layer_update_function)

Refer to the :doc:`../tutorials/get_started_opart` for another example of `OpartEmisPure`.