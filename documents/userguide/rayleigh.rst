Rayleigh scattering
========================

The cross section of Rayleigh scattering of gas from polarizability is expressed as 

:math:`\sigma(\nu) = \frac{ 128 \pi^5 }{3} \nu^4 \alpha^2 F_k`

where 
:math:`\alpha`
is polarizability and 
:math:`F_k`
is the King factor.


In ExoJAX, the cross section can be computed using `spec.rayleigh.xsvector_rayleigh_gas <../exojax/exojax.spec.html#exojax.spec.rayleigh.xsvector_rayleigh_gas>`_ .

.. code:: ipython3
    
    from exojax.atm.polarizability import polarizability
    from exojax.atm.polarizability import king_correction_factor
    from exojax.utils.grids import wavenumber_grid

    nus, wav, res = wavenumber_grid(
        3000.0, 3100.0, 128, xsmode="premodit", wavelength_order="descending", unit="nm"
    )
    sigma = xsvector_rayleigh_gas(nus, polarizability["CO"], king_correction_factor["CO"])




Lorentz - Lorenz relation
----------------------------

When you need the refractive index, the Lorentz-Lorenz relation `exojax.atm.lorentz_lorenz.refractive_index_Lorentz_Lorenz <../exojax/exojax.atm.html#exojax.atm.lorentz_lorenz.refractive_index_Lorentz_Lorenz>`_ is available. This formulae converts polarizability to refractive index.


Uses OpaRayleigh
-----------------------------

``OpaRayleigh`` is ``opa`` for Rayleigh scattering. It's easy to use.


.. code:: ipython3
    
    N=1000
    nu_grid, wav, res = wavenumber_grid(300, 40000.0, N, xsmode="premodit", unit="nm")
    opa = OpaRayleigh(nu_grid,"N2")
    xs = opa.xsvector()
    