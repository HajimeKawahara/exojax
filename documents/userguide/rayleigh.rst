Rayleigh scattering
========================

Rayleigh scattering of gas from polarizability can be computed using `spec.rayleigh.xsvector_rayleigh_gas <../exojax/exojax.spec.html#exojax.spec.rayleigh.xsvector_rayleigh_gas>`_ .

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

