Cross Section for Many Lines using PreMODIT
===========================================

Update: October 30/2022, Hajime Kawahara

We demonstarte the Precomputation of opacity version of Modified
Discrete Integral Transform (PreMODIT), which is the modified version of
MODIT for exojax. PreMODIT uses the evenly-spaced logarithm grid (ESLOG)
as a wavenumber dimension. PreMODIT takes advantage especially for the
case that the number of the molecular line is large (typically >
100,000).

Here, we use FP64, but if you want you can use FP32 (but slightly large
errors):

.. code:: ipython3

    from jax import config
    config.update("jax_enable_x64", True)

.. code:: ipython3

    import matplotlib.pyplot as plt
    from exojax.database.core.line_strength import line_strength, doppler_sigma, gamma_hitran, gamma_natural
    from exojax.database.exomol.api import MdbExomol 
    from exojax.utils.grids import wavenumber_grid
    from exojax.utils.constants import Tref_original
    # Setting wavenumber bins and loading HITRAN database
    nu_grid, wav, R = wavenumber_grid(1900.0,
                                  2300.0,
                                  350000,
                                  unit="cm-1",
                                  xsmode="premodit")
    isotope=1
    mdbCO = MdbHitran('CO', nu_grid, isotope=isotope)
    
    # set T, P and partition function
    Mmol = mdbCO.molmass
    Tfix = 1000.0  # we assume T=1000K
    Pfix = 1.e-3  # we compute P=1.e-3 bar
    Ppart = Pfix  #partial pressure of CO. here we assume a 100% CO atmosphere.



.. parsed-literal::

    xsmode =  premodit
    xsmode assumes ESLOG in wavenumber space: xsmode=premodit
    ======================================================================
    The wavenumber grid should be in ascending order.
    The users can specify the order of the wavelength grid by themselves.
    Your wavelength grid is in ***  descending  *** order
    ======================================================================
    radis engine =  vaex


We need to precompute some quantities. These can be computed using
initspec.init_premodit. In PreMODIT, we need to specify (Twt and Tref).
You might need to change dE to ensure the precision of the cross
section.

.. code:: ipython3

    from exojax.opacity import initspec
    
    Twt = 1000.0
    Tref_broadening = Tref_original
    dit_grid_resolution = 0.2
    lbd, multi_index_uniqgrid, elower_grid, ngamma_ref_grid, n_Texp_grid, R, pmarray = initspec.init_premodit(
        mdbCO.nu_lines,
        nu_grid,
        mdbCO.elower,
        mdbCO.gamma_air,
        mdbCO.n_air,
        mdbCO.line_strength_ref_original,
        Twt=Twt,
        Tref=Tref_original,
        Tref_broadening=Tref_broadening,
        dit_grid_resolution=dit_grid_resolution,
        diffmode=0,
        warning=False)



.. parsed-literal::

    # of reference width grid :  8
    # of temperature exponent grid : 2


.. parsed-literal::

    uniqidx: 100%|██████████| 6/6 [00:00<00:00, 23109.11it/s]

.. parsed-literal::

    Premodit: Twt= 1000.0 K Tref= 296.0 K
    Making LSD:|####################| 100%


.. parsed-literal::

    


Precompute the normalized Dopper width and the partition function ratio:

.. code:: ipython3

    from exojax.database.core.broadening import normalized_doppler_sigma
    
    molecular_mass = mdbCO.molmass
    nsigmaD = normalized_doppler_sigma(Tfix, molecular_mass, R)
    qt = mdbCO.qr_interp(isotope, Tfix, Tref_original)
        

Let’s compute the cross section! The current PreMODIT has three
different diffmode. We initialized PreMODIT with diffmode=0. Then, we
should use xsvector_zeroth.

.. code:: ipython3

    from exojax.opacity.premodit.premodit import xsvector_zeroth
    xs = xsvector_zeroth(Tfix, Pfix, nsigmaD, lbd, Tref_original, R, pmarray, nu_grid,
                       elower_grid, multi_index_uniqgrid, ngamma_ref_grid,
                       n_Texp_grid, qt, Tref_broadening)
        


.. code:: ipython3

    fig=plt.figure(figsize=(10,5))
    ax=fig.add_subplot(111)
    plt.plot(nu_grid,xs,lw=1,alpha=0.5,label="PreMODIT")
    plt.legend(loc="upper right")
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("cross section (cm2)")
    plt.show()



.. image:: Cross_Section_using_Precomputation_Modified_Discrete_Integral_Transform_files/Cross_Section_using_Precomputation_Modified_Discrete_Integral_Transform_10_0.png


.. code:: ipython3

    from exojax.opacity import OpaDirect
    opa = OpaDirect(mdbCO, nu_grid)
    xsv = opa.xsvector(Tfix, Pfix, Ppart)

.. code:: ipython3

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(211)
    plt.plot(nu_grid, xs, lw=1, alpha=0.5, label="PreMODIT")
    plt.plot(nu_grid, xsv, lw=1, alpha=0.5, label="Direct LPF")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Section (cm2)")
    ax = fig.add_subplot(212)
    plt.plot(nu_grid, xsv - xs, lw=2, alpha=0.5, label="PreMODIT")
    plt.ylabel("LPF - PreMODIT (cm2)")
    plt.legend(loc="upper left")
    plt.show()



.. image:: Cross_Section_using_Precomputation_Modified_Discrete_Integral_Transform_files/Cross_Section_using_Precomputation_Modified_Discrete_Integral_Transform_12_0.png


