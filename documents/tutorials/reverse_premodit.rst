HMC-NUTS Retrieval of a Methane High-Resolution Emission Spectrum
=================================================================

In this tutorial, we try to fit the ExoJAX emission model to a mock
high-resolution spectrum. This spectrum was computed assuming a powerlaw
temperature profile and CH4 opacity + CIA.

In this tutorial, we use ``PreMODIT`` as an opacity calculator.

.. code:: ipython3

    """ Reverse modeling of Methane emission spectrum using MODIT
    """
    # coding: utf-8
    import numpy as np
    import matplotlib.pyplot as plt
    from jax import random
    import jax.numpy as jnp
    import pandas as pd
    import pkg_resources

.. code:: ipython3

    # if needed
    #import ssl
    #ssl._create_default_https_context = ssl._create_unverified_context

.. code:: ipython3

    from exojax.spec.atmrt import ArtEmisPure
    from exojax.spec.api import MdbExomol
    from exojax.spec.opacalc import OpaPremodit
    from exojax.spec.contdb import CdbCIA
    from exojax.spec.opacont import OpaCIA
    from exojax.spec.response import ipgauss_sampling
    from exojax.spec.spin_rotation import convolve_rigid_rotation
    from exojax.spec import molinfo
    from exojax.spec.unitconvert import nu2wav
    from exojax.utils.grids import wavenumber_grid
    from exojax.utils.grids import velocity_grid
    from exojax.utils.astrofunc import gravity_jupiter
    from exojax.utils.instfunc import resolution_to_gaussian_std
    from exojax.test.data import SAMPLE_SPECTRA_CH4_NEW


.. parsed-literal::

    /home/exoplanet01/exojax/src/exojax/spec/dtau_mmwl.py:14: FutureWarning: dtau_mmwl might be removed in future.
      warnings.warn("dtau_mmwl might be removed in future.", FutureWarning)


.. code:: ipython3

    # if you wanna use FP64
    #from jax.config import config
    #config.update("jax_enable_x64", True)

We use numpyro as a probabilistic programming language (PPL). We have
other options such as BlackJAX, PyMC etc.

.. code:: ipython3

    # PPL
    import arviz
    from numpyro.diagnostics import hpdi
    from numpyro.infer import Predictive
    from numpyro.infer import MCMC, NUTS
    import numpyro
    import numpyro.distributions as dist

We use a sample data of the methane emission spectrum in ExoJAX,
normlized it, and add Gaussian noise.

.. code:: ipython3

    # loading the data
    filename = pkg_resources.resource_filename(
        'exojax', 'data/testdata/' + SAMPLE_SPECTRA_CH4_NEW)
    dat = pd.read_csv(filename, delimiter=",", names=("wavenumber", "flux"))
    nusd = dat['wavenumber'].values
    flux = dat['flux'].values
    wavd = nu2wav(nusd, wavelength_order="ascending")

.. code:: ipython3

    sigmain = 0.05
    norm = 20000
    nflux = flux / norm + np.random.normal(0, sigmain, len(wavd))

We first set the wavenumber grid enough to cover the observed spectrum.

.. code:: ipython3

    Nx = 7500
    nu_grid, wav, res = wavenumber_grid(np.min(wavd) - 10.0,
                                        np.max(wavd) + 10.0,
                                        Nx,
                                        unit='AA',
                                        xsmode='premodit', wavelength_order="ascending")


.. parsed-literal::

    xsmode =  premodit
    xsmode assumes ESLOG in wavenumber space: mode=premodit
    ======================================================================
    We changed the policy of the order of wavenumber/wavelength grids
    wavenumber grid should be in ascending order and now 
    users can specify the order of the wavelength grid by themselves.
    Your wavelength grid is in ***  ascending  *** order
    This might causes the bug if you update ExoJAX. 
    Note that the older ExoJAX assumes ascending order as wavelength grid.
    ======================================================================


.. parsed-literal::

    /home/exoplanet01/exojax/src/exojax/utils/grids.py:145: UserWarning: Resolution may be too small. R=617160.1067701889
      warnings.warn('Resolution may be too small. R=' + str(resolution),


We would analyze the emission spectrum. So, we use ``ArtEmisPure`` as
``art`` (Atmospheric Radiative Transfer) object.

.. code:: ipython3

    Tlow = 400.0
    Thigh = 1500.0
    art = ArtEmisPure(nu_grid=nu_grid, pressure_top=1.e-8, pressure_btm=1.e2, nlayer=100)
    art.change_temperature_range(Tlow, Thigh)
    Mp = 33.2


.. parsed-literal::

    rtsolver:  fbased2st
    Flux-based two-stream solver, isothermal layer (ExoJAX1, HELIOS-R1 like)


``beta_inst`` is a standard deviation of the instrumental profile.

.. code:: ipython3

    Rinst = 100000.
    beta_inst = resolution_to_gaussian_std(Rinst)

As usual, we make ``mdb`` and ``opa`` for CH4. Because CH4 has a lot of
lines, we choose ``PreMODIT`` as an opacity calculator. What is
``database/CH4/12C-1H4/YT10to10/``? You can check this “name” from the
ExoMol website.

.. code:: ipython3

    ### CH4 setting (PREMODIT)
    mdb = MdbExomol('.database/CH4/12C-1H4/YT10to10/',
                    nurange=nu_grid,
                    gpu_transfer=False)
    print('N=', len(mdb.nu_lines))
    diffmode = 0
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[Tlow, Thigh],
                      dit_grid_resolution=0.2,allow_32bit=True)


.. parsed-literal::

    HITRAN exact name= (12C)(1H)4
    HITRAN exact name= (12C)(1H)4


.. parsed-literal::

    /home/exoplanet01/exojax/src/exojax/utils/molname.py:178: FutureWarning: e2s will be replaced to exact_molname_exomol_to_simple_molname.
      warnings.warn(
    /home/exoplanet01/exojax/src/exojax/utils/molname.py:65: UserWarning: No isotope number identified.
      warnings.warn("No isotope number identified.", UserWarning)
    /home/exoplanet01/exojax/src/exojax/utils/molname.py:65: UserWarning: No isotope number identified.
      warnings.warn("No isotope number identified.", UserWarning)
    /home/exoplanet01/exojax/src/exojax/spec/molinfo.py:28: UserWarning: exact molecule name is not Exomol nor HITRAN form.
      warnings.warn("exact molecule name is not Exomol nor HITRAN form.")
    /home/exoplanet01/exojax/src/exojax/spec/molinfo.py:29: UserWarning: No molmass available
      warnings.warn("No molmass available", UserWarning)


.. parsed-literal::

    Molecule:  CH4
    Isotopologue:  12C-1H4
    Background atmosphere:  H2
    ExoMol database:  None
    Local folder:  .database/CH4/12C-1H4/YT10to10
    Transition files: 
    	 => File 12C-1H4__YT10to10__06000-06100.trans
    	 => File 12C-1H4__YT10to10__06100-06200.trans
    #            i_upper    i_lower    A           nu_lines            gup    jlower    jupper    elower       eupper        Sij0
    0            1033220    1024493    0.00024832  5999.9999849999995  50     12        12        6673.851806  12673.851791  6.323472505220992e-39
    1            1064746    1291636    0.00039538  5999.999983999999   75     13        12        4933.785965  10933.785949  7.117220393764851e-35
    2            1071252    1291978    0.0017425   6000.000033         75     13        12        6546.843546  12546.843579  1.2340167731327585e-37
    3            1071787    1292023    0.00018938  6000.000016         75     13        12        6624.447597  12624.447613  9.197315036051529e-39
    4            1117488    895034     0.0017477   6000.000004999999   75     11        12        5307.511357  11307.511362  5.114842491594696e-35
    ...          ...        ...        ...         ...                 ...    ...       ...       ...          ...           ...
    217,996,053  916862     789269     0.00076054  6199.999923999999   69     10        11        7699.45898   13899.458904  1.711663352498023e-40
    217,996,054  917793     789351     0.0010122   6199.9998590000005  69     10        11        7782.015923  13982.015782  1.5250560431095478e-40
    217,996,055  919787     789549     0.00051839  6199.999945         69     10        11        7988.72336   14188.723305  2.8596894155983876e-41
    217,996,056  97486      134836     3.7486e-05  6199.999949999999   21     4         3         4077.40367   10277.40362   1.136726310326331e-34
    217,996,057  996182     847586     0.011161    6199.99995          125    11        12        7423.072664  13623.072614  1.7438490879626268e-38
    Broadening code level: a1


.. parsed-literal::

    /home/exoplanet01/anaconda3/lib/python3.10/site-packages/radis-0.15-py3.10-linux-x86_64.egg/radis/api/exomolapi.py:607: AccuracyWarning: The default broadening parameter (alpha = 0.0488 cm^-1 and n = 0.4) are used for J'' > 16 up to J'' = 40
      warnings.warn(


.. parsed-literal::

    N= 80505310
    OpaPremodit: params automatically set.
    Robust range: 397.77407283130566 - 1689.7679243628259 K
    Tref changed: 296.0K->1153.6267095763965K


.. parsed-literal::

    /home/exoplanet01/exojax/src/exojax/utils/jaxstatus.py:19: UserWarning: JAX is 32bit mode. We recommend to use 64bit mode. 
    You can change to 64bit mode by writing 
    
        from jax import config 
        config.update("jax_enable_x64", True)
    
      warnings.warn(msg+how_change_msg)
    /home/exoplanet01/exojax/src/exojax/spec/opacalc.py:169: UserWarning: dit_grid_resolution is not None. Ignoring broadening_parameter_resolution.
      warnings.warn(
    /home/exoplanet01/exojax/src/exojax/spec/api.py:390: RuntimeWarning: divide by zero encountered in log
      self.logsij0 = np.log(self.line_strength_ref)


.. parsed-literal::

    OpaPremodit: Tref_broadening is set to  774.5966692414833 K
    # of reference width grid :  3
    # of temperature exponent grid : 2


.. parsed-literal::

    uniqidx: 100%|██████████| 1/1 [00:01<00:00,  1.68s/it]


.. parsed-literal::

    Premodit: Twt= 461.3329793405918 K Tref= 1153.6267095763965 K
    Making LSD:|####################| 100%


As a continuum model, we assume CIA (H2 vs H2). Check how to use ``cdb``
and ``opa``.

.. code:: ipython3

    ## CIA setting
    cdbH2H2 = CdbCIA('.database/H2-H2_2011.cia', nu_grid)
    opcia = OpaCIA(cdb=cdbH2H2, nu_grid=nu_grid)
    mmw = 2.33  # mean molecular weight
    mmrH2 = 0.74
    molmassH2 = molinfo.molmass_isotope('H2')
    vmrH2 = (mmrH2 * mmw / molmassH2)  # VMR


.. parsed-literal::

    H2-H2


.. code:: ipython3

    #settings before HMC
    vsini_max = 100.0
    vr_array = velocity_grid(res, vsini_max)

.. code:: ipython3

    print("ready")


.. parsed-literal::

    ready


Then, we make a function that computes the model spectrum. Here, we use
naive functions of a spin rotation and ``ipgass_sampling``, but you can
also use ``sop`` instead.

.. code:: ipython3

    def frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini):
        g = gravity_jupiter(Rp=Rp, Mp=Mp)  # gravity in the unit of Jupiter
        #molecule
        xsmatrix = opa.xsmatrix(Tarr, art.pressure)
        mmr_arr = art.constant_mmr_profile(MMR_CH4)
        dtaumCH4 = art.opacity_profile_xs(xsmatrix, mmr_arr, opa.mdb.molmass, g)
        #continuum
        logacia_matrix = opcia.logacia_matrix(Tarr)
        dtaucH2H2 = art.opacity_profile_cia(logacia_matrix, Tarr, vmrH2, vmrH2,
                                            mmw, g)
        #total tau
        dtau = dtaumCH4 + dtaucH2H2
        F0 = art.run(dtau, Tarr) / norm
        Frot = convolve_rigid_rotation(F0, vr_array, vsini, u1, u2)
        mu = ipgauss_sampling(nusd, nu_grid, Frot, beta_inst, RV, vr_array)
        return mu

.. code:: ipython3

    import matplotlib.pyplot as plt
    #g = gravity_jupiter(0.88, 33.2)
    Rp = 0.88
    Mp = 33.2
    alpha = 0.1
    MMR_CH4 = 0.0059
    vsini = 20.0
    RV = 10.0
    T0 = 1200.0
    u1 = 0.0
    u2 = 0.0
    Tarr = art.powerlaw_temperature(T0, alpha)
    Ftest = frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini)

.. code:: ipython3

    Tarr = art.powerlaw_temperature(1500.0, alpha)
    Ftest2 = frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini)

.. code:: ipython3

    plt.plot(nusd, nflux)
    plt.plot(nusd, Ftest, ls="dashed")
    plt.plot(nusd, Ftest2, ls="dotted")
    plt.yscale("log")
    plt.show()



.. image:: reverse_premodit_files/reverse_premodit_27_0.png


The following is the numpyro model, i.e. prior and sample.

.. code:: ipython3

    def model_c(y1):
        Rp = numpyro.sample('Rp', dist.Uniform(0.4, 1.2))
        RV = numpyro.sample('RV', dist.Uniform(5.0, 15.0))
        MMR_CH4 = numpyro.sample('MMR_CH4', dist.Uniform(0.0, 0.015))
        T0 = numpyro.sample('T0', dist.Uniform(1000.0, 1500.0))
        alpha = numpyro.sample('alpha', dist.Uniform(0.05, 0.2))
        vsini = numpyro.sample('vsini', dist.Uniform(15.0, 25.0))
        u1 = 0.0
        u2 = 0.0
        Tarr = art.powerlaw_temperature(T0, alpha)
        mu = frun(Tarr, MMR_CH4, Mp, Rp, u1, u2, RV, vsini)
        numpyro.sample('y1', dist.Normal(mu, sigmain), obs=y1)

.. code:: ipython3

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    num_warmup, num_samples = 500, 1000
    #kernel = NUTS(model_c, forward_mode_differentiation=True)
    kernel = NUTS(model_c, forward_mode_differentiation=False)

Let’s run the HMC-NUTS. In my environment, it took ~2 hours using A4500.
We observed here the number of divergences of 179. If you use FP64,
maybe no divergence.

.. code:: ipython3

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, y1=nflux)
    mcmc.print_summary()


.. parsed-literal::

    sample: 100%|██████████| 1500/1500 [2:18:31<00:00,  5.54s/it, 511 steps of size 4.79e-03. acc. prob=0.69]   


.. parsed-literal::

    
                    mean       std    median      5.0%     95.0%     n_eff     r_hat
       MMR_CH4      0.01      0.00      0.01      0.01      0.01    196.80      1.02
            RV      9.40      0.38      9.38      8.80     10.07    231.02      1.00
            Rp      0.57      0.10      0.56      0.42      0.70    196.13      1.02
            T0   1221.15     17.40   1220.48   1192.18   1246.90    270.27      1.00
         alpha      0.11      0.01      0.11      0.10      0.12    153.76      1.00
         vsini     19.68      0.70     19.66     18.48     20.82    203.31      1.00
    
    Number of divergences: 179


.. code:: ipython3

    # SAMPLING
    posterior_sample = mcmc.get_samples()
    pred = Predictive(model_c, posterior_sample, return_sites=['y1'])
    predictions = pred(rng_key_, y1=None)
    median_mu1 = jnp.median(predictions['y1'], axis=0)
    hpdi_mu1 = hpdi(predictions['y1'], 0.9)

.. code:: ipython3

    # PLOT
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 6.0))
    ax.plot(wavd[::-1], median_mu1, color='C0')
    ax.plot(wavd[::-1], nflux, '+', color='black', label='data')
    ax.fill_between(wavd[::-1],
                    hpdi_mu1[0],
                    hpdi_mu1[1],
                    alpha=0.3,
                    interpolate=True,
                    color='C0',
                    label='90% area')
    plt.xlabel('wavelength ($\AA$)', fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=16)
    plt.savefig("pred_diffmode" + str(diffmode) + ".png")
    plt.close()

Draw a corner plot

.. code:: ipython3

    pararr = ['Rp', 'T0', 'alpha', 'MMR_CH4', 'vsini', 'RV']
    arviz.plot_pair(arviz.from_numpyro(mcmc),
                    kind='kde',
                    divergences=False,
                    marginals=True)
    plt.savefig("corner_diffmode" + str(diffmode) + ".png")
    #plt.show()



.. image:: reverse_premodit_files/reverse_premodit_36_0.png

