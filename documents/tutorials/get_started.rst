Get Started
===========

First, we recommend FP64 unless you can think precision seriously. Use
jax.config to set FP64:

.. code:: ipython3

    from jax.config import config
    config.update("jax_enable_x64", True)

The following schematic figure explains how ExoJAX works; (1) loading
databases (*db), (2) calculating opacity (opa), (3) running atmospheric
radiative transfer (art), (4) applying operations on the spectrum (sop)

.. code:: ipython3

    from IPython.display import Image
    Image("../exojax.png")




.. image:: get_started_files/get_started_4_0.png



1. Loading a molecular database using mdb.
------------------------------------------

ExoJAX has an API for molecular databases, called “mdb” (or “adb” for
atomic datbases). Prior to loading the database, define the wavenumber
range first.

.. code:: ipython3

    from exojax.utils.grids import wavenumber_grid
    
    nu_grid, wav, resolution = wavenumber_grid(1900.,
                                               2300.,
                                               100000,
                                               unit="cm-1",
                                               xsmode="premodit")



.. parsed-literal::

    xsmode =  premodit
    xsmode assumes ESLOG in wavenumber space: mode=premodit


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/grids.py:126: UserWarning: Resolution may be too small. R=523403.606697253
      warnings.warn('Resolution may be too small. R=' + str(resolution),


Then, let’s load the molecular database. We here use Carbon monooxide in
Exomol. CO/12C-16O/Li2015 means Carbon monooxide/ isotopes = 12C + 16O /
database name. You can check the database name in the ExoMol website
(https://www.exomol.com/).

.. code:: ipython3

    from exojax.spec.api import MdbExomol
    
    mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid)



.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/molname.py:133: FutureWarning: e2s will be replaced to exact_molname_exomol_to_simple_molname.
      warnings.warn(


.. parsed-literal::

    HITRAN exact name= (12C)(16O)
    Background atmosphere:  H2
    Reading .database/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2
    .broad is used.
    Broadening code level= a0
    default broadening parameters are used for  71  J lower states in  152  states


2. Computation of the Cross Section using opa
---------------------------------------------

ExoJAX has various opacity calculator classes, so-called “opa”. Here, we
use a memory-saved opa, OpaPremodit. We assume the robust tempreature
range we will use is 500-1500K.

.. code:: ipython3

    from exojax.spec.opacalc import OpaPremodit
    
    opa = OpaPremodit(mdb, nu_grid, auto_trange=[500.0, 1500.0])


.. parsed-literal::

    OpaPremodit: params automatically set.
    Robust range: 484.50562701065246 - 1804.6009417674848 K
    Tref changed: 296.0K->521.067611616332K
    Tref_broadening is set to  866.0254037844389 K
    # of reference width grid :  4
    # of temperature exponent grid : 2


.. parsed-literal::

    uniqidx: 100%|██████████| 2/2 [00:00<00:00, 5174.96it/s]

.. parsed-literal::

    Premodit: Twt= 1153.8856089961712 K Tref= 521.067611616332 K


.. parsed-literal::

    


.. parsed-literal::

    Making LSD:|####################| 100%
    Making LSD:|####################| 100%
    Making LSD:|####################| 100%


Then let’s compute cross section for two different temperature 500 and
1500 K for P=1.0 bar. opa.xsvector can do that!

.. code:: ipython3

    P = 1.0 #bar
    T_1 = 500.0 #K
    xsv_1 = opa.xsvector(T_1, P) #cm2
    
    T_2 = 1500.0 #K
    xsv_2 = opa.xsvector(T_2, P) #cm2

Plot them. It can be seen that different lines are stronger at different
temperatures.

.. code:: ipython3

    import matplotlib.pyplot as plt
    plt.plot(nu_grid,xsv_1,label=str(T_1)+"K") #cm2
    plt.plot(nu_grid,xsv_2,alpha=0.5,label=str(T_2)+"K") #cm2
    plt.legend()
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("cross section (cm2)")
    plt.show()



.. image:: get_started_files/get_started_16_0.png


You can also plot the line strengths at T=1500K. We can first change the
mdb reference temperature and then plot the line intensity.

.. code:: ipython3

    mdb.change_reference_temperature(T_2)
    plt.plot(mdb.nu_lines,mdb.line_strength_ref,".")
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("line strength (cm)")
    plt.yscale("log")
    plt.show()


.. parsed-literal::

    Tref changed: 521.067611616332K->1500.0K



.. image:: get_started_files/get_started_18_1.png


3. Atmospheric Radiative Transfer
---------------------------------

ExoJAX can solve the radiative transfer and derive the emission
spectrum. To do so, ExoJAX has “art” class. ArtEmisPure means
Atomospheric Radiative Transfer for Emission with Pure absorption. So,
ArtEmisPure does not include scattering. We set the number of the
atmospheric layer to 100 (nlayer) and the pressure at bottom and top
atmosphere to 100 and 1.e-8 bar.

.. code:: ipython3

    from exojax.spec.atmrt import ArtEmisPure
    art = ArtEmisPure(nu_grid=nu_grid, pressure_btm=1.e2, pressure_top=1.e-8, nlayer=100)



.. parsed-literal::

    /home/kawahara/exojax/src/exojax/spec/dtau_mmwl.py:14: FutureWarning: dtau_mmwl might be removed in future.
      warnings.warn("dtau_mmwl might be removed in future.", FutureWarning)


Let’s assume the power law temperature model, within 500 - 1500 K.

:math:`T = T_0 P^\alpha`

where :math:`T_0=1200` K and :math:`\alpha=0.1`.

.. code:: ipython3

    art.change_temperature_range(500.0, 1500.0)
    Tarr = art.powerlaw_temperature(1200.0,0.1)

Also, the mass mixing ratio of CO (MMR) should be defined.

.. code:: ipython3

    mmr_profile = art.constant_mmr_profile(0.01)

Surface gravity is also important quantity of the atmospheric model,
which is a function of planetary radius and mass. Here we assume 1 RJ
and 10 MJ.

.. code:: ipython3

    from exojax.utils.astrofunc import gravity_jupiter
    gravity = gravity_jupiter(1.0,10.0)

In addition to the CO cross section, we would consider `collisional
induced
absorption <https://en.wikipedia.org/wiki/Collision-induced_absorption_and_emission>`__
(CIA) as a continuum opacity. “cdb” class can be used.

.. code:: ipython3

    from exojax.spec.contdb import CdbCIA
    from exojax.spec.opacont import OpaCIA
    
    cdb = CdbCIA(".database/H2-H2_2011.cia",nurange=nu_grid)
    opacia = OpaCIA(cdb, nu_grid=nu_grid)


.. parsed-literal::

    H2-H2


Before running the radiative transfer, we need cross sections for
layers, called xsmatrix for CO and logacia_matrix for CIA (strictly
speaking, the latter is not cross section but coefficient because CIA
intensity is proportional density square).

.. code:: ipython3

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    logacia_matrix = opacia.logacia_matrix(Tarr)

Convert them to opacity

.. code:: ipython3

    dtau_CO = art.opacity_profile_lines(xsmatrix, mmr_profile, mdb.molmass, gravity)
    vmrH2 = 0.855 #VMR of H2
    mmw = 2.33 # mean molecular weight of the atmosphere
    dtaucia = art.opacity_profile_cia(logacia_matrix, Tarr, vmrH2, vmrH2, mmw, gravity)

Add two opacities.

.. code:: ipython3

    dtau = dtau_CO + dtaucia

Then, run the radiative transfer

.. code:: ipython3

    F = art.run(dtau, Tarr)
    
    fig=plt.figure(figsize=(15,4))
    plt.plot(nu_grid,F)
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("flux (erg/s/cm2/cm-1)")
    plt.show()



.. image:: get_started_files/get_started_37_0.png


You can check the contribution function too!

.. code:: ipython3

    from exojax.plot.atmplot import plotcf

.. code:: ipython3

    cf=plotcf(nu_grid, dtau, Tarr,art.pressure, art.dParr)



.. image:: get_started_files/get_started_40_0.png


Spectral Operators: rotational broadening, instrumental profile, Doppler velocity shift and so on, any operation on spectra.
----------------------------------------------------------------------------------------------------------------------------

The above spectrum is called “raw spectrum” in ExoJAX. The effects
applied to the raw spectrum is handled in ExoJAX by the spectral
operator (sop). First, we apply the spin rotation of a planet.

.. code:: ipython3

    from exojax.spec.specop import SopRotation
    sop_rot = SopRotation(nu_grid, resolution, vsini_max=100.0)
    
    vsini = 50.0
    u1=0.0
    u2=0.0 
    Frot = sop_rot.rigid_rotation(F, vsini, u1, u2) 


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/grids.py:126: UserWarning: Resolution may be too small. R=523403.606697253
      warnings.warn('Resolution may be too small. R=' + str(resolution),


.. code:: ipython3

    fig=plt.figure(figsize=(15,4))
    plt.plot(nu_grid,F, label="raw spectrum")
    plt.plot(nu_grid,Frot, label="rotated")
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("flux (erg/s/cm2/cm-1)")
    plt.legend()
    plt.show()



.. image:: get_started_files/get_started_44_0.png


Then, the instrumental profile with relative radial velocity is applied.

.. code:: ipython3

    from exojax.spec.specop import SopInstProfile
    from exojax.utils.instfunc import resolution_to_gaussian_std
    sop_inst = SopInstProfile(nu_grid, resolution, vrmax=1000.0)
    
    RV=40.0 #km/s
    resolution_inst = 3000.0
    beta_inst = resolution_to_gaussian_std(resolution_inst)
    Finst = sop_inst.ipgauss(Frot, beta_inst)
    nu_obs = nu_grid[::50]
    Fobs = sop_inst.sampling(Finst, RV, nu_obs)


.. parsed-literal::

    42.43671169022172


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/grids.py:126: UserWarning: Resolution may be too small. R=523403.606697253
      warnings.warn('Resolution may be too small. R=' + str(resolution),


.. code:: ipython3

    fig=plt.figure(figsize=(15,4))
    plt.plot(nu_grid,Frot, label="rotated")
    plt.plot(nu_grid,Finst, label="rotated+IP")
    plt.plot(nu_obs,Fobs, ".", label="rotated+IP (sampling)")
    
    
    plt.xlabel("wavenumber (cm-1)")
    plt.ylabel("flux (erg/s/cm2/cm-1)")
    plt.legend()
    plt.show()



.. image:: get_started_files/get_started_47_0.png


That’s it.


