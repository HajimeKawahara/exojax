Forward Modelling of a Spectrum using PreMODIT and Comparison with LPF
======================================================================

Here, we try to compute a emission spectrum using PreMODIT. We recommend
to use FP64 as follows:

.. code:: ipython3

    from jax import config
    config.update("jax_enable_x64", True)

.. code:: ipython3

    from exojax.rt import rtransfer as rt
    from exojax.opacity.premodit.premodit import premodit
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('bmh')

.. code:: ipython3

    #ATMOSPHERE                                                                     
    NP=100
    T0=1295.0 #K
    Parr, dParr, k=rt.pressure_layer(NP=NP, numpy=True)
    Tarr = T0*(Parr)**0.1
    Tarr[Tarr<400.0] = 400.0
    Tarr[Tarr>1500.0]=1500.0

We set a wavenumber grid using wavenumber_grid. Specify
xsmode=“premodit” though it is not mandatory. PreMODIT uses FFT, so the
(internal) wavenumber grid should be logarithm.

.. code:: ipython3

    from exojax.utils.grids import wavenumber_grid
    nus,wav,R=wavenumber_grid(22900,23000,10000,unit="AA",xsmode="premodit")


.. parsed-literal::

    xsmode =  premodit
    xsmode assumes ESLOG in wavenumber space: mode=premodit


Loading a molecular database of CO and CIA (H2-H2)… For PreMODIT,
gpu_transfer=False can save the device memory use.

.. code:: ipython3

    from exojax.database.exomol.api import MdbExomol , contdb
    mdbCO=exomol.api.MdbExomol('.database/CO/12C-16O/Li2015',nus,gpu_transfer=False)
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)


.. parsed-literal::

    HITRAN exact name= (12C)(16O)
    Background atmosphere:  H2
    Reading .database/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2
    .broad is used.
    Broadening code level= a0
    default broadening parameters are used for  71  J lower states in  152  states
    H2-H2


.. code:: ipython3

    molmassCO=mdbCO.molmass

.. code:: ipython3

    from exojax.opacity import OpaPremodit
    diffmode = 0
    opa = OpaPremodit(mdb=mdbCO,
                          nu_grid=nus,
                          diffmode=diffmode,
                          auto_trange=[400.0, 1500.0])
        


.. parsed-literal::

    OpaPremodit: params automatically set.
    Robust range: 397.77407283130566 - 1689.7679243628259 K
    Tref changed: 296.0K->1153.6267095763965K


.. parsed-literal::

    uniqidx: 100%|██████████| 1/1 [00:00<00:00, 7516.67it/s]


.. parsed-literal::

    Premodit: Twt= 461.3329793405918 K Tref= 1153.6267095763965 K


Let’s compute a cross section matrix, i.e. cross sections in all of the
layers.

.. code:: ipython3

    xsm = opa.xsmatrix(Tarr, Parr)    

.. code:: ipython3

    xsm




.. parsed-literal::

    DeviceArray([[1.47016232e-32, 1.48565634e-32, 1.50140347e-32, ...,
                  1.86423775e-35, 1.86310102e-35, 1.86308484e-35],
                 [1.85513450e-32, 1.87468595e-32, 1.89455653e-32, ...,
                  2.35215235e-35, 2.35137451e-35, 2.35123660e-35],
                 [2.34091428e-32, 2.36558562e-32, 2.39065925e-32, ...,
                  2.96747766e-35, 2.96720989e-35, 2.96697031e-35],
                 ...,
                 [2.50140371e-22, 2.50333571e-22, 2.50526616e-22, ...,
                  2.30743374e-23, 2.30652451e-23, 2.30561585e-23],
                 [2.36326890e-22, 2.36473336e-22, 2.36619707e-22, ...,
                  2.86537631e-23, 2.86427012e-23, 2.86316460e-23],
                 [2.23454067e-22, 2.23566292e-22, 2.23678481e-22, ...,
                  3.52741762e-23, 3.52609784e-23, 3.52477883e-23]],            dtype=float64)



Then, let’s compute the opacity delta tau. Here, we need to assume
gravity and Mass Mixing Ratio :)

.. code:: ipython3

    from exojax.rt.rtransfer import dtauM
    g = 2478.57 # gravity
    MMR = 0.1
    dtau = dtauM(dParr, xsm, MMR * np.ones_like(Parr), molmassCO, g)

We also compute the cross section using the direct computation (LPF) for
the comparison purpose.

.. code:: ipython3

    #direct LPF for comparison
    
    #Reload mdb beacuse we need gpu_transfer for LPF. This makes big difference in the device memory use. 
    mdbCO=exomol.api.MdbExomol('.database/CO/12C-16O/Li2015',nus, gpu_transfer=True)
    
    
    #we need sigmaDM for LPF
    from exojax.database.core.broadening import doppler_sigma
    from jax import jit
    from exojax.opacity.initspec import init_lpf
    from exojax.opacity.lpf.lpf import xsmatrix as xsmatrix_lpf
    from exojax.database.core.broadening  import gamma_exomol
    from exojax.database.core.broadening import gamma_natural
    from exojax.database.core.broadening import SijT
    from jax import vmap
    
    qt = vmap(mdbCO.qr_interp)(Tarr)
    
    # Strength, Dopper width, and Lorentian width
    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
        (Tarr,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
            (mdbCO.nu_lines,Tarr,molmassCO)
    gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
            (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
    gammaLMN=gamma_natural(mdbCO.A)
    gammaLM=gammaLMP+gammaLMN[None,:]
    
    numatrix=init_lpf(mdbCO.nu_lines,nus)
    xsmdirect=xsmatrix_lpf(numatrix,sigmaDM,gammaLM,SijM)


.. parsed-literal::

    HITRAN exact name= (12C)(16O)
    Background atmosphere:  H2
    Reading .database/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2
    .broad is used.
    Broadening code level= a0
    default broadening parameters are used for  71  J lower states in  152  states


Let’s see the cross section matrix!

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(20,4))
    ax=fig.add_subplot(211)
    c=plt.imshow(np.log10(xsm),cmap="bone_r",vmin=-23,vmax=-19)
    plt.colorbar(c,shrink=0.8)
    plt.text(50,30,"PreMODIT")
    
    ax.set_aspect(0.1/ax.get_data_ratio())
    ax=fig.add_subplot(212)
    c=plt.imshow(np.log10(xsmdirect),cmap="bone_r",vmin=-23,vmax=-19)
    plt.colorbar(c,shrink=0.8)
    plt.text(50,30,"DIRECT")
    ax.set_aspect(0.1/ax.get_data_ratio())
    plt.show()



.. image:: Forward_modeling_using_PreMODIT_files/Forward_modeling_using_PreMODIT_19_0.png


.. code:: ipython3

    from exojax.rt import planck
    from exojax.rt.rtransfer import rtrun
    sourcef = planck.piBarr(Tarr,nus)
    F0=rtrun(dtau,sourcef)
    
    
    #also for LPF
    dtaumdirect=dtauM(dParr,xsmdirect,MMR*np.ones_like(Tarr),molmassCO,g)
    F0direct=rtrun(dtaumdirect,sourcef)

The difference is very small except around the edge (even for this it’s
only 1%).

.. code:: ipython3

    fig=plt.figure()
    ax=fig.add_subplot(211)
    plt.plot(wav[::-1],F0,label="PreMODIT")
    plt.plot(wav[::-1],F0direct,ls="dashed",label="direct")
    plt.legend()
    ax=fig.add_subplot(212)
    plt.plot(wav[::-1],(F0-F0direct)/np.median(F0direct)*100,label="PreMODIT")
    plt.legend()
    #plt.ylim(-0.1,0.1)
    plt.ylabel("residual (%)")
    plt.xlabel("wavelength ($\AA$)")
    plt.show()



.. image:: Forward_modeling_using_PreMODIT_files/Forward_modeling_using_PreMODIT_22_0.png


applying an instrumental response and planet/stellar rotation to the raw
spectrum

.. code:: ipython3

    from exojax.postproc import response
    from exojax.utils.constants import c
    import jax.numpy as jnp
    
    wavd=jnp.linspace(22920,23000,500) #observational wavelength grid
    nusd = 1.e8/wavd[::-1]
    
    RV=10.0 #RV km/s
    vsini=20.0 #Vsini km/s
    u1=0.0 #limb darkening u1
    u2=0.0 #limb darkening u2
    
    Rinst=100000.
    beta=c/(2.0*np.sqrt(2.0*np.log(2.0))*Rinst) #IP sigma need check 
    
    Frot=response.rigidrot(nus,F0,vsini,u1,u2)
    F=response.ipgauss_sampling(nusd,nus,Frot,beta,RV)


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/spec/response.py:22: UserWarning: rigidrot is deprecated and do not work for VJP. Use convolve_rigid_rotation instead.
      warnings.warn(


.. code:: ipython3

    plt.plot(wav[::-1],F0)
    plt.plot(wavd[::-1],F)
    plt.xlim(22920,23000)




.. parsed-literal::

    (22920.0, 23000.0)




.. image:: Forward_modeling_using_PreMODIT_files/Forward_modeling_using_PreMODIT_25_1.png



