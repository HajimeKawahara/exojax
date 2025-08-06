Forward Modeling of an Emission Spectrum using the DIT Cross Section
====================================================================

.. code:: ipython3

    from exojax.rt import rtransfer as rt
    from exojax.opacity.modit import dit
    from exojax.opacity.lpf import lpf

.. code:: ipython3

    #ATMOSPHERE                                                                     
    NP=100
    T0=1295.0 #K
    Parr, dParr, k=rt.pressure_layer(NP=NP)
    Tarr = T0*(Parr)**0.1

A T-P profile we assume is …

.. code:: ipython3

    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    plt.plot(Tarr,Parr)
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.show()



.. image:: Forward_modeling_using_the_DIT_Cross_Section_for_methane_files/Forward_modeling_using_the_DIT_Cross_Section_for_methane_4_0.png


We set a wavenumber grid using wavenumber_grid.

.. code:: ipython3

    from exojax.utils.grids import wavenumber_grid
    nus,wav,res=wavenumber_grid(16360,16560,10000,unit="AA",xsmode="dit")


.. parsed-literal::

    nugrid is linear: mode= dit


Loading a molecular database of CH4 and CIA (H2-H2)…

.. code:: ipython3

    from exojax.database.exomol.api import MdbExomol , contdb
    mdbCH4=exomol.api.MdbExomol('.database/CH4/12C-1H4/YT10to10/',nus,crit=1.e-30)
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)


.. parsed-literal::

    Background atmosphere:  H2
    Note: Couldn't find the hdf5 format. We convert data to the hdf5 format. After the second time, it will become much faster.
    Reading transition file
    .broad is used.
    Broadening code level= a1
    default broadening parameters are used for  12  J lower states in  29  states
    H2-H2


.. code:: ipython3

    len(mdbCH4.A)




.. parsed-literal::

    140031



.. code:: ipython3

    from exojax.database import molinfo 
    molmassCH4=molinfo.molmass("CH4")

Computing the relative partition function,

.. code:: ipython3

    from jax import vmap
    qt=vmap(mdbCH4.qr_interp)(Tarr)

Pressure and Natural broadenings

.. code:: ipython3

    from jax import jit
    from exojax.database.core.broadening  import gamma_exomol
    from exojax.database.core.broadening import gamma_natural
    
    gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
            (Parr,Tarr,mdbCH4.n_Texp,mdbCH4.alpha_ref)
    gammaLMN=gamma_natural(mdbCH4.A)
    gammaLM=gammaLMP+gammaLMN[None,:]

Doppler broadening

.. code:: ipython3

    from exojax.database.core.broadening import doppler_sigma
    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
            (mdbCH4.nu_lines,Tarr,molmassCH4)

And line strength

.. code:: ipython3

    from exojax.database.core.broadening import SijT
    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
        (Tarr,mdbCH4.logsij0,mdbCH4.nu_lines,mdbCH4.elower,qt)

DIT

.. code:: ipython3

    dgm_sigmaD=dit.dgmatrix(sigmaDM,0.1)
    dgm_gammaL=dit.dgmatrix(gammaLM,0.2)

.. code:: ipython3

    #show the DIT grids 
    from exojax.plot.ditplot import plot_dgm
    plot_dgm(dgm_sigmaD,dgm_gammaL,sigmaDM,gammaLM,0,6)



.. image:: Forward_modeling_using_the_DIT_Cross_Section_for_methane_files/Forward_modeling_using_the_DIT_Cross_Section_for_methane_21_0.png


.. code:: ipython3

    from exojax.opacity import initspec 
    cnu,indexnu,pmarray=initspec.init_dit(mdbCH4.nu_lines,nus)
    xsmdit=dit.xsmatrix(cnu,indexnu,pmarray,sigmaDM,gammaLM,SijM,nus,dgm_sigmaD,dgm_gammaL)

.. code:: ipython3

    import numpy as np
    fig=plt.figure(figsize=(20,4))
    ax=fig.add_subplot(111)
    c=plt.imshow(np.log10(xsmdit),cmap="bone_r",vmin=-23,vmax=-19)
    plt.colorbar(c,shrink=0.8)
    plt.text(50,30,"DIT")
    ax.set_aspect(0.4/ax.get_data_ratio())
    plt.show()


.. parsed-literal::

    /tmp/ipykernel_31797/3074525130.py:4: RuntimeWarning: divide by zero encountered in log10
      c=plt.imshow(np.log10(xsmdit),cmap="bone_r",vmin=-23,vmax=-19)
    /tmp/ipykernel_31797/3074525130.py:4: RuntimeWarning: invalid value encountered in log10
      c=plt.imshow(np.log10(xsmdit),cmap="bone_r",vmin=-23,vmax=-19)



.. image:: Forward_modeling_using_the_DIT_Cross_Section_for_methane_files/Forward_modeling_using_the_DIT_Cross_Section_for_methane_23_1.png


computing delta tau for CH4

.. code:: ipython3

    from exojax.rt.rtransfer import dtauM
    import jax.numpy as jnp
    Rp=0.88
    Mp=33.2
    g=2478.57730044555*Mp/Rp**2
    #g=1.e5 #gravity cm/s2
    MMR=0.0059 #mass mixing ratio
    
    # 0-padding for negative values
    xsmnp=np.array(xsmdit)
    print(len(xsmnp[xsmnp<0.0]))
    xsmnp[xsmnp<0.0]=0.0
    xsmditc=jnp.array(xsmnp)
    #-------------------------------
    
    dtaum=dtauM(dParr,xsmditc,MMR*np.ones_like(Tarr),molmassCH4,g)


.. parsed-literal::

    4222


computing delta tau for CIA

.. code:: ipython3

    from exojax.rt.rtransfer import dtauCIA
    mmw=2.33 #mean molecular weight
    mmrH2=0.74
    molmassH2=molinfo.molmass("H2")
    vmrH2=(mmrH2*mmw/molmassH2) #VMR
    dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,\
                mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)

The total delta tau is a summation of them

.. code:: ipython3

    dtau=dtaum+dtaucH2H2

you can plot a contribution function using exojax.plot.atmplot

.. code:: ipython3

    from exojax.plot.atmplot import plotcf
    plotcf(nus,dtau,Tarr,Parr,dParr)
    plt.show()



.. image:: Forward_modeling_using_the_DIT_Cross_Section_for_methane_files/Forward_modeling_using_the_DIT_Cross_Section_for_methane_31_0.png


radiative transfering…

.. code:: ipython3

    from exojax.rt import planck
    from exojax.rt.rtransfer import rtrun
    sourcef = planck.piBarr(Tarr,nus)
    F0=rtrun(dtau,sourcef)

.. code:: ipython3

    fig=plt.figure(figsize=(20,4))
    ax=fig.add_subplot(211)
    plt.plot(wav[::-1],F0,lw=1,label="DIT")
    plt.legend()
    plt.xlabel("wavelength ($\AA$)")
    plt.savefig("ch4.png")



.. image:: Forward_modeling_using_the_DIT_Cross_Section_for_methane_files/Forward_modeling_using_the_DIT_Cross_Section_for_methane_34_0.png


