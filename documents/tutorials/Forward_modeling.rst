Forward Modeling of an Emission Spectrum
========================================

.. code:: ipython3

    from exojax.rt import rtransfer as rt

.. code:: ipython3

    #ATMOSPHERE                                                                     
    NP=100
    T0=1295.0 #K
    Parr, dParr, k=rt.pressure_layer(NP=NP)
    Tarr = T0*(Parr)**0.1

A T-P profile we assume is …

.. code:: ipython3

    import matplotlib.pyplot as plt
    plt.plot(Tarr,Parr)
    plt.yscale("log")
    plt.gca().invert_yaxis()
    plt.show()



.. image:: Forward_modeling_files/Forward_modeling_4_0.png


We set a wavenumber grid using wavenumber_grid.

.. code:: ipython3

    from exojax.utils.grids import wavenumber_grid
    nus,wav,res=wavenumber_grid(22920,23000,1000,unit="AA")


.. parsed-literal::

    xsmode assumes ESLOG in wavenumber space: mode=lpf


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/grids.py:124: UserWarning: Resolution may be too small. R=286712.70993002696
      warnings.warn('Resolution may be too small. R=' + str(resolution),


Loading a molecular database of CO and CIA (H2-H2)…

.. code:: ipython3

    from exojax.spec import api, contdb
    mdbCO=api.MdbExomol('.database/CO/12C-16O/Li2015',nus,crit=1.e-46)
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)


.. parsed-literal::

    Background atmosphere:  H2
    Reading .database/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2
    .broad is used.
    Broadening code level= a0
    H2-H2


.. code:: ipython3

    from exojax.spec import molinfo
    molmassCO=molinfo.molmass("CO")

Computing the relative partition function,

.. code:: ipython3

    from jax import vmap
    qt=vmap(mdbCO.qr_interp)(Tarr)

Pressure and Natural broadenings

.. code:: ipython3

    from jax import jit
    from exojax.spec.exomol import gamma_exomol
    from exojax.spec import gamma_natural
    
    gammaLMP = jit(vmap(gamma_exomol,(0,0,None,None)))\
            (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
    gammaLMN=gamma_natural(mdbCO.A)
    gammaLM=gammaLMP+gammaLMN[None,:]

Doppler broadening

.. code:: ipython3

    from exojax.spec import doppler_sigma
    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
            (mdbCO.nu_lines,Tarr,molmassCO)

And line strength

.. code:: ipython3

    from exojax.spec import SijT
    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
        (Tarr,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)

nu matrix

.. code:: ipython3

    from exojax.opacity import make_numatrix0
    numatrix=make_numatrix0(nus,mdbCO.nu_lines)

Or you can use initspec.init_lpf instead.

.. code:: ipython3

    #Or you can use initspec.init_lpf instead.
    from exojax.opacity import initspec
    numatrix=initspec.init_lpf(mdbCO.nu_lines,nus)

Providing numatrix, thermal broadening, gamma, and line strength, we can
compute cross section.

.. code:: ipython3

    from exojax.opacity.lpf import xsmatrix
    xsm=xsmatrix(numatrix,sigmaDM,gammaLM,SijM)

xsmatrix has the shape of (# of layers, # of nu grid)

.. code:: ipython3

    import numpy as np
    np.shape(xsm)




.. parsed-literal::

    (100, 1000)



.. code:: ipython3

    import numpy as np
    plt.imshow(xsm,cmap="afmhot")
    plt.show()



.. image:: Forward_modeling_files/Forward_modeling_26_0.png


computing delta tau for CO

.. code:: ipython3

    from exojax.rt.rtransfer import dtauM
    Rp=0.88
    Mp=33.2
    g=2478.57730044555*Mp/Rp**2
    #g=1.e5 #gravity cm/s2
    MMR=0.0059 #mass mixing ratio
    dtaum=dtauM(dParr,xsm,MMR*np.ones_like(Tarr),molmassCO,g)


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



.. image:: Forward_modeling_files/Forward_modeling_35_0.png


radiative transfering…

.. code:: ipython3

    from exojax.rt import planck
    from exojax.rt.rtransfer import rtrun
    sourcef = planck.piBarr(Tarr,nus)
    F0=rtrun(dtau,sourcef)

.. code:: ipython3

    plt.plot(wav[::-1],F0)




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f2baa2c2970>]




.. image:: Forward_modeling_files/Forward_modeling_38_1.png


applying an instrumental response and planet/stellar rotation to the raw
spectrum

.. code:: ipython3

    from exojax.spec import response
    from exojax.utils.constants import c
    import jax.numpy as jnp
    
    wavd=jnp.linspace(22920,23000,500) #observational wavelength grid
    nusd = 1.e8/wavd[::-1]
    
    RV=10.0 #RV km/s
    vsini=20.0 #Vsini km/s
    u1=0.0 #limb darkening u1
    u2=0.0 #limb darkening u2
    
    R=100000.
    beta=c/(2.0*np.sqrt(2.0*np.log(2.0))*R) #IP sigma need check 
    
    Frot=response.rigidrot(nus,F0,vsini,u1,u2)
    F=response.ipgauss_sampling(nusd,nus,Frot,beta,RV)

.. code:: ipython3

    plt.plot(wav[::-1],F0)
    plt.plot(wavd[::-1],F)




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f2baa4ff190>]




.. image:: Forward_modeling_files/Forward_modeling_41_1.png


The flux decreases at the edges of the left and right sides are
artificial due to the convolution. You might need to some margins of the
wavenumber range to eliminate these artifacts.

.. code:: ipython3

    np.savetxt("spectrum.txt",np.array([wavd,F]).T,delimiter=",")

