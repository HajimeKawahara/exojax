CIA Absorption Coefficient
===============================

Use `contdb.CdbCIA <../exojax/exojax.spec.html#exojax.spec.contdb.CdbCIA>`_ for loading the CIA database.

.. code:: ipython3

    from exojax.spec.rtransfer import nugrid
    nus,wav,res=nugrid(5000,50000,1000,unit="AA")
    from exojax.spec import contdb
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)


.. parsed-literal::

    WARNING: resolution may be too small. R= 1000.0
    H2-H2

`hitrancia.logacia <../exojax/exojax.spec.html#exojax.spec.hitrancia.logacia>`_ can provide a log10 of absorption coeffcient as a function of
temperature

.. code:: ipython3

    from exojax.spec.hitrancia import logacia
    import jax.numpy as jnp
    Tfix=jnp.array([1000.0,1300.0,1600.0])
    lc=logacia(Tfix,nus,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)

Plotting...

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn
    plt.style.use('bmh')
    for i in range(0,len(Tfix)):
        plt.plot(wav[::-1],lc[i,:],lw=1,label=str(int(Tfix[i]))+" K")
    plt.axvspan(22876.0,23010.0,alpha=0.3,color="green")
    plt.xlabel("wavelength ($\\AA$)")
    plt.ylabel("absorption coefficient ($cm^5$)")
    plt.legend()
    plt.savefig("cia.png")

.. image:: CIA_opacity/output_5_0.png


.. code:: ipython3

    #max value
    import numpy as np
    1.e8/nus[np.argmax(lc[1,:])]




.. parsed-literal::

    23858.80474469375


