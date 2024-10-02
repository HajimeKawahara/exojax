CIA coefficient
===============

.. code:: ipython3

    from exojax.utils.grids import wavenumber_grid
    
    nus, wav, res = wavenumber_grid(5000, 50000, 1000, unit="AA", xsmode="lpf")
    from exojax.spec import contdb
    
    cdbH2H2 = contdb.CdbCIA(".database/H2-H2_2011.cia", nus)


.. parsed-literal::

    xsmode =  lpf
    xsmode assumes ESLOG in wavenumber space: xsmode=lpf
    ======================================================================
    The wavenumber grid should be in ascending order.
    The users can specify the order of the wavelength grid by themselves.
    Your wavelength grid is in ***  descending  *** order
    ======================================================================
    H2-H2


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/spec/unitconvert.py:63: UserWarning: Both input wavelength and output wavenumber are in ascending order.
      warnings.warn(
    /home/kawahara/exojax/src/exojax/utils/grids.py:144: UserWarning: Resolution may be too small. R=433.86018742134854
      warnings.warn("Resolution may be too small. R=" + str(resolution), UserWarning)


logacia can provide an absorption coeffcient as a function of
temperature

.. code:: ipython3

    from exojax.spec.hitrancia import interp_logacia_vector
    import jax.numpy as jnp
    
    Tfix = jnp.array([1000.0, 1300.0, 1600.0])
    lc = interp_logacia_vector(Tfix, nus, cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac)

plotting…

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    plt.style.use("bmh")
    for i in range(0, len(Tfix)):
        plt.plot(wav, lc[:, i], lw=1, label=str(int(Tfix[i])) + " K")
    plt.axvspan(22876.0, 23010.0, alpha=0.3, color="green")
    plt.xlabel("wavelength ($\\AA$)")
    plt.ylabel("absorption coefficient ($cm^5$)")
    plt.legend()
    plt.savefig("cia.png")



.. image:: CIA_opacity_files/CIA_opacity_5_0.png


.. code:: ipython3

    #max value
    import numpy as np
    1.e8 / nus[np.argmax(lc[1, :])]





.. parsed-literal::

    23858.80474469375


