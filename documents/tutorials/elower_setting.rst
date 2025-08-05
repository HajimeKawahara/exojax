Optimal Elower Maximum
^^^^^^^^^^^^^^^^^^^^^^^^^

PreMODIT make a grid of Elower. When the maximum value of Elower in the
database is large, the use of the device memory might become too large for
your GPU device memory and require additional computation time . 
We can use the lines whose Elower are below
user-defined value, by using elower_max option in mdb. But, before that,
we need to know the optimal value of the max Elower. This tutorial
explains how to do that. First use FP64 as usual.

.. code:: ipython3

    from jax import config
    config.update("jax_enable_x64", True)

Make “mdb”.

.. code:: ipython3

    from exojax.utils.grids import wavenumber_grid
    from exojax.database.exomol.api import MdbExomol
    
    nu_grid, wav, resolution = wavenumber_grid(2200.,
                                               2300.,
                                               10000,
                                               unit="cm-1",
                                               xsmode="premodit")
    mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid)



.. parsed-literal::

    xsmode =  premodit
    xsmode assumes ESLOG in wavenumber space: mode=premodit
    HITRAN exact name= (12C)(16O)
    Background atmosphere:  H2


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/grids.py:126: UserWarning: Resolution may be too small. R=224940.4617885842
      warnings.warn('Resolution may be too small. R=' + str(resolution),


.. parsed-literal::

    Reading .database/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2
    .broad is used.
    Broadening code level= a0
    default broadening parameters are used for  71  J lower states in  152  states


Because the device memory use (and computational cost) is proportional
to the maximum of Elower, we should check what the maximum value of
Elower is.

.. code:: ipython3

    import numpy as np
    print(np.min(mdb.elower),"-",np.max(mdb.elower),"cm-1")


.. parsed-literal::

    522.4751 - 84862.9693 cm-1


We assume we will use < 700K. Then, ~85000 cm-1 is enough high.
spec.optgrid.optelower can recommend the optimal value of Elower that
does not change the cross section within 1 %.

.. code:: ipython3

    from exojax.opacity.premodit.optgrid import optelower
    
    Tmax = 700.0 #K
    Pmin = 1.e-8 #bar
    
    Eopt = optelower(mdb, nu_grid, Tmax, Pmin)
    print("optimal elower_max=",Eopt)



.. parsed-literal::

    Maximum Elower =  84862.9693
    OpaPremodit: init w/o params setting
    Call self.apply_params() to complete the setting.
    OpaPremodit: params manually set.
    Tref changed: 296.0K->296.0K


.. parsed-literal::

    uniqidx: 0it [00:00, ?it/s]


.. parsed-literal::

    Premodit: Twt= 700.0 K Tref= 296.0 K


.. parsed-literal::

    opt Emax:  84%|████████▍ | 711/845 [03:56<00:44,  3.01it/s]

.. parsed-literal::

    optimal elower_max= 13922.4751


.. parsed-literal::

    


The optimal value of the maximum Elower is just 13923 cm-1. We can use
elower_max option to set the user-defined Elower max value.

.. code:: ipython3

    mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid, elower_max=13923.)



.. parsed-literal::

    HITRAN exact name= (12C)(16O)
    Background atmosphere:  H2
    Reading .database/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2
    .broad is used.
    Broadening code level= a0


.. code:: ipython3

    print(np.min(mdb.elower),"-",np.max(mdb.elower),"cm-1")


.. parsed-literal::

    522.4751 - 13791.2151 cm-1

