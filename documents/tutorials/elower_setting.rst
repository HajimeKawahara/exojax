Reducing Device Memory Use by Setting the Optimal Elower Maximum
================================================================

PreMODIT make a grid of Elower. When the maximum value of Elower in the
database is large, the use of the device memory becomes too large for
your GPU CUDA memory. We can use the lines whose Elower are below
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

    2024-10-02 10:20:59.968678: W external/xla/xla/service/gpu/nvptx_compiler.cc:765] The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.6.20). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.


.. parsed-literal::

    xsmode =  premodit
    xsmode assumes ESLOG in wavenumber space: xsmode=premodit
    ======================================================================
    The wavenumber grid should be in ascending order.
    The users can specify the order of the wavelength grid by themselves.
    Your wavelength grid is in ***  descending  *** order
    ======================================================================
    HITRAN exact name= (12C)(16O)
    radis engine =  vaex
    		 => Downloading from http://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.def


.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/grids.py:144: UserWarning: Resolution may be too small. R=224940.4617885842
      warnings.warn("Resolution may be too small. R=" + str(resolution), UserWarning)
    /home/kawahara/exojax/src/exojax/utils/molname.py:197: FutureWarning: e2s will be replaced to exact_molname_exomol_to_simple_molname.
      warnings.warn(
    /home/kawahara/exojax/src/exojax/utils/molname.py:91: FutureWarning: exojax.utils.molname.exact_molname_exomol_to_simple_molname will be replaced to radis.api.exomolapi.exact_molname_exomol_to_simple_molname.
      warnings.warn(
    /home/kawahara/exojax/src/exojax/utils/molname.py:91: FutureWarning: exojax.utils.molname.exact_molname_exomol_to_simple_molname will be replaced to radis.api.exomolapi.exact_molname_exomol_to_simple_molname.
      warnings.warn(


.. parsed-literal::

    		 => Downloading from http://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.pf
    		 => Downloading from http://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.states.bz2
    		 => Downloading from http://www.exomol.com/db/CO/12C-16O/12C-16O__H2.broad
    		 => Downloading from http://www.exomol.com/db/CO/12C-16O/12C-16O__He.broad
    		 => Downloading from http://www.exomol.com/db/CO/12C-16O/12C-16O__air.broad
    Note: Caching states data to the vaex format. After the second time, it will become much faster.
    Molecule:  CO
    Isotopologue:  12C-16O
    Background atmosphere:  H2
    ExoMol database:  None
    Local folder:  .database/CO/12C-16O/Li2015
    Transition files: 
    	 => File 12C-16O__Li2015.trans
    		 => Downloading from http://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.trans.bz2
    		 => Caching the *.trans.bz2 file to the vaex (*.h5) format. After the second time, it will become much faster.
    		 => You can deleted the 'trans.bz2' file by hand.
    Broadening code level: a0


.. parsed-literal::

    /home/kawahara/exojax/src/radis/radis/api/exomolapi.py:685: AccuracyWarning: The default broadening parameter (alpha = 0.07 cm^-1 and n = 0.5) are used for J'' > 80 up to J'' = 152
      warnings.warn(


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
    OpaPremodit: initialization without parameters setting
    Call self.apply_params() to complete the setting.
    OpaPremodit: params manually set.
    OpaPremodit: Tref_broadening is set to  455.19226706964173 K
    # of reference width grid :  2
    # of temperature exponent grid : 2


.. parsed-literal::

    uniqidx: 0it [00:00, ?it/s]

.. parsed-literal::

    Premodit: Twt= 700.0 K Tref= 296.0 K
    Making LSD: 0%Making LSD: 100%


.. parsed-literal::

    opt Emax:  84%|████████▍ | 711/845 [04:40<00:52,  2.53it/s]

.. parsed-literal::

    Found the optimal maximum elower: 13922.4751 cm-1
    optimal elower_max= 13922.4751


The optimal value of the maximum Elower is just 13923 cm-1. We can use
elower_max option to set the user-defined Elower max value.

.. code:: ipython3

    mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=nu_grid, elower_max=13923.)



.. parsed-literal::

    /home/kawahara/exojax/src/exojax/utils/molname.py:197: FutureWarning: e2s will be replaced to exact_molname_exomol_to_simple_molname.
      warnings.warn(
    /home/kawahara/exojax/src/exojax/utils/molname.py:91: FutureWarning: exojax.utils.molname.exact_molname_exomol_to_simple_molname will be replaced to radis.api.exomolapi.exact_molname_exomol_to_simple_molname.
      warnings.warn(
    /home/kawahara/exojax/src/exojax/utils/molname.py:91: FutureWarning: exojax.utils.molname.exact_molname_exomol_to_simple_molname will be replaced to radis.api.exomolapi.exact_molname_exomol_to_simple_molname.
      warnings.warn(


.. parsed-literal::

    HITRAN exact name= (12C)(16O)
    radis engine =  vaex
    Molecule:  CO
    Isotopologue:  12C-16O
    Background atmosphere:  H2
    ExoMol database:  None
    Local folder:  .database/CO/12C-16O/Li2015
    Transition files: 
    	 => File 12C-16O__Li2015.trans
    Broadening code level: a0


.. code:: ipython3

    print(np.min(mdb.elower),"-",np.max(mdb.elower),"cm-1")


.. parsed-literal::

    522.4751 - 13791.2151 cm-1

