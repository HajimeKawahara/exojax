Molecular Database and Conversion
=====================================

*Update: May 27/2021, Hajime Kawahara*

exojax uses a specific mdb (molecular database) class for each molecular/atom database, `moldb.MdbExomol <../exojax/exojax.spec.html#exojax.spec.moldb.MdbExomol>`_ is for ExoMol, `moldb.MdbExoHit <../exojax/exojax.spec.html#exojax.spec.moldb.MdbHit>`_ is for HITRAN/HITEMP. Here, we use MdbExomol for explanation because most instances are common in mdb classes. 

Click below for the details for mdb class for each database.

- :doc:`exomol`
- :doc:`hitran`


.. code:: ipython
	  
	  >>> from exojax.spec import moldb
	  >>> from exojax.spec.rtransfer import nugrid
	  >>> nus,wav,res=nugrid(22880.,23000.,1000,unit="AA")
	  >>> mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus)
	  default broadening parameters are used for  71  J lower states in  152  states

Basic Quantities
----------------

In mdb, the line centers in the unit of cm-1, for instance, are stored as

.. code:: ipython
	  
	  >>> mdbCO.nu_lines

Examples of the other quantities in mdb are listed in the table below. 

+-----------------------+-------------+----+------+
|**quantity**           |**instance** |unit|np/jnp|
+-----------------------+-------------+----+------+
|line center            |nu_lines     |cm-1|np    |
+-----------------------+-------------+----+------+
|line center            |dev_nu_lines |cm-1|jnp   |
+-----------------------+-------------+----+------+
|lower state energy     |elower       |cm-1|jnp   |
+-----------------------+-------------+----+------+
|natural broadening     |gamma_natural|cm-1|jnp   |
+-----------------------+-------------+----+------+
|Einstein coefficient   |A            |s-1 |jnp   |
+-----------------------+-------------+----+------+
|reference line strength|Sij0         |cm  |np    |
+-----------------------+-------------+----+------+
|log_e Sij0             |logsij0      |    |jnp   |
+-----------------------+-------------+----+------+

The fourth column indicates the data type of the array; np means the array is numpy nd array (float64). jnp is jax.numpy array on device. Remember that the jnp array is float32. Why do we have both np and jnp versions of the line center? This is because the float64 is needed to keep precision for some conversions. See the last section of "  :doc:`../tutorials/opacity` " for more details. Because the lien strength is small number in general, we have np version of S0 and jnp version of **logarithm** of S0. In fact, logsij0 is used to compute the line strength.

Line Strength
------------------

The line strength is a function of temperature, expressed as

:math:`S (T) = S_0 \frac{Q(T_\mathrm{ref})}{Q(T)} \frac{e^{- h c E_\mathrm{low} /k_B T}}{e^{- h c E_\mathrm{low}  /k_B T_\mathrm{ref}}} \frac{1- e^{- h c \hat{\nu} /k_B T}}{1-e^{- h c \hat{\nu} /k_B T_\mathrm{ref}}}`

The reference line strength `S_0` is the line strength at :math:`T_\mathrm{ref}=296` K (Sij0). exojax can compute S(T) in jax based. As I said, s0=logsij0 is used to compute S(T) as

:math:`S (T) = q_t^{-1} e^{  s_0 - c_2 E_\mathrm{low}  (T^{-1} - T_\mathrm{ref}^{-1}) }  \frac{1- e^{- c_2 \hat{\nu}/ T}}{1-e^{- c_2 \hat{\nu}/T_\mathrm{ref}}}`

where :math:`c_2 = h c/k_B`. In exojax, S(T) is computed using the normalized partition function :math:`q_t=Q(T)/Q(T_\mathrm{ref})` as


.. code:: ipython
	  
	  >>> from exojax.spec import SijT
	  >>> qt=mdbCO.qr_interp(Tfix)
	  >>> Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)

