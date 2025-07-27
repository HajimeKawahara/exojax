HITRAN/HITEMP
--------------

*May (2021) Hajime Kawahara*

See ":doc:`../tutorials/opacity`" for tutorial to compute a cross section profile using HITRAN/HITEMP.


Molecular Database
======================

An example to use
`HITRAN <https://hitran.org/>`_
/
`HITEMP <https://hitran.org/hitemp/>`_
from exojax is like that.

.. code:: ipython
	  
	  >>> from exojax.utils.grids import wavenumber_grid
	  >>> from exojax.database import moldb 
	  >>> nus,wav,res=nugrid(22880.,23000.,1000,unit="AA")
	  >>> mdbCO=moldb.MdbHit('.database/05_HITEMP2019.par.bz2',nus)
	  bunziping

HITEMP H2O and CO2
======================

For H2O and CO2, HITEMP provides multiple par files. To use those files, provide the directory path for ``moldb.MdbHit`` as follows.

.. code:: ipython
	  
	  >>> mdbH2O=moldb.MdbHit('.database/01_HITEMP2010',nus)
	  >>> mdbCO2=moldb.MdbHit('.database/02_HITEMP2010',nus)

extract option
======================
	  
``extract=True`` in ``moldb.MdbHit`` extracts the opacity data in the wavenumber range of ``nus`` with ``margin``. Theforefore it can reduce the use of DRAM. It may be useful for large databases such as CH4.

.. code:: ipython
	  
	  >>> mdbCH4=moldb.MdbHit('.database/06_HITEMP2020.par.bz2',nus,extract=True)

This creates a new directory (such as ``6101.281269066504_6108.7354917532075_1.0`` ) and the extracted data and header files (``06_HITEMP2020.header`` and  ``06_HITEMP2020.par`` ) in it:

.. code:: sh
	  
	  ls .database
	  
	  06_HITEMP2020.par
	  6101.281269066504_6108.7354917532075_1.0:
	  06_HITEMP2020.header  06_HITEMP2020.par

	  
Basic Quantities
==================

These are the basic quantities of MdbHit.

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
|air pressure broadening|gamma_air    |cm-1|jnp   |
+-----------------------+-------------+----+------+
|self broadning         |gamma_self   |cm-1|jnp   |
+-----------------------+-------------+----+------+
|Einstein coefficient   |A            |s-1 |jnp   |
+-----------------------+-------------+----+------+
|reference line strength|Sij0         |cm  |np    |
+-----------------------+-------------+----+------+
|log_e Sij0             |logsij0      |    |jnp   |
+-----------------------+-------------+----+------+
|statistical weight     |gpp          |    |jnp   |
+-----------------------+-------------+----+------+
|temperature exponent   |n_air        |    |jnp   |
+-----------------------+-------------+----+------+

Collision Induced Absorption (CIA)
==================================

`Collision Induced Absorption (CIA) <https://en.wikipedia.org/wiki/Collision-induced_absorption_and_emission>`_
is one of the important continuum in exoplanet/brown dwarfs atmosphere.
HITRAN provides
`the CIA files <https://hitran.org/cia/>`_
. To load the CIA file, use contdb.

.. code:: ipython

	  >>> from exojax.database import contdb 	  
	  >>> cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)

See ":doc:`../tutorials/CIA_opacity`" for tutorial.
