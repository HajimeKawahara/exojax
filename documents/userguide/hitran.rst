HITRAN/HITEMP
--------------

Molecular Database
======================

An example to use
`HITRAN <https://hitran.org/>`_
/
`HITEMP <https://hitran.org/hitemp/>`_
from exojax is like that.

.. code:: ipython
	  
	  >>> from exojax.spec.rtransfer import nugrid
	  >>> from exojax.spec import moldb
	  >>> nus,wav,res=nugrid(22880.,23000.,1000,unit="AA")
	  >>> mdbCO=moldb.MdbHit('.database/05_HITEMP2019.par.bz2',nus)
	  bunziping
	  

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
|log10 Sij0             |logsij0      |    |jnp   |
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

	  >>> from exojax.spec import contdb	  
	  >>> cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)


