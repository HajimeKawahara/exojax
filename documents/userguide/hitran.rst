HITRAN/HITEMP
--------------

Line Database
===============

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
