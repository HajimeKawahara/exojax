ExoMol, HITEMP, HITRAN common API
--------------------------------------

*August 24th (2022) Hajime Kawahara*

In ExoJAX 2, the standard molecular database IO for ExoMol, HITEMP, and HITRAN was shared with the radis team.


ExoMol
==========

How to load ExoMol CO database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython
	  
	  >>> MdbExomol(".database/CO/12C-16O/Li2015", nurange=[4200.0, 4300.0])



HITEMP
======================

How to load HITEMP CO database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython
	  
	  >>> MdbHitemp(".database/CO/", nurange=[4200.0, 4300.0])
	  >>> MdbHitemp(".database/05/", nurange=[4200.0, 4300.0])
	  >>> MdbHitemp(".database/CO/05_HITEMP2019/", nurange=[4200.0, 4300.0])

The style used in ExoJAX 1 is also acceptable (not recommended): 

.. code:: ipython
	  
	  >>> MdbHitemp(".database/CO/05_HITEMP2019/05_HITEMP2019.par.bz2", nurange=[4200.0, 4300.0])


HITRAN
======================

How to load HITRAN CO database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython
	  
	  >>> Mdbhitran(".database/CO/", nurange=[4200.0, 4300.0])
	  >>> Mdbhitran(".database/05/", nurange=[4200.0, 4300.0])
	  
The style used in ExoJAX 1 is also acceptable (not recommended): 

.. code:: ipython
	  
	  >>> Mdbhitran(".database/CO/05_hit12.par", nurange=[4200.0, 4300.0])
