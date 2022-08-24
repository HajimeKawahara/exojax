ExoMol, HITEMP, HITRAN common API
--------------------------------------

*August 24th (2022) Hajime Kawahara*

In ExoJAX 2, the standard molecular database IO for ExoMol, HITEMP, and HITRAN was shared with the radis team.


HITEMP
======================

How to load HITEMP CO database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython
	  
	  >>> MdbHitemp(".database/CO/", nurange=[4200.0, 4300.0])
	  >>> MdbHitemp(".database/05/", nurange=[4200.0, 4300.0])
	  >>> MdbHitemp(".database/CO/05_HITEMP2019/", nurange=[4200.0, 4300.0])

The style used in ExoJAX 1 is also acceptable: 

.. code:: ipython
	  
	  >>> MdbHitemp(".database/CO/05_HITEMP2019/05_HITEMP2019.par.bz2", nurange=[4200.0, 4300.0])
