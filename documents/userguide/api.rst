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

If you have the error like,

.. code:: sh

	Please fix/delete the radis.json entry, change the `databank_name`, or change the default local databases path entry 'DEFAULT_DOWNLOAD_PATH' in `radis.config` or ~/radis.json

remove radis.json and retry it.

Isotope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HITEMP includes all of the isotopes.  To know which isotopes are included in mdb, use uniqiso instance.

.. code:: ipython
	  
	  >>> mdb = MdbHitemp(".database/CO/", nurange=[4200.0, 4210.0], crit=1.e-30)
	  >>> mdb.uniqiso #-> [1,2,3,4,6]

You can know what isotope name "isotope=1" corresponds to

.. code:: ipython
	  
	  >>> mdb.exact_isotope_name(1) #-> (12C)(16O)

Loading HITEMP for Each Isotope
--------------------------------------

Sometimes it's useful to take it out for each isotope.
To load C12 O16 (isotope = 1), use the isotope option. 
"isotope" is the isotope number used in HITRAN/HITEMP, which starts from 1.

.. code:: ipython
	  
	  >>> mdb = MdbHitemp(".database/CO/", nurange=[4200.0, 4300.0], isotope = 1)

Parition Function (Ratio) for Each Isotope
-----------------------------------------

In MdbHitemp, QT_interp and qr_interp has an isotope option. 

.. code:: ipython
	  
	  >>> T = 1000 #K
	  >>> isotope = 1
	  >>> QT = mdb.QT_interp(isotope, T) # partition function Q(T) for isotope=1
	  >>> q_ratio = mdb.qr_interp(isotope, T) # partition function ratio Q(T)/Q(Tref)

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
