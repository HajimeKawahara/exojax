ExoMol, HITEMP, HITRAN common API
--------------------------------------

*November 4th (2022) Hajime Kawahara*

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


Using DataFrames
===========================================

ExoJAX mdb class inherits DataFrame of the common API, in "df" instance as. 
This DataFrame is not masked by "nurange" and/or "crit" options and has the format of Vaex lazy I/O.

.. code:: python

	>>> mdb = MdbExomol(".database/CO/12C-16O/Li2015", nurange=[4200.0, 4300.0])
	>>> mdb.df
	#        i_upper    i_lower    A          nu_lines      gup    jlower    jupper    elower      Sij0
	0        84         42         1.155e-06  2.405586      3      0         1         66960.7124  3.811968898414225e-164
	1        83         41         1.161e-06  2.441775      3      0         1         65819.903   9.663028103692631e-162
	2        82         40         1.162e-06  2.477774      3      0         1         64654.9206  2.7438392479197905e-159
	3        81         39         1.159e-06  2.513606      3      0         1         63465.8042  8.73322833971394e-157
	4        80         38         1.152e-06  2.549292      3      0         1         62252.5793  3.115220404216648e-154
	...      ...        ...        ...        ...           ...    ...       ...       ...         ...
	125,491  306        253        7.164e-10  22147.135424  15     6         7         80.7354     1.8282485593637477e-31
	125,492  474        421        9.852e-10  22147.86595   23     10        11        211.4041    2.0425455665383687e-31
	125,493  348        295        7.72e-10   22147.897299  17     7         8         107.6424    1.9589545250222689e-31
	125,494  432        379        9.056e-10  22148.262711  21     9         10        172.978     2.0662209116961706e-31
	125,495  390        337        8.348e-10  22148.273111  19     8         9         138.3903    2.0387827253771594e-31

For instance, if you want to call "i_upper", use "values" like:

.. code:: python

	>>> i_upper = mdb.df.i_upper.values
	>>> i_upper
	array([ 84,  83,  82, ..., 348, 432, 390])


Notice the above array is not masked. So, the length is different from for instance "mdb.nu_lines".

.. code:: python

	>>> len(i_upper)
	125496
	>>> len(mdb.nu_lines)
	771





