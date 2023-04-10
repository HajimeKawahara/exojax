PreMODIT
=================

PreMODIT is the successor algorithm to MODIT. 
The problem with :doc:`modit`` is that the lineshape density (LSD) has to be recalculated 
from all transition information each time the temperature or pressure conditions are changed. 
This means that all transition information must be stored in device memory, 
which is not memory efficient.

PreMODIT is an algorithm that solves the above problem.
Details of the algorithm will be described in a forthcoming paper (in preparation).
But the basic idea is to compress the line information before storing it in device memory.
While this saves device memory, the disadvantage is that the temperature range over which accuracy can be 
can be guaranteed must be set in advance. So we need the "auto_trange" option in `OpaPremodit <../exojax/exojax.spec.html#exojax.spec.opacalc.OpaPremodit>`_.

.. code:: ipython
	
    >>> from exojax.spec.opacalc import OpaPremodit
    >>> diffmode = 0
    >>> opa = OpaPremodit(mdb=mdbCO,
                      nu_grid=nus,
                      diffmode=diffmode,
                      auto_trange=[400.0, 1500.0])

This means that 1% accuracy is guaranteed between 400 - 1500 K. 
If you are more familiar with PreMODIT's algorithm, you can specify the parameters directly using the "manual_params" option.

.. code:: ipython
	
    >>> from exojax.spec.opacalc import OpaPremodit
    >>> diffmode = 0
    >>> dE = 300.0 # cm-1
    >>> Tref = 400.0 # in Kelvin
    >>> Twt = 1000.0 # in Kelvin
    >>> opa = OpaPremodit(mdb=mdbCO,
                      nu_grid=nus,
                      diffmode=diffmode,
                      manual_params=[dE, Tref, Twt])

Single Broadening Parameter Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, this algorithm constructs one grid for the broadening parameter. 
However, reducing the number of broadening grids may be useful for fitting, 
since the device memory usage becomes 
broadening grid number x free parameter number x atmospheric layer number x wavenumber grid number x F64/F32 byte number. 
By setting "single_broadening" option to True, PreMODIT can be used with a single broadening parameter.


.. code:: ipython
	
    >>> opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[500.0, 1500.0],
                      single_broadening=True)
    

In the above case, we assumed the median of broadening parameters of mdb. If you want to give the specific values use "single_broadening_parameters" option.