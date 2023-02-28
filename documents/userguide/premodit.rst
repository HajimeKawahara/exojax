PreMODIT
=================

PreMODIT is the successor algorithm to MODIT. 
The problem with :doc:`modit`` is that the lineshape density (LSD) must be recalculated 
from all transition information each time the temperature or pressure conditions are changed. 
This means that all transition information must be stored in the device memory, 
which is not memory efficient.

PreMODIT is an algorithm that solves the above problem.
Details of the algorithm will be described in a forthcoming paper (in preparation).
But, the basic idea is to densify the line information before storing it in device memory.
While this saves device memory, the disadvantage is that the temperature range for which accuracy 
can be guaranteed must be set in advance. So, we need "auto_trange" option in `OpaPremodit <../exojax/exojax.spec.html#exojax.spec.opacalc.OpaPremodit>`_.

.. code:: ipython
	
    >>> from exojax.spec.opacalc import OpaPremodit
    >>> diffmode = 0
    >>> opa = OpaPremodit(mdb=mdbCO,
                      nu_grid=nus,
                      diffmode=diffmode,
                      auto_trange=[400.0, 1500.0])

This means that a 1 % accuracy is guaranteed between 400 - 1500 K. 
If you are more familiar with the algorithm of PreMODIT, you can directly specify the parameters using "manual_params" option.

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
