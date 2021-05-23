Radiative Transfer
======================

Atmospheric Layer Model
---------------------------

The upward flux of the n-th layer (with pressure of :math:`P_n`) isconnected to that of the (n-1)-th layer with transmission T and source function S. 

:math:`F_{n} = \mathcal{T}_n F_{n+1} + (1-\mathcal{T}_n) \, \mathcal{S}_n`

where :math:`P_{n-1} < P_n`. So, we needa transmission and source function. 

Transmission for Pure Absorption: trans2E3
-------------------------------------------

Currently, exojax supports only a transmission for pure absorption. In this case, the transmission is given as


:math:`\mathcal{T}_n = 2 E_3(\Delta \tau_n ) = ( 1 - \Delta \tau_n) \exp{(- \Delta \tau_n)} + (\Delta \tau_n )^2 E_1(\Delta \tau_n )`

where :math:`E_j(x)` is the exopential integral of the :math:`j` -th order. In exojax, :math:`2 E_3(x)` is available as

.. code:: ipython
	  
	  >>> from exojax.spec.rtransfer import trans2E3
	  >>> trans2E3(1.0)
	  DeviceArray(0.21938396, dtype=float32)

`trans2E3 <../exojax/exojax.spec.html#exojax.spec.rtransfer.trans2E3>`_ is auto-differentiable.
	  
.. code:: ipython
	  	  
	  >>> from jax import grad
	  >>> grad(trans2E3)(1.0)
	  DeviceArray(-0.29698896, dtype=float32)

Source Function
---------------------------

In the case that a black body emission as a source as,  

:math:`\mathcal{S} = \pi B(T)`

we can use `piBarr <../exojax/exojax.spec.html#exojax.spec.planck.piBarr>`_.


.. code:: ipython

	  >>> from exojax.spec import planck	  
	  >>> sourcef = planck.piBarr(Tarr,nus)
