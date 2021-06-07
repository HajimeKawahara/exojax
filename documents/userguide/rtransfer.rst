Radiative Transfer
======================

Exojax uses a layer-based atmospheric model for `radiative transfer <https://en.wikipedia.org/wiki/Radiative_transfer>`_ (RT). Currently, the only supported RT is the emission model with no scattering.

Atmospheric Layer Model
---------------------------

The upward flux of the n-th layer (with pressure of :math:`P_n`) is connected to that of the (n-1)-th layer with transmission T and source function S. 

:math:`F_{n} = \mathcal{T}_n F_{n+1} + (1-\mathcal{T}_n) \, \mathcal{S}_n`

where :math:`P_{n-1} < P_n`. So, we need to specify a transmission and source function. 

Source Function
---------------------------

In the case that a black body emission as a source as,  

:math:`\mathcal{S} = \pi B(T)`

we can use `piBarr <../exojax/exojax.spec.html#exojax.spec.planck.piBarr>`_.


.. code:: ipython

	  >>> from exojax.spec import planck	  
	  >>> sourcef = planck.piBarr(Tarr,nus)

Transmission for Pure Absorption: trans2E3
-------------------------------------------

Currently, exojax supports only a transmission for pure absorption. In this case, the transmission is given as

:math:`\mathcal{T}_n = 2 E_3(\Delta \tau_n ) = ( 1 - \Delta \tau_n) \exp{(- \Delta \tau_n)} + (\Delta \tau_n )^2 E_1(\Delta \tau_n )`

where :math:`\Delta \tau_n` is delta opacity in the n-th layer, :math:`E_j(x)` is the exopential integral of the :math:`j` -th order. In exojax, :math:`2 E_3(x)` is available as

.. code:: ipython
	  
	  >>> from exojax.spec.rtransfer import trans2E3
	  >>> trans2E3(1.0)
	  DeviceArray(0.21938396, dtype=float32)

`trans2E3 <../exojax/exojax.spec.html#exojax.spec.rtransfer.trans2E3>`_ is auto-differentiable.
	  
.. code:: ipython
	  	  
	  >>> from jax import grad
	  >>> grad(trans2E3)(1.0)
	  DeviceArray(-0.29698896, dtype=float32)


`trans2E3 <../exojax/exojax.spec.html#exojax.spec.rtransfer.trans2E3>`_ is used in `rtrun <../exojax/exojax.spec.html#exojax.spec.rtransfer.rtrun>`_, which gives an emission spectral model. Then, `rtrun <../exojax/exojax.spec.html#exojax.spec.rtransfer.rtrun>`_ has two inputs, one is the arrays of :math:`\Delta \tau_n` and source funtion.

.. code:: python
	  
	 F0=rtrun(dtau,sourcef) 

See ":doc:`../tutorials/forward_modeling`" to know how to use `rtrun <../exojax/exojax.spec.html#exojax.spec.rtransfer.rtrun>`_ in a forward modeling. Note that exojax uses a linear algebraic formulation to solve the RT. The detail description is provided in Kawahara et al.
