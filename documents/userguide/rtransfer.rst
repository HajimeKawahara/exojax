Radiative Transfer
======================



Atmospheric Layer Model
---------------------------

:math:`F_{n} = \mathcal{T}_n F_{n+1} + (1-\mathcal{T}_n) \, \mathcal{S}_n`


:math:`F_0 = \mathcal{T}_0 ( \mathcal{T}_1 ( \mathcal{T}_2 (\cdots\mathcal{T}_{N-2} (\mathcal{T}_{N-1} F_B + \mathcal{Q}_{N-1} ) + \mathcal{Q}_{N-2}) + \cdots + \mathcal{Q}_2) + \mathcal{Q}_1) + \mathcal{Q}_0`


Transmission for Pure Absorption: trans2E3
-------------------------------------------

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
