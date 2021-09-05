FP32 and FP64
=============================

ExoJAX uses mixture of FP32 and FP64, depending on functions.



Resolution of wavenumber 
--------------------------------


JAX with FP64
-------------------

The default precision of JAX is FP32 although one can use JAX in FP64 but without XLA.
One can switch to FP64 by using config.update in jax:

.. code:: ipython
       
       >>> from jax.config import config                                                  
       >>> config.update("jax_enable_x64", True)
	  

