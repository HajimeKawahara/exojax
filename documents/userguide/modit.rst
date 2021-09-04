Modified Discrete Integral Transform (MODIT)
==============================================

MODIT is a modified version of DIT. 

How is the LSD computed?
---------------------------

DIT needs to compute the lineshape density (LSD) matrix. A linear contribution is computed using `getix <../exojax/exojax.spec.html#exojax.spec.dit.getix>`_.

.. code:: ipython
       
       >>> from exojax.spec.dit import getix
       >>> import jax.numpy as jnp
       >>> y=jnp.array([1.1,4.3])
       >>> yv=jnp.arange(6)
       >>> getix(y,yv)
       (DeviceArray([0.10000002, 0.3000002 ], dtype=float32), DeviceArray([1, 4], dtype=int32))    

For wavenumber, the F64 precision is required. So, `npgetix <../exojax/exojax.spec.html#exojax.spec.dit.npgetix>`_ is used for precomputation, which is numpy version of getix. 
