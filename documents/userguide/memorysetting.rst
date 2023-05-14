
Reducing Device Memory for PreMODIT 
========================================


For Premodit, 

We recommend to try the following points

- Consider to decrease the resolution of the broadening parameters, using `broadening_resolution` option in OpaPremodit. See :doc:`premodit` for the details of `broadening_resolution`.
- Divide the wavenumber range into multiple segments


Device Memory Control
========================

First, read the following webpage on JAX gpu memory allocation:

https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html


If you do not want to pre-allocate device storage and 
and want to allocate the device memory exactly as needed, 
use this setting. But, this might cause 2-3 times slow down of the code.

.. code:: ipython3

    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

.. code:: ipython3

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

You can check the device memory usage by inserting the following at any point.

.. code:: ipython3

    from cuda import cudart
    cudart.cudaMemGetInfo()

