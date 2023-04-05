Device Memory Control
========================

First, read the following webpage on JAX gpu memory allocation:

https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html


If you do not want the preallocation of the device memory and 
you would like to allocate the device memory exactly what is needed on demand, 
set like this.

.. code:: ipython3

    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'




.. code:: ipython3

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


You can check the device memory use by inserting the following anywhere you like.

.. code:: ipython3

    from cuda import cudart
    cudart.cudaMemGetInfo()