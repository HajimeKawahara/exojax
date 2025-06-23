On the Device Memory Use
============================

`Last update: May 20th (2023) Hajime Kawahara`


Frequent device memory overflows (memory on GPU) occur when modeling a wide wavelength range with high wavelength resolution. 
In this section, we discuss how to reduce memory in the following ways.


Estimating Device Memory Use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For PreMODIT, `exojax.utils.memuse.device_memory_use` can estimate an approximate requirement of the device memory. 

.. code:: ipython

    >>> from exojax.utils.memuse import device_memory_use
    >>> from exojax.test.emulate_mdb import mock_wavenumber_grid
    >>> from exojax.rt import ArtEmisPure
    >>> from exojax.opacity import OpaPremodit
    >>> from exojax.test.emulate_mdb import mock_mdbExomol
    >>> nu_grid, wav, res = mock_wavenumber_grid()
    >>> art = ArtEmisPure(nu_grid,
                      pressure_top=1.e-8,
                      pressure_btm=1.e2,
                      nlayer=100)
    >>> art.change_temperature_range(400.0, 1500.0)

    >>> mdb = mock_mdbExomol()
    >>> opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      auto_trange=[art.Tlow, art.Thigh],
                      broadening_resolution={
                          "mode": "manual",
                          "value": 0.2
                      })
    >>> nfree = 10 # the number of the free parameters for HMC, optimization etc.
    >>> memuse = device_memory_use(opa, art=art, nfree=nfree)
    



Reducing Device Memory for PreMODIT 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


For Premodit, 

We recommend to try the following points

- Consider to decrease the resolution of the broadening parameters, using `broadening_resolution` option in OpaPremodit. See :doc:`premodit` for the details of `broadening_resolution`.
- Divide the wavenumber range into multiple segments. Note that the calculation time for the forward spectrum modeling part increases almost linearly with the number of the segments. So we recommend dividing the wavenumber range with the least required number of the segments.



Device Memory Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

