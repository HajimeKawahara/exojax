Mie Scattering
========================

``OpaMie`` computes cloud optical properties using
`miepython <https://miepython.readthedocs.io/>`__ for homogeneous spheres.
You can calculate them directly or interpolate a precomputed grid (``miegrid``).
ExoJAX integrates the single-particle results over a lognormal size distribution.

Direct calculation
------------------------

The direct calculation method is conducted as follows.
Please note that the initialization of OpaMie requires a particulate database (``pdb``).

:doc:`pdb`


.. code:: ipython3
    
    from exojax.opacity import OpaMie
    opa = OpaMie(pdb_nh3, nus)
    sigma_extinction, sigma_scattering, asymmetric_factor = opa.mieparams_vector_direct(rg, sigmag)

Here ``rg`` is the geometric mean radius in cm and ``sigmag > 1`` is the
geometric standard deviation. The cross sections are in cm² per particle;
the asymmetry factor is dimensionless. For layer-dependent parameters, use
``opa.mieparams_matrix_direct(rg_layer, sigmag_layer)``.

The refractive index convention in ExoJAX remains ``m = n + ik`` for
absorbing particles. ExoJAX converts it internally to miepython's ``n - ik``
convention. PyMieScatt is a legacy backend and is no longer supported.
The old APIs have been removed; update existing code as follows:

* ``mie_lognormal_pymiescatt`` → ``mie_lognormal``
* ``mieparams_vector_direct_from_pymiescatt`` → ``mieparams_vector_direct``
* ``mieparams_matrix_direct_from_pymiescatt`` → ``mieparams_matrix_direct``

For specific examples, please refer to 
:doc:`../tutorials/Ackerman_and_Marley_cloud_model`
for example.

.. warning::
    
    The direct calculation is not differentiable with JAX.
    If you need these values to be differentiable, you must create a miegrid and interpolate the opacity and asymmetric factor from the miegrid as shown below.


Generate a custom miegrid
------------------------------------------------------

You can create a miegrid as shown in the code below.

.. code:: ipython3
    
    from exojax.database.pardb import PdbCloud

    pdb = PdbCloud("NH3")
    pdb.generate_miegrid(
        sigmagmin=1.01,
        sigmagmax=4.0,
        Nsigmag=10,
        log_rg_min=-7.0,  # log10 of the radius in cm
        log_rg_max=-3.0,
        Nrg=40,
    )
    pdb.load_miegrid()
    opa = OpaMie(pdb, nus)

Grid generation uses miepython and runs outside JAX. The existing ``.npz``
format, including the seven stored Mie parameters, is unchanged, so previously
generated grids can still be loaded. Grid interpolation does not call the solver.
Existing grids retain their original values; regenerate them to use the new solver.

For large grids, enabling miepython's Numba acceleration is strongly recommended.
Set ``MIEPYTHON_USE_JIT=1`` before the first import of miepython, for example by
starting Python or Jupyter with the environment variable set:

.. code:: console

    MIEPYTHON_USE_JIT=1 python generate_miegrid.py
    MIEPYTHON_USE_JIT=1 jupyter lab

Here ``generate_miegrid.py`` is a script containing your grid-generation code.
Numba acceleration speeds up the solver; JAX differentiability still comes
from grid interpolation.


Gets the cloud opacity and asymmetric factor from the miegrid
-----------------------------------------------------------------

Once the miegrid is created, you can interpolate to obtain the opacity and asymmetric factor from the mie parameters, using ``opa.mieparams_vector``.

.. code:: ipython3
    
    sigma_extinction, sigma_scattering, asymmetric_factor = opa.mieparams_vector(rg, sigmag)


The opacity obtained in this way is differentiable. You can use ``rg`` and ``sigmag`` as parameters for gradient-based optimization or HMC-NUTS.
