Mie Scattering
========================

Last update: April 12th (2024) Hajime Kawahara

Currently, ``ExoJAX`` relies entirely on ``PyMieScatt`` (https://github.com/bsumlin/PyMieScatt) for Mie Scattering. The ``opa`` for Mie Scattering is ``OpaMie``. 
There are two methods for calculations: directly calling ``PyMieScatt`` and using pre-calculated grid models (miegrid). 

Direct calculation
------------------------

The direct calculation method is conducted as follows.
Please note that the initialization of OpaMie requires a particulate database (``pdb``).

:doc:`pdb`


.. code:: ipython3
    
    from exojax.opacity import OpaMie
    opa = OpaMie(pdb_nh3, nus)
    sigma_extinction, sigma_scattering, asymmetric_factor = opa.mieparams_vector_direct_from_pymiescatt(rg, sigmag)
    #sigma_extinction, sigma_scattering, asymmetric_factor = opa.mieparams_vector(rg,sigmag) # if using MieGrid

For specific examples, please refer to 
:doc:`../tutorials/Jupiter_Hires_Modeling`
for example.

.. warning::
    
    The cloud opacity and asymmetric factor calculated using this method are not differentiable because it directly calls ``PyMieScatt``. 
    If you need these values to be differentiable, you must create a miegrid and interpolate the opacity and asymmetric factor from the miegrid as shown below.


Generates custom miegrid (mgd)
------------------------------------------------------

You can create a miegrid as shown in the code below.

.. code:: ipython3
    
    from exojax.database.mie import compute_mie_coeff_lognormal_grid
    from exojax.database.pardb  import PdbCloud

    pdb = PdbCloud("NH3")
    filename = "miegrid_lognorm_"+pdb.condensate+".mgd"
    print(filename)
    
    Nsigmag = 10
    sigmag_arr = np.logspace(-1,1,Nsigmag)
    Nrg = 40
    rg_arr = np.logspace(-7,-3,Nsigmag) #cm
    
    miegrid = compute_mie_coeff_lognormal_grid(
        pdb.refraction_index,
        pdb.refraction_index_wavelength_nm,
        sigmag_arr,
        rg_arr,
        npart=1.0,
    )
    np.savez(filename, miegrid)


Gets the cloud opacity and asymmetric factor from the miegrid
-----------------------------------------------------------------

Once the miegrid is created, you can interpolate to obtain the opacity and asymmetric factor from the mie parameters, using ``opa.mieparams_vector``.

.. code:: ipython3
    
    sigma_extinction, sigma_scattering, asymmetric_factor = opa.mieparams_vector(rg, sigmag)


The opacity obtained in this way is differentiable. You can use ``rg`` and ``sigmag`` as parameters for gradient-based optimization or HMC-NUTS.
