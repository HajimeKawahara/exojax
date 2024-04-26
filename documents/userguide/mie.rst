Mie Scattering
========================

Last update: April 12th (2024) Hajime Kawahara

Currently, ``ExoJAX`` relies entirely on ``PyMieScatt`` (https://github.com/bsumlin/PyMieScatt) for Mie Scattering. The ``opa`` for Mie Scattering is ``OpaMie``. 
There are two methods for calculations: directly calling ``PyMieScatt`` and using pre-calculated grid models (Miegrid). 
However, as of version 1.5, the method using Miegrid is not yet fully developed. 

Direct calculation
------------------------

The direct calculation method is conducted as follows.
Please note that the initialization of OpaMie requires a particulate database (``pdb``).

:doc:`pdb`


.. code:: ipython3
    
    from exojax.spec.opacont import OpaMie
    opa = OpaMie(pdb_nh3, nus)
    sigma_extinction, sigma_scattering, asymmetric_factor = opa.mieparams_vector_direct_from_pymiescatt(rg, sigmag)
    #sigma_extinction, sigma_scattering, asymmetric_factor = opa.mieparams_vector(rg,sigmag) # if using MieGrid

For specific examples, please refer to 
:doc:`../tutorials/Jupiter_Hires_Modeling`
for example.


Generates custom Miegrid (mgd), under development
------------------------------------------------------

See mie.py, but a sample code is like the following

.. code:: ipython3
    
    from exojax.spec.mie import compute_mie_coeff_lognormal_grid
    from exojax.spec.pardb import PdbCloud

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





