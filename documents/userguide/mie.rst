Mie Scattering
========================




Generates custom miegrid (mgd)
---------------------------------

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





