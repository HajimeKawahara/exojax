Multi Molecule Handler
========================

``ExoJAX`` provides building blocks so that calculations can be performed on multiple molecules and in segmented wavelength ranges.
However, we have begun to offer pilot versions of handlers to facilitate handling of multiple molecules and wavelength splitting.


``Multimol`` class handles ``mdb`` and ``opa`` for multiple molecules and wavelength segments, by providing methods called ``multimdb`` and ``multiopa``.

.. code:: ipython3

	from exojax.database.multimol  import MultiMol
    
    mul = MultiMol(molmulti=[["H2O","CH4"],["H2O","CH4"]], 
               dbmulti=[["ExoMol","HITEMP"],["ExoMol","HITEMP"]])

    multimdb = mul.multimdb(nu_grid_list, crit=1.e-30, Ttyp=1000.)    
    multiopa = mul.multiopa_premodit(multimdb, nu_grid_list, auto_trange=[500.,1500.], dit_grid_resolution=0.2)

where ``nu_grid_list`` is a list of the wavenumber grid.

The cross section can be computed using ``multiopa[order][molecule].xsmatrix`` as below.

.. code:: ipython3

    for k in range(len(wavelength_grid_segments)):
        for i in range(len(mul.masked_molmulti[k])):
            xsm = multiopa[k][i].xsmatrix(Tarr, Parr)

We continue to develop this user-friendly multi molecule/segements handler for release 2.2.0. Any feedback is welcome!
