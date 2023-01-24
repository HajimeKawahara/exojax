Filtering by Quantum States (ExoMol)
=====================================

When we want to filter the lines based on quantum states, such as vibration states (v).

.. code:: ipython
	
    >>> from exojax.utils.grids import wavenumber_grid
    >>> from exojax.spec import api
	
    >>> nus, wav, res = wavenumber_grid(24000.0, 26000.0, 1000, unit="AA")
	>>> mdb = api.MdbExomol(""CO/12C-16O/Li2015/"", nus, optional_quantum_states=True, activation=False)

Let's check DataFrame. 

.. code:: ipython
	
    >>> print(mdb.df)

You find these fields are available for Li2015:

- i_upper    i_lower    A          nu_lines      gup    jlower    jupper    elower      v_l    v_u    kp_l    kp_u    Sij0

We would mask the active lines with the condition delta nu = 3.

.. code:: ipython
	
    >>> load_mask = (mdb.df["v_u"] - mdb.df["v_l"] == 3) * mdb.df_load_mask

Then, "activate" the mdb, i.e. making instances (such as mdb.nu_lines ... ), computing broadening parameters etc. 

.. code:: ipython
	
    >>> mdb.activate(mdb.df, load_mask)

