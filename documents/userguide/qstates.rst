Quantum States Filtering (ExoMol) 
=====================================

When we would like to filter the lines based on quantum states, such as vibration states (v), we can mask the lines by manual. To do so, we do not activate mdb when initialization. Also, we need to load the optional quantum states. Here is the example for the initialization. 

.. code:: ipython
	
    >>> from exojax.utils.grids import wavenumber_grid
    >>> from exojax.spec import api
	
    >>> nus, wav, res = wavenumber_grid(24000.0, 26000.0, 1000, unit="AA")
	>>> mdb = api.MdbExomol(""CO/12C-16O/Li2015/"", nus, optional_quantum_states=True, activation=False)

Then, let's check DataFrame. 

.. code:: ipython
	
    >>> print(mdb.df)

You find the following fields are available for Li2015:

- i_upper    i_lower    A          nu_lines      gup    jlower    jupper    elower      v_l    v_u    kp_l    kp_u    Sij0

For instance, v_l means the rotational quantum number (nu) for the lower state, v_u the upper state. 
We would use the lines with the condition delta v = 3. Make the mask using DataFrame.

.. code:: ipython
	
    >>> mask = (mdb.df["v_u"] - mdb.df["v_l"] == 3) 

Activate the mdb with the mask we made. The activation includes making the instances (such as mdb.nu_lines ... ), computing broadening parameters etc.  

.. code:: ipython
	
    >>> mdb.activate(mdb.df, mask)

Then, we can use mdb as usual. This is a plot of the activated lines and all of the lines in DataFrame.
    
.. image:: qstates/COdv.png
