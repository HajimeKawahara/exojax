Non-air broadening
===============================

Originally, HITRAN/HITEMP assumed the terrestrial atmosphere of the Earth. This atmosphere is called "air". 
In contrast, gas giants have hydrogen/helium atmospheres, and rocky planets may have other types of atmospheres.  
The difference in the background atmosphere affects in particular the properties of pressure broadening.
The non-air broadening parameters are provided by the HITRAN team (https://github.com/hitranonline/planetary-broadeners). 
Currently, we have two options for including non-air broadening.

Non-air broadening in MdbHitran
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In MdbHitran we have an option for non-air broadening which uses a built-in option in radis.api.
We can directly use non-air broadening coefficients for some molecules using the "nonair_broadening" option in MdbHitran.

.. code:: ipython
	
	>>> nus, wav, res = wavenumber_grid(22920.0,
                                    23100.0,
                                    100000,
                                    unit='AA',
                                    xsmode="modit")
    >>> mdb = MdbHitran("CO",nus, nonair_broadening=True)
    >>> print(mdb.n_h2)

+-----------------------+-------------+
| background atmosphere | attribute   |
+-----------------------+-------------+
|hydrogen               |n_h2         |
+-----------------------+-------------+
|helium                 |n_he         |
+-----------------------+-------------+
|CO2                    |n_co2        |
+-----------------------+-------------+
|H2O                    |n_h2o        |
+-----------------------+-------------+

Manual calculation of non-air broadening parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ExoJAX has a built-in calculator of the non-air broadening parameters for CO, given the rotational quantum states of the transition. 
The formula is from Y. Tan et al (2022). Here is an example. We use the dataframe to use Jlower and branch (Jupper - Jlower).

.. code:: ipython
	
	>>> from exojax.database.exomol.api import MdbExomol 
	>>> from exojax.utils.grids import wavenumber_grid
	>>> 
	>>> nus, wav, r = wavenumber_grid(24000.0, 26000.0, 1000, unit="AA", xsmode="premodit")
	>>> mdb = MdbHitemp("CO", nus, inherit_dataframe=True)
	>>> 
	>>> # manual non-air broadening
	>>> from exojax.database.molinfo  import m_transition_state
	>>> from exojax.database.nonair  import gamma_nonair, temperature_exponent_nonair
	>>> from exojax.database.nonair  import nonair_coeff_CO_in_H2
	>>> 
	>>> df_mask = mdb.df[mdb.df_load_mask] # wavenumber masking 
	>>> m = m_transition_state(df_mask["jl"],df_mask["branch"])
	>>> n_Texp_H2 = temperature_exponent_nonair(m, nonair_coeff_CO_in_H2).values
	>>> gamma_ref_H2 = gamma_nonair(m, nonair_coeff_CO_in_H2).values


We can also check if the pressure shift can be ignored in terms of velocity as follows.

.. code:: ipython
	
	>>> from exojax.utils.constants import ccgs
	>>> df_mask = mdb.df[mdb.df_load_mask]
	>>> dnu = df_mask["delta_h2"].values/mdb.nu_lines
	>>> maxdv = np.max(dnu * ccgs*1.e-5)
	>>> print("maximum velocity shift by nonair shift = ", maxdv, "km/s")
	>>> # maximum velocity shift by nonair shift =  -0.26374224782584194 km
