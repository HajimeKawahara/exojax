Tutorials by Use Case
=====================

Start with the spectroscopy mode closest to your problem, then move to
retrieval, opacity, database, or atmosphere-specific examples as needed.

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   Emission spectroscopy <tutorials/get_started>
   Transmission spectroscopy <tutorials/get_started_transmission>
   Reflection spectroscopy <tutorials/get_started_reflection>
   GPU memory-efficient emission spectra <tutorials/get_started_opart>
   Forward modeling overview <various_forward>

Forward Modeling and Radiative Transfer
---------------------------------------

.. toctree::
   :maxdepth: 1

   Pure Absorption Radiative Transfer <tutorials/pure_absorption_rt>
   Rigid stellar rotation and Gaussian convolution <tutorials/Rigid_Rotation>

Retrievals
----------

.. toctree::
   :maxdepth: 1

   HMC-NUTS retrieval with LPF <tutorials/Reverse_modeling>
   HMC-NUTS retrieval with PreMODIT <tutorials/Reverse_modeling_for_methane_using_PreMODIT>
   Methane high-resolution retrieval with PreMODIT <tutorials/reverse_premodit>
   Stochastic variational inference <tutorials/get_started_svi>
   Nested sampling <tutorials/get_started_ns>
   Equilibrium chemistry retrieval <tutorials/equilibrium_chemistry>
   Reverse modeling with precomputed grids <tutorials/reverse_precompute_grid>

|:ringed_planet:| An example of HMC-NUTS for an actual Jupiter reflection spectrum is available in
`exojaxample_jupiter <https://github.com/HajimeKawahara/exojaxample_jupiter>`_.

|:ringed_planet:| An example of HMC-NUTS for an actual hot Saturn transmission spectrum
(JWST/ERS, NIRSpec/G395H) is available in
`exojaxample_WASP39b <https://github.com/sh-tada/exojaxample_WASP39b>`_.

Opacity Methods
---------------

.. toctree::
   :maxdepth: 1

   Wavenumber stitching with OpaStitch <tutorials/Cross_Section_using_OpaStitch>
   Discrete Integral Transform (DIT) <tutorials/Cross_Section_using_Discrete_Integral_Transform>
   Voigt profile <tutorials/voigt_function>
   Voigt-Hjerting function <tutorials/hjerting>
   Correlated-k transmission with ExoMolOP <tutorials/transmission_ckd_exomolop>
   CKD emission with OpaCKD <tutorials/ckd_emispure>
   CKD transmission with OpaCKD <tutorials/ckd_transpure>
   CKD transmission from saved data <tutorials/ckd_transpure_loadonly>

Databases and Line Physics
--------------------------

.. toctree::
   :maxdepth: 1

   Filtering lines by quantum number <tutorials/select_quantum_states>
   R-branch and P-branch of CO <tutorials/branch>
   Fortrat diagram <tutorials/Fortrat>
   CIA opacity <tutorials/CIA_opacity>
   H-minus continuum <tutorials/Hminus>
   Fe I line list from Kurucz <tutorials/Forward_modeling_for_Fe_I_lines_of_Kurucz>
   Chemical abundances with FastChem2 <tutorials/Using_FastChem2_to_calculate_the_chemical_abundances>

Atmospheres, Clouds, and Scattering
-----------------------------------

.. toctree::
   :maxdepth: 1

   Rayleigh scattering <userguide/rayleigh>
   Mie scattering <userguide/mie>
   Jupiter-like ammonia clouds <tutorials/jupiters/Jupiter_cloud_model_using_amp>
   High-resolution Jupiter reflection spectrum <tutorials/jupiters/Jupiter_Hires_Modeling>
   Ackerman and Marley cloud model <tutorials/Ackerman_and_Marley_cloud_model>

Practical Analysis Tips
-----------------------

.. toctree::
   :maxdepth: 1

   Photometry with SopPhoto <tutorials/Photometry>
   Fitting telluric lines <tutorials/Fitting_Telluric_Lines>
   Line identification with ExoMolHR <tutorials/exomolhr>
   Reducing memory for HITEMP CH4 <tutorials/Reducing_memory_for_HITEMP>
   Memory settings <userguide/memorysetting>
   Choosing Elower maximum <tutorials/elower_setting>

Legacy MODIT Tutorials
----------------------

These pages are kept for reference, but they are not the recommended
starting point for current workflows.

.. toctree::
   :maxdepth: 1

   MODIT cross section <tutorials/Cross_Section_using_Modified_Discrete_Integral_Transform>
   HMC-NUTS retrieval with MODIT <tutorials/Reverse_modeling_for_methane_using_MODIT>
   Reverse modeling with VALD using MODIT <tutorials/Reverse_modeling_with_VALD_using_MODIT>
