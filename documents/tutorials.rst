Basics
==============

.. note::

   We will update and reorganise the opacity calculators by Release 2.0. 
   Meanwhile, use LPF when the number of lines is less than ~ a few hundred.
   For larger numbers, consider to use MODIT or a more experimetal calculator PreMODIT.
   MODIT is limited by a device memory limits while while PreMODIT has no such limitation.
   Therefore, if you want to for instance, million to billion lines, use PreMODIT.

Get Started
----------------

.. toctree::
   :maxdepth: 1

   tutorials/get_started.rst

Optimization
=================

Gradient-based Optimization of Spectra
--------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/optimize_spectrum_JAXopt.rst
   tutorials/optimize_voigt.rst
   tutorials/optimize_voigt_JAXopt.rst




Retrievals
=======================

Reverse Modeling (a.k.a Retrieval) using Various Opacity Calculators
------------------------------------------------------------------------

.. toctree::
   :maxdepth: 1
   
   tutorials/Reverse_modeling.rst
   tutorials/Reverse_modeling_for_methane_using_MODIT.rst
   tutorials/Reverse_modeling_with_VALD_using_MODIT.rst


Examples for Manual Settings of Opacity
==============================================

If you like, you do not need "opa" nor "art" to compute opacity and spectra. 

Cross Section using Various Opacity Calculators
--------------------------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/opacity.rst
   tutorials/opacity_exomol.rst   
   tutorials/Cross_Section_using_Precomputation_Modified_Discrete_Integral_Transform.rst
   tutorials/Comparing_HITEMP_and_ExoMol.rst
   tutorials/Cross_Section_using_Modified_Discrete_Integral_Transform.rst
   tutorials/Cross_Section_using_Discrete_Integral_Transform.rst

Manual Forward Modeling
--------------------------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/Forward_modeling.rst
   tutorials/Forward_modeling_using_PreMODIT_Cross_Section_for_methane.rst
   various_forward.rst



Others
================

Micro Chemical/Physical Processes
--------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/CIA_opacity.rst
   tutorials/branch.rst
   tutorials/Fortrat.rst
   tutorials/Ackerman_and_Marley_cloud_model.rst
   tutorials/Terminal_Velocity_of_Cloud_Particles.rst   
   
Macro Physical Processes
--------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/Rigid_Rotation.rst


Others
------------------

.. toctree::
   :maxdepth: 1

   tutorials/Using_FastChem2_to_calculate_the_chemical_abundances.rst
   tutorials/hjerting.rst
   tutorials/pure_absorption_rt.rst
   tutorials/voigt_function.rst


