Basics
==============

Simple Usage
----------------

.. toctree::
   :maxdepth: 1

   tutorials/simple_usage.rst

Cross Section using Various Opacity Calculators
--------------------------------------------------------

We will update and reorganise the opacity calculators by Release 2.0. 
But meanwhile, use LPF when the number of lines is less than ~ a few hundred.
For larger numbers, consider to use MODIT or a more experimetal calculator PreMODIT. 

.. toctree::
   :maxdepth: 1

   tutorials/opacity.rst
   tutorials/opacity_exomol.rst   
   tutorials/Cross_Section_using_Precomputation_Modified_Discrete_Integral_Transform.rst
   tutorials/Comparing_HITEMP_and_ExoMol.rst
   tutorials/Cross_Section_using_Modified_Discrete_Integral_Transform.rst
   tutorials/Cross_Section_using_Discrete_Integral_Transform.rst

Forward Modeling using Various Opacity Calculators
--------------------------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/Forward_modeling.rst
   tutorials/Forward_modeling_using_DIT.rst
   tutorials/Forward_modeling_using_MODIT.rst
   tutorials/Forward_modeling_using_PreMODIT.rst
   tutorials/Forward_modeling_using_the_MODIT_Cross_Section_for_methane.rst
   tutorials/Forward_modeling_for_Fe_I_lines_of_Kurucz.rst
   tutorials/Forward_modeling_for_metal_line.rst
   tutorials/Forward_modeling_using_the_DIT_Cross_Section_for_methane.rst

Reverse Modeling (a.k.a Retrieval) using Various Opacity Calculators
------------------------------------------------------------------------

.. toctree::
   :maxdepth: 1
   
   tutorials/Reverse_modeling.rst
   tutorials/Reverse_modeling_for_methane_using_MODIT.rst
   tutorials/Reverse_modeling_with_VALD_using_MODIT.rst

Gradient-based Optimization of Spectra
--------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/optimize_spectrum_JAXopt.rst
   tutorials/optimize_voigt.rst
   tutorials/optimize_voigt_JAXopt.rst


Micro Chemical/Physical Processes
--------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/CIA_opacity.rst
   tutorials/branch.rst
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

   tutorials/Reducing_memory_for_HITEMP.rst
   tutorials/Using_FastChem2_to_calculate_the_chemical_abundances.rst
   tutorials/hjerting.rst
   tutorials/pure_absorption_rt.rst
   tutorials/voigt_function.rst
