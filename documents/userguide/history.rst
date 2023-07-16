History
===============

Version History
^^^^^^^^^^^^^^^^^^^^

Version 1.4
-------------------

- transmission spectra #356
- operators on spectra class #363
- multiple molecules handler #388

ExoJAX now strongly recommends using 64-bit as default.


Version 1.3
-------------------

- more GPU memory saved method in PreMODIT (so called diffmode) #332
- opacity calculator class opa #346 see this tutorial
- atmospheric RT class art #346
- optimal Elower max (reducing device memory use) in PreMODIT #331 #332 see this tutorial
- optional quantum states #336 #338 see this tutorial
- faster IP and spin convolutions #350
- molecular mass mdb.molmass available #328

Version 1.2
-------------------

- Moved on a common I/O of molecular/atomic database with radis.api for ExoMol/HITRAN/HITEMP #272
- Removed old moldb for ExoMol/HITRAN/HITEMP
- PreMODIT algorithm, applicable to a wide wavenumber range #265 #287 #288 #307
- Memory saved version of spin rotation and instrumental response #295


Version 1.1
-------------------

- VALD3
- reverse mode available
- compatibility with JAXopt #212 

Version 1.0
-------------------

- Auto-differentiable Spectrum Model of exoplanets/brown dwarfs built on JAX using the molecular/atomic database, ExoMol, HITRAN/HITEMP and VALD3.
- Bayesian inference using HMC-NUTS/NumPyro
- Two opacity calculators available: Direct computation of the Voigt line profile (lpf) and the modified discrete integral transform (MODIT). The latter is a fast opacity calculator for the number of lines N >~ 1000.
- Transparent open-source code with documentations, including a peer-reviewed paper, API, user guide, and tutorials using real data
- HITRAN/CIA and H- as continuous opacity
- Cloud modeling based on Ackerman and Marley
- Quick computation of the opacity and emission spectra for observers (autospec)

Before Version 1
----------------------

