.. exojax documentation master file, created by
   sphinx-quickstart on Mon Jan 11 14:38:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
ExoJAX
==================================

Version 1.6 (:doc:`userguide/history`) 

Note: Paper II will be under peer review. We plan to release version 2.0 at the time of acceptance of the paper II.

`ExoJAX <https://github.com/HajimeKawahara/exojax>`_ provides an auto-differentiable high-resolution spectrum model for exoplanets/brown dwarfs using `JAX <https://github.com/google/jax>`_. 
ExoJAX enables a fully Bayesian inference of the high-dispersion data to fit the line-by-line spectral computation to the observed spectrum, 
from end-to-end (i.e. from molecular/atomic databases to real spectra), 
by combining it with `the Hamiltonian Monte Carlo <https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo>`_ 
in recent probabilistic programming languages such as `NumPyro <https://github.com/pyro-ppl/numpyro>`_. 
So, the notable features of ExoJAX are summarized as 

- **HMC-NUTS, gradient-based optimizer available**
- **Easy to use the latest molecular/atomic data in** :doc:`userguide/api`,  **and** :doc:`userguide/atomll` 
- **A transparent open-source project; anyone who wants to participate can join the development!**

.. admonition:: For a more geek-oriented explanation
   
   ExoJAX is a spectral model based on the `Differentiable Programming (DP) <https://arxiv.org/abs/2403.14606>`_ paradigm!
   ExoJAX aims to provide building blocks for retrieval code, much like Minecraft |:bricks:|.

|:green_circle:| If you have an error and/or want to know the up-to-date info, visit `ExoJAX wiki <https://github.com/HajimeKawahara/exojax/wiki>`_. 
Or use `the discussions form <https://github.com/HajimeKawahara/exojax/discussions>`_ on github or directly raise `issues <https://github.com/HajimeKawahara/exojax/issues>`_.

Contents
==================================

.. toctree::
   :maxdepth: 1
	      
   userguide/installation.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:
	     
   tutorials.rst

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   userguide.rst   

   
.. toctree::
   :maxdepth: 1
   :caption: API:

   exojax/exojax.rst

ExoJAX example (exojaxample)
---------------------------------

- |:ringed_planet:| `exojaxample_WASP39b <https://github.com/sh-tada/exojaxample_WASP39b>`_ : An example of HMC-NUTS for actual hot Saturn (JWST/ERS, NIRSPEC/G395H)

- |:ringed_planet:| `exojaxample_jupiter <https://github.com/HajimeKawahara/exojaxample_jupiter>`_ : An example of HMC-NUTS for actual Jupiter reflection spectrum


References 
---------------------

- Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom,
  `ApJS 258, 31 (2022) <https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d>`_
  (Paper I)


  
License & Attribution
---------------------

Copyright 2021-2023, Contributors

- `Hajime Kawahara <http://secondearths.sakura.ne.jp/en/index.html>`_ (@HajimeKawahara, maintainer)
- `Yui Kawashima <https://sites.google.com/view/yuikawashima/home>`_ (@ykawashima, co-maintainer)
- Kento Masuda (@kemasuda)
- Ian Crossfield
- Dirk van den Bekerom (@dcmvdbekerom)
- Daniel Kitzmann (@daniel-kitzmann)
- Brett Morris (@bmorris3)
- Erwan Pannier (@erwanp) and `RADIS <https://github.com/radis/radis>`_ community
- Stevanus Nugroho (@astrostevanus)
- Tako Ishikawa (@chonma0ctopus)
- Yui Kasagi (@YuiKasagi)
- Shotaro Tada (@sh-tada)

ExoJAX is free software made available under the MIT License. See the ``LICENSE``.
   
