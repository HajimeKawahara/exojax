.. exojax documentation master file, created by
   sphinx-quickstart on Mon Jan 11 14:38:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
ExoJAX
==================================

Version 2.0 (:doc:`userguide/history`) 

|:frog:| `DeepWiki for ExoJAX <https://deepwiki.com/HajimeKawahara/exojax>`_


`ExoJAX <https://github.com/HajimeKawahara/exojax>`_ provides an auto-differentiable high-resolution spectrum model 
for exoplanets/brown dwarfs using `JAX <https://github.com/google/jax>`_. 
It enables fully Bayesian inference for high-dispersion data, fitting line-by-line spectral computations to observed spectra end-to-end
-- from molecular/atomic databases to real spectra -- 
by integrating with Hamiltonian Monte Carlo - No U Turn Sampler (HMC-NUTS), Stochastic Variational Inference (SVI), 
Nested Sampling, and other inference techniques available in modern probabilistic programming frameworks 
such as `NumPyro <https://github.com/pyro-ppl/numpyro>`_.
So, the notable features of ExoJAX are summarized as 

- **HMC-NUTS, SVI, Nested Sampling, Gradient-based Inference Techiques and Optimizers Available**
- **Easy to use the latest molecular/atomic data in** :doc:`userguide/api`,  **and** :doc:`userguide/atomll` 
- **A transparent open-source project; anyone who wants to participate can join the development!**

.. admonition:: For a more geek-oriented explanation
   
   ExoJAX is a spectral model based on the `Differentiable Programming (DP) <https://arxiv.org/abs/2403.14606>`_ paradigm!
   ExoJAX aims to provide building blocks for retrieval code, much like Minecraft |:bricks:|.

|:green_circle:| If you have an error and/or want to know the up-to-date info, visit `ExoJAX wiki <https://github.com/HajimeKawahara/exojax/wiki>`_. 
Or use `the discussions form <https://github.com/HajimeKawahara/exojax/discussions>`_ on github or directly raise `issues <https://github.com/HajimeKawahara/exojax/issues>`_.

.. Warning:: 

	Recently, logging into HITRAN has become mandatory to access HITEMP files, causing issues with automated HITEMP file retrieval by RADIS. For more details, please refer to `this resource <https://github.com/radis/radis/issues/717>`_. (Feb. 3, 2025) 


Contents
==================================

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   introduction.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials.rst

.. toctree::
   :maxdepth: 2
   :caption: Gallery:

   examples/index


.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   userguide.rst   

   
.. toctree::
   :maxdepth: 1
   :caption: API:

   exojax/exojax.rst



Actual Analysis Examples using ExoJAX (external) 
------------------------------------------------------

- |:ringed_planet:| `exojaxample_WASP39b <https://github.com/sh-tada/exojaxample_WASP39b>`_ : An example of HMC-NUTS for actual hot Saturn (JWST/ERS, NIRSPEC/G395H)

- |:ringed_planet:| `AtmosphericRetrieval_HR7672B <https://github.com/YuiKasagi/AtmosphericRetrieval_HR7672B>`_ : An example of HMC-NUTS for a faint companion HR 7672 B by Subaru/REACH.

- |:ringed_planet:| `exojaxample_jupiter <https://github.com/HajimeKawahara/exojaxample_jupiter>`_ : An example of HMC-NUTS for actual Jupiter reflection spectrum

- |:test_tube:| HMC analysis of experimental spectroscopy data for hot methane gas `Gascell_Exojax. <https://github.com/KoHosokawa/Gascell_Exojax>`_

- |:page_facing_up:| Chromatic Transit Variation for WASP-39b `Tada et al. <https://arxiv.org/abs/2503.08988>`_ (arXiv)

- |:page_facing_up:| HMC-NUTS for Gl 229 B  (T-dwarf) Emission Spectrum `Kawashima et al. <https://arxiv.org/abs/2410.11561>`_ (arXiv)


References 
---------------------

- |:page_facing_up:|  Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom, `ApJS 258, 31 (2022) <https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d>`_ (Paper I)

- |:page_facing_up:| Kawahara, Kawashima, Tada et al., `ApJ 985, 263 (2025) <https://iopscience.iop.org/article/10.3847/1538-4357/adcba2>`_, (Paper II)

License & Attribution
---------------------

Copyright 2021-2025, Contributors

- `Hajime Kawahara <http://secondearths.sakura.ne.jp/en/index.html>`_ (@HajimeKawahara, maintainer)
- `Yui Kawashima <https://sites.google.com/view/yuikawashima/home>`_ (@ykawashima, co-maintainer)
- Shotaro Tada (@sh-tada)
- Yui Kasagi (@YuiKasagi)
- Kento Masuda (@kemasuda)
- Tako Ishikawa (@chonma0ctopus)
- Ian Crossfield
- Dirk van den Bekerom (@dcmvdbekerom)
- Daniel Kitzmann (@daniel-kitzmann)
- Brett Morris (@bmorris3)
- Erwan Pannier (@erwanp) and Nicolas Minesi (@minouHub) from `RADIS <https://github.com/radis/radis>`_ community
- Stevanus Nugroho (@astrostevanus)
- Ko Hosokawa (@KoHosokawa)
- Hibiki Yama 

ExoJAX is free software made available under the MIT License. See the ``LICENSE``.
   
