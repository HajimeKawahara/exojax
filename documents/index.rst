.. exojax documentation master file, created by
   sphinx-quickstart on Mon Jan 11 14:38:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`[PDF version] <http://secondearths.sakura.ne.jp/exojax/exojax.pdf>`_
   
ExoJAX
==================================

`ExoJAX <https://github.com/HajimeKawahara/exojax>`_ provides an auto-differentiable high-resolution spectrum model for exoplanets/brown dwarfs using `JAX <https://github.com/google/jax>`_. ExoJAX enables a fully Bayesian inference of the high-dispersion data to fit the line-by-line spectral computation to the observed spectrum, from end-to-end (i.e. from molecular/atomic databases to real spectra), by combining it with `the Hamiltonian Monte Carlo <https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo>`_ in recent probabilistic programming languages such as `NumPyro <https://github.com/pyro-ppl/numpyro>`_. So, the notable features of ExoJAX are summarized as 

- **HMC-NUTS available**
- **Easy to use the latest molecular/atomic data in** :doc:`userguide/exomol`, :doc:`userguide/hitran` **and** :doc:`userguide/atomll`
- **A transparent open-source project; anyone who wants to participate can join the development!**

|:green_circle:| If you have an error and/or want to know the up-to-date info, visit `ExoJAX wiki <https://github.com/HajimeKawahara/exojax/wiki>`_. Or use `the discussions form <https://github.com/HajimeKawahara/exojax/discussions>`_ on github or directly raise `issues <https://github.com/HajimeKawahara/exojax/issues>`_.
  
Contents
==================================

.. toctree::
   :maxdepth: 1
	      
   userguide/installation.rst
   userguide/ql.rst	     
   userguide/database.rst	     

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


References 
---------------------

- Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom, [ApJS 258, 31 (2022)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d) (Paper I)


  
License & Attribution
---------------------

Copyright 2021-2022, Contributors

- `Hajime Kawahara <http://secondearths.sakura.ne.jp/en/index.html>`_ (@HajimeKawahara, maintainer)
- Yui Kawashima (@ykawashima)
- Kento Masuda (@kemasuda)
- Ian Crossfield
- Dirk van den Bekerom (@dcmvdbekerom)
- Daniel Kitzmann (@daniel-kitzmann)
- Brett Morris (@bmorris3)
- Erwan Pannier (@erwanp) and `RADIS <https://github.com/radis/radis>`_ community
- Stevanus Nugroho (@astrostevanus)
- Tako Ishikawa (@chonma0ctopus)

ExoJAX is free software made available under the MIT License. See the ``LICENSE``.
   
