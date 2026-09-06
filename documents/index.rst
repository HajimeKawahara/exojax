ExoJAX
======

Version 2.5.0 (:doc:`userguide/history`)

`ExoJAX <https://github.com/HajimeKawahara/exojax>`_ is a JAX-based toolkit for
differentiable high-resolution spectral modeling of exoplanets and brown
dwarfs. It provides building blocks for emission, transmission, and reflection
spectra, from opacity data to inference-ready models.

ExoJAX is designed for workflows that combine line-by-line spectral modeling
with optimization and Bayesian inference tools such as
`NumPyro <https://github.com/pyro-ppl/numpyro>`_.

Key Features
------------

- High-resolution emission, transmission, and reflection spectroscopy
- Molecular, atomic, continuum, cloud, and correlated-k opacity workflows
- Differentiable modeling for optimization, HMC-NUTS, SVI, and nested sampling

Start Here
----------

- New users: :doc:`Introduction <introduction>` and :doc:`Tutorials <tutorials>`
- Precomputed opacity and a small first example: :doc:`Starter Opacity Data <opacity_data>`
- Practical examples: :doc:`Gallery <examples/index>`
- Topic-based documentation: :doc:`User Guide <userguide>`
- API reference: :doc:`API Reference <exojax/exojax>`
- External notes: `DeepWiki for ExoJAX <https://deepwiki.com/HajimeKawahara/exojax>`_

.. warning::

   HITEMP access may require HITRAN login credentials. If automated downloads
   fail, see the database-related user guide pages and the
   `RADIS issue note <https://github.com/radis/radis/issues/717>`_.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   introduction.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials.rst
   opacity_data.rst

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

.. toctree::
   :maxdepth: 1
   :caption: Project:

   publications.rst
   credits.rst

References
----------

- Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom,
  `ApJS 258, 31 (2022) <https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d>`_
  (Paper I)
- Kawahara, Kawashima, Tada et al.,
  `ApJ 985, 263 (2025) <https://iopscience.iop.org/article/10.3847/1538-4357/adcba2>`_
  (Paper II)

ExoJAX is free software made available under the MIT License.
