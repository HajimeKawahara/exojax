Installation
============

Last update: August 2026

ExoJAX requires Python 3.10 or later and is tested on Python 3.10--3.13. Linux,
macOS, and Windows through WSL are the primary supported environments.

Install from PyPI
-----------------

The simplest way to install ExoJAX is from
`PyPI <https://pypi.org/project/exojax/>`_:

.. code:: sh

    pip install exojax

Install from Source
-------------------

To install the development version from the
`GitHub repository <https://github.com/HajimeKawahara/exojax>`_:

.. code:: sh

    git clone https://github.com/HajimeKawahara/exojax.git
    cd exojax
    pip install .

.. note::

    ExoJAX uses a ``pyproject.toml``-based build. ``setup.py install`` is
    deprecated and should not be used.

If an older version of ExoJAX is already installed, uninstall it first to avoid
conflicts with removed modules:

.. code:: sh

    pip uninstall exojax

JAX and Accelerators
--------------------

ExoJAX runs on CPU through JAX. GPU or TPU acceleration requires a JAX
installation that matches your hardware, driver, and CUDA/ROCm environment.

For CPU-only use:

.. code:: sh

    pip install --upgrade jax

For GPU or TPU use, follow the official
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.
JAX packaging and supported accelerator builds change over time, so the
official guide should be treated as the source of truth.

Platform Notes
--------------

- Linux and macOS are supported.
- On Windows, use Windows Subsystem for Linux (WSL). Native Windows is not a
  primary supported environment.
- VALD line lists use the bundled-dependency PyTables cache backend by
  default. The optional ``vaex`` backend requires a separately installed
  version of ``vaex`` that is compatible with your Python environment.

Related Pages
-------------

- :doc:`../tutorials/Differentiable_Programming`
- :doc:`../tutorials/OnDemand_Opacity`
- :doc:`../publications`
