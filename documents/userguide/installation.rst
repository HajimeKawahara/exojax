Installation and References
----------------------------------

*last update January 19th (2025) Hajime Kawahara, updated for pyproject.toml build system*

.. admonition:: Python 3.9 or later is required

    ExoJAX requires Python 3.9 or later.

.. warning::

    For using `vaex <https://github.com/vaexio/vaex>`_ in the common API for molecular database I/O, we currently recommend using Python 3.9 or 3.10.

Linux, Windows WSL, macOS
=========================

The simplest way to install ExoJAX is from `PyPI <https://pypi.org/project/exojax/>`_:

.. code:: sh

    pip install exojax

Alternatively, clone the code from the `GitHub repository <https://github.com/HajimeKawahara/exojax>`_ and install locally:

.. code:: sh

    git clone https://github.com/HajimeKawahara/exojax.git
    cd exojax
    pip install .

.. note::

    This project now uses a ``pyproject.toml``-based build.  
    ``setup.py install`` is deprecated and should not be used.

If you have an older version of ExoJAX already installed, it is recommended to uninstall it first to avoid conflicts with removed modules:

.. code:: sh

    pip uninstall exojax

JAX and GPU Support
===================

To take advantage of GPU acceleration, you need a compatible GPU and to install the appropriate ``jaxlib`` build for your CUDA or ROCm version.

First, check your CUDA version (if using NVIDIA GPUs):

.. code:: sh

    nvcc -V

Then install JAX with GPU support as described in the official  
`JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.

Example for CUDA 12:

.. code:: sh

    pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Example for CUDA 11:

.. code:: sh

    pip install --upgrade "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

If you do not have a GPU, you can simply install the CPU version:

.. code:: sh

    pip install --upgrade jax



Windows Anaconda
===================

Not supported yet.
		

		
References
=================

|:green_circle:| **ExoJAX Paper I**:  Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom (2021) accepted by ApJS: `arXiv:2105.14782 <http://arxiv.org/abs/2105.14782>`_

|:green_circle:| **ExoJAX Paper II**:  Kawahara, Kawashima, Tada et al: `arXiv:2410.06900 <http://arxiv.org/abs/2410.06900>`_


Many techniques/databases are used in ExoJAX.

- JAX: Bradbury, J., Frostig, R., Hawkins, P., et al. 2018, JAX: composable transformations of Python+NumPy programs, `JAX <http://github.com/google/jax>`_
- NumPyro: Phan, D., Pradhan, N., & Jankowiak, M. 2019, `arXiv:1912.11554 <http://arxiv.org/abs/1912.11554>`_
- JAXopt: Blondel, M., Berthet, Q., Cuturi, M. et al. 2021 `arXiv:2105.15183 <http://arxiv.org/abs/2105.15183>`_
- `Optax <https://optax.readthedocs.io/en/latest/>`_
- Vaex: Breddels and Veljanoski (2018) `arXiv:https://arxiv.org/abs/1801.02638 <https://arxiv.org/abs/1801.02638>`_
- Algorithm 916: Zaghloul and Ali (2012) `arXiv:1106.0151 <https://arxiv.org/abs/1106.0151>`_
- DIT: `van den Bekerom and Pannier (2021) <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ 
- ExoMol: `Tennyson et al. (2016) <https://www.sciencedirect.com/science/article/abs/pii/S0022285216300807?via%3Dihub>`_
- HITRAN/HITEMP
- VALD3
- VIRGA for refractive indices of condensates
- PyMieScatt for Mie scattering
- Flux-adding treatment by `Robinson and Crisp (2018) <https://www.sciencedirect.com/science/article/pii/S0022407317305101?via%3Dihub>`_
- RADIS, see below.
- Other many packages/algorithms. See `arXiv:2105.14782 <http://arxiv.org/abs/2105.14782>`_ and a forthcoming paper (Kawahara, Kawashima et al.) for the details.


Related Projects
=====================

- `RADIS <https://github.com/radis/radis>`_

| ExoJAX draws a lot of inspiration from a fast line-by-line code for high-resolution infrared molecular spectra `RADIS <https://github.com/radis/radis>`_, including DIT, the use of Vaex, and so on. 
| Since version 1.2 we have been using a common molecular database I/O API in Radis.

- `REACH <http://secondearths.sakura.ne.jp/reach/>`_

| ExoJAX was originally developed to interpret data from a new high-dispersion coronagraphic capability at the Subaru telescope, the `REACH <http://secondearths.sakura.ne.jp/reach/>`_ project (SCExAO+IRD). REACH is supported by `RESCEU <http://www.resceu.s.u-tokyo.ac.jp/top.php>`_, ABC and `JSPS KAKENHI JP20H00170 <https://kaken.nii.ac.jp/en/grant/KAKENHI-PROJECT-20H00170/>`_ (Kawahara). See also `Lozi et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018SPIE10703E..59L/abstract>`_ for SCExAO, `Kotani et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018SPIE10702E..11K/abstract>`_ for IRD, `Jovanovic et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017arXiv171207762J/abstract>`_ for post-coronagraphic injection, and `Kawahara et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJS..212...27K/abstract>`_ for high dispersion coronagraphy.
